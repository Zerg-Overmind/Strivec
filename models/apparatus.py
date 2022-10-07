import torch
import torch.nn
import torch.nn.functional as F
import numpy as np
import time
import os
# from torch_scatter import segment_coo
from torch_scatter import segment_coo
from torch.utils.cpp_extension import load
from plyfile import PlyData, PlyElement


parent_dir = os.path.dirname(os.path.abspath(__file__))
render_utils_cuda = load(
    name='render_utils_cuda',
    sources=[
        os.path.join(parent_dir, path)
        for path in ['cuda/render_utils.cpp', 'cuda/render_utils_kernel.cu']],
    verbose=True)


search_geo_cuda = load(
    name='search_geo_cuda',
    sources=[
        os.path.join(parent_dir, path)
        for path in ['cuda/search_geo.cpp', 'cuda/search_geo.cu']],
    verbose=True)


search_geo_hier_cuda = load(
    name='search_geo_hier_cuda',
    sources=[
        os.path.join(parent_dir, path)
        for path in ['cuda/search_geo_hier.cpp', 'cuda/search_geo_hier.cu']],
    verbose=True)


def positional_encoding(positions, freqs):
    freq_bands = (2 ** torch.arange(freqs).float()).to(positions.device)  # (F,)
    pts = (positions[..., None] * freq_bands).reshape(
        positions.shape[:-1] + (freqs * positions.shape[-1],))  # (..., DF)
    pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
    return pts


def raw2alpha(sigma, dist):
    # sigma, dist  [N_rays, N_samples]
    alpha = 1. - torch.exp(-sigma * dist)

    T = torch.cumprod(torch.cat([torch.ones(alpha.shape[0], 1).to(alpha.device), 1. - alpha + 1e-10], -1), -1)

    weights = alpha * T[:, :-1]  # [N_rays, N_samples]
    return alpha, weights, T[:, -1:]


def SHRender(xyz_sampled, viewdirs, features):
    sh_mult = eval_sh_bases(2, viewdirs)[:, None]
    rgb_sh = features.view(-1, 3, sh_mult.shape[-1])
    rgb = torch.relu(torch.sum(sh_mult * rgb_sh, dim=-1) + 0.5)
    return rgb


def RGBRender(xyz_sampled, viewdirs, features):
    rgb = features
    return rgb


def filter_ray_by_points(xyz_sampled, geo, half_range):
    # xyz_dist = torch.abs(
    #     xyz_sampled[..., None, :] - geo[None, None, ..., :3])  # chunksize * raysampleN * 4096 * 3
    # mask_inrange = torch.all(xyz_dist <= self.local_range[None, None, None, :],
    #                          dim=-1)  # chunksize * raysampleN * 4096
    # mask_inrange = torch.any(mask_inrange.view(mask_inrange.shape[0], -1), dim=-1)
    mask_inrange = render_utils_cuda.filter_ray_by_points(xyz_sampled, geo.contiguous(), half_range) > 0
    return mask_inrange

def filter_ray_by_projection(rays_o, rays_d, geo, half_range):
    # xyz_dist = torch.abs(
    #     xyz_sampled[..., None, :] - geo[None, None, ..., :3])  # chunksize * raysampleN * 4096 * 3
    # mask_inrange = torch.all(xyz_dist <= self.local_range[None, None, None, :],
    #                          dim=-1)  # chunksize * raysampleN * 4096
    # mask_inrange = torch.any(mask_inrange.view(mask_inrange.shape[0], -1), dim=-1)
    # rays_d has to be unit
    tensoRF_per_ray = render_utils_cuda.filter_ray_by_projection(rays_o.contiguous(), rays_d.contiguous(), geo.contiguous(), torch.square(half_range))
    return tensoRF_per_ray

def filter_ray_by_cvrg(xyz_sampled,mask_inbox, units, xyz_min, xyz_max, tensoRF_cvrg_mask):
    # xyz_dist = torch.abs(
    #     xyz_sampled[..., None, :] - geo[None, None, ..., :3])  # chunksize * raysampleN * 4096 * 3
    # mask_inrange = torch.all(xyz_dist <= self.local_range[None, None, None, :],
    #                          dim=-1)  # chunksize * raysampleN * 4096
    # mask_inrange = torch.any(mask_inrange.view(mask_inrange.shape[0], -1), dim=-1)
    # rays_d has to be unit
    tensoRF_per_ray = search_geo_cuda.filter_ray_by_cvrg(xyz_sampled.contiguous(), mask_inbox.contiguous(), units.contiguous(), xyz_min.contiguous(), xyz_max.contiguous(), tensoRF_cvrg_mask)
    return tensoRF_per_ray

class AlphaGridMask(torch.nn.Module):
    def __init__(self, device, aabb, alpha_volume, mask_cache_thres=None):
        super(AlphaGridMask, self).__init__()
        self.device = device

        self.aabb = aabb.to(self.device)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invgridSize = 1.0 / self.aabbSize * 2
        self.alpha_volume = alpha_volume.view(1, 1, *alpha_volume.shape[-3:])
        self.gridSize = torch.LongTensor([alpha_volume.shape[-1], alpha_volume.shape[-2], alpha_volume.shape[-3]]).to(self.device)
        # mask = (self.alpha_volume >= mask_cache_thres).squeeze(0).squeeze(0)
        # self.register_buffer('mask', mask)
        # self.register_buffer('xyz2ijk_scale', (self.gridSize - 1) / self.aabbSize)
        # self.register_buffer('xyz2ijk_shift', -self.aabbSize[0] * self.xyz2ijk_scale)

    @torch.no_grad()
    def sample_alpha(self, xyz_sampled):
        xyz_sampled = self.normalize_coord(xyz_sampled)
        alpha_vals = F.grid_sample(self.alpha_volume, xyz_sampled.view(1, -1, 1, 1, 3), align_corners=True).view(-1)

        return alpha_vals

    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled - self.aabb[0]) * self.invgridSize - 1


class MLPRender_Fea(torch.nn.Module):
    def __init__(self, inChanel, viewpe=6, feape=6, featureC=128):
        super(MLPRender_Fea, self).__init__()
        if isinstance(inChanel, list):
            inChanel = sum(inChanel)
        self.in_mlpC = 2 * viewpe * 3 + 2 * feape * inChanel + 3 + inChanel
        self.viewpe = viewpe
        self.feape = feape
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC, 3)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb


class MLPRender_PE(torch.nn.Module):
    def __init__(self, inChanel, viewpe=6, pospe=6, featureC=128):
        super(MLPRender_PE, self).__init__()

        self.in_mlpC = (3 + 2 * viewpe * 3) + (3 + 2 * pospe * 3) + inChanel  #
        self.viewpe = viewpe
        self.pospe = pospe
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC, 3)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.pospe > 0:
            indata += [positional_encoding(pts, self.pospe)]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb


class MLPRender(torch.nn.Module):
    def __init__(self, inChanel, viewpe=6, featureC=128):
        super(MLPRender, self).__init__()

        self.in_mlpC = (3 + 2 * viewpe * 3) + inChanel
        self.viewpe = viewpe

        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC, 3)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb


def draw_box(center_xyz, local_range, log, step, rot_m=None):
    sx, sy, sz = local_range[0], local_range[1], local_range[2]
    shift = torch.as_tensor([[sx, sy, sz],
                             [-sx, sy, sz],
                             [sx, -sy, sz],
                             [sx, sy, -sz],
                             [sx, -sy, -sz],
                             [-sx, -sy, sz],
                             [-sx, sy, -sz],
                             [-sx, -sy, -sz],
                             ], dtype=center_xyz.dtype, device=center_xyz.device)[None, ...]
                             
    corner_xyz = center_xyz[..., None, :] + (torch.matmul(shift, rot_m) if rot_m is not None else shift)

    corner_xyz = corner_xyz.cpu().detach().numpy().reshape(-1, 3)

    vertex = np.array([(corner_xyz[i,0], corner_xyz[i,1], corner_xyz[i,2]) for i in range(len(corner_xyz))],
                         dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
   
    edge = np.array([(8 * a + 0, 8 * a + 1, 255, 165, 0) for a in range(len(center_xyz))] +
                     [(8 * b + 1, 8 * b + 5, 255, 165, 0) for b in range(len(center_xyz))] +
                     [(8 * c + 5, 8 * c + 2, 255, 165, 0) for c in range(len(center_xyz))] +
                     [(8 * d + 2, 8 * d + 0, 255, 165, 0) for d in range(len(center_xyz))] +
                     [(8 * e + 6, 8 * e + 7, 0  , 255, 0) for e in range(len(center_xyz))] +
                     [(8 * f + 7, 8 * f + 4, 255, 0,   0) for f in range(len(center_xyz))] +
                     [(8 * g + 4, 8 * g + 3, 255, 165, 0) for g in range(len(center_xyz))] +
                     [(8 * h + 3, 8 * h + 6, 255, 165, 0) for h in range(len(center_xyz))] +
                     [(8 * i + 1, 8 * i + 6, 255, 165, 0) for i in range(len(center_xyz))] +
                     [(8 * j + 5, 8 * j + 7, 0,   0, 255) for j in range(len(center_xyz))] +
                     [(8 * k + 2, 8 * k + 4, 255, 165, 0) for k in range(len(center_xyz))] +
                     [(8 * l + 0, 8 * l + 3, 255, 165, 0) for l in range(len(center_xyz))]
                     ,dtype = [('vertex1', 'i4'),('vertex2', 'i4'),
                     ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    ver = PlyElement.describe(vertex, 'vertex')
    edg = PlyElement.describe(edge, 'edge')
    os.makedirs('{}/rot_tensoRF'.format(log), exist_ok=True)
    with open('{}/rot_tensoRF/{}.ply'.format(log, step), mode='wb') as f:
        PlyData([ver, edg], text=True).write(f)

def draw_box_pca(center_xyz, pca_cluster, local_range, log, step, args, rot_m=None):
    sx, sy, sz = local_range[0], local_range[1], local_range[2]
    shift = torch.as_tensor([[sx, sy, sz],
                             [-sx, sy, sz],
                             [sx, -sy, sz],
                             [sx, sy, -sz],
                             [sx, -sy, -sz],
                             [-sx, -sy, sz],
                             [-sx, sy, -sz],
                             [-sx, -sy, -sz],
                             ], dtype=center_xyz.dtype, device=center_xyz.device)[None, ...]
    corner_xyz = center_xyz[..., None, :] + (torch.matmul(shift, rot_m) if rot_m is not None else shift)

    corner_xyz = corner_xyz.cpu().detach().numpy().reshape(-1, 3)

    vertex = np.array([(corner_xyz[i,0], corner_xyz[i,1], corner_xyz[i,2]) for i in range(len(corner_xyz))],
                         dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    
    edge = np.array([(8 * a + 0, 8 * a + 1, 255, 165, 0) for a in range(len(center_xyz))] +
                     [(8 * b + 1, 8 * b + 5, 255, 165, 0) for b in range(len(center_xyz))] +
                     [(8 * c + 5, 8 * c + 2, 255, 165, 0) for c in range(len(center_xyz))] +
                     [(8 * d + 2, 8 * d + 0, 255, 165, 0) for d in range(len(center_xyz))] +
                     [(8 * e + 6, 8 * e + 7, 0  , 255, 0) for e in range(len(center_xyz))] +
                     [(8 * f + 7, 8 * f + 4, 255, 0,   0) for f in range(len(center_xyz))] +
                     [(8 * g + 4, 8 * g + 3, 255, 165, 0) for g in range(len(center_xyz))] +
                     [(8 * h + 3, 8 * h + 6, 255, 165, 0) for h in range(len(center_xyz))] +
                     [(8 * i + 1, 8 * i + 6, 255, 165, 0) for i in range(len(center_xyz))] +
                     [(8 * j + 5, 8 * j + 7, 0,   0, 255) for j in range(len(center_xyz))] +
                     [(8 * k + 2, 8 * k + 4, 255, 165, 0) for k in range(len(center_xyz))] +
                     [(8 * l + 0, 8 * l + 3, 255, 165, 0) for l in range(len(center_xyz))]
                     ,dtype = [('vertex1', 'i4'),('vertex2', 'i4'),
                     ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    np.savetxt(args.pointfile[:-4] + "_{}_{}_vox_tensorfs".format(args.datadir.split("/")[-1], args.vox_range[step][0]) + ".txt", center_xyz.cpu().numpy(), delimiter=";")
    for k_cl in range(len(pca_cluster)):
       if len(pca_cluster[k_cl])>0:
         with open(args.pointfile[:-4] + "_{}_{}_vox_pca".format(args.datadir.split("/")[-1], args.vox_range[step][0]) + ".txt", 'a+') as ff:
           np.savetxt(ff, pca_cluster[k_cl][0], delimiter=";")
       
    ver = PlyElement.describe(vertex, 'vertex')
    edg = PlyElement.describe(edge, 'edge')
    os.makedirs('{}/rot_tensoRF'.format(log), exist_ok=True)
    with open('{}/rot_tensoRF/{}_pca.ply'.format(log, step), mode='wb') as f:
        PlyData([ver, edg], text=True).write(f)


''' Misc
'''
#
class Raw2Alpha(torch.autograd.Function):
    @staticmethod
    def forward(ctx, density, shift, interval):
        '''
        alpha = 1 - exp(-softplus(density + shift) * interval)
              = 1 - exp(-log(1 + exp(density + shift)) * interval)
              = 1 - exp(log(1 + exp(density + shift)) ^ (-interval))
              = 1 - (1 + exp(density + shift)) ^ (-interval)
        '''
        exp, alpha = render_utils_cuda.raw2alpha(density, shift, interval);
        if density.requires_grad:
            ctx.save_for_backward(exp)
            ctx.interval = interval
        return alpha

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_back):
        '''
        alpha' = interval * ((1 + exp(density + shift)) ^ (-interval-1)) * exp(density + shift)'
               = interval * ((1 + exp(density + shift)) ^ (-interval-1)) * exp(density + shift)
        '''
        exp = ctx.saved_tensors[0]
        interval = ctx.interval
        return render_utils_cuda.raw2alpha_backward(exp, grad_back.contiguous(), interval), None, None


class Raw2Alpha_randstep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, density, shift, interval):
        '''
        alpha = 1 - exp(-softplus(density + shift) * interval)
              = 1 - exp(-log(1 + exp(density + shift)) * interval)
              = 1 - exp(log(1 + exp(density + shift)) ^ (-interval))
              = 1 - (1 + exp(density + shift)) ^ (-interval)
        '''
        exp, alpha = render_utils_cuda.raw2alpha_randstep(density, shift, interval);
        if density.requires_grad:
            ctx.save_for_backward(exp)
            ctx.interval = interval
        return alpha

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_back):
        '''
        alpha' = interval * ((1 + exp(density + shift)) ^ (-interval-1)) * exp(density + shift)'
               = interval * ((1 + exp(density + shift)) ^ (-interval-1)) * exp(density + shift)
        '''
        exp = ctx.saved_tensors[0]
        interval = ctx.interval
        return render_utils_cuda.raw2alpha_randstep_backward(exp, grad_back.contiguous(), interval), None, None



def raw2alpha_only(sigma, dist):
    # sigma, dist  [N_rays, N_samples]
    alpha = 1. - torch.exp(-sigma*dist)
    return alpha

def grid_xyz(center, local_range, local_dims):
    xs = torch.linspace(center[0]-local_range[0], center[0]+local_range[0], steps=local_dims[0]+1,
                        device=local_range.device,  dtype=local_range.dtype)
    ys = torch.linspace(center[1]-local_range[1], center[1]+local_range[1], steps=local_dims[1]+1,
                        device=local_range.device,  dtype=local_range.dtype)
    zs = torch.linspace(center[2]-local_range[2], center[2]+local_range[2], steps=local_dims[2]+1,
                        device=local_range.device,  dtype=local_range.dtype)
    xx = xs.view(-1, 1, 1).repeat(1, len(ys), len(zs))
    yy = ys.view(1, -1, 1).repeat(len(xs), 1, len(zs))
    zz = zs.view(1, 1, -1).repeat(len(xs), len(ys), 1)
    return torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)

class Alphas2Weights(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alpha, ray_id, N):
        weights, T, alphainv_last, i_start, i_end = render_utils_cuda.alpha2weight(alpha, ray_id, N)
        if alpha.requires_grad:
            ctx.save_for_backward(alpha, weights, T, alphainv_last, i_start, i_end)
            ctx.n_rays = N
        return weights, alphainv_last

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_weights, grad_last):
        alpha, weights, T, alphainv_last, i_start, i_end = ctx.saved_tensors
        grad = render_utils_cuda.alpha2weight_backward(
            alpha, weights, T, alphainv_last,
            i_start, i_end, ctx.n_rays, grad_weights, grad_last)
        return grad, None, None


def randomize_ray(rays_o, rgb_train, alpha, ijs, c2ws, focal, cent):
    b, _ = rays_o.shape
    xyshift = torch.rand(b, 1, 1, 2, device=rgb_train.device)
    inds = torch.round(xyshift).long()
    revers_mask = alpha[torch.arange(b, dtype=torch.int64, device=rgb_train.device), inds[:, 0, 0, 0], inds[:, 0, 0, 1]]
    # print("revers_mask", revers_mask.shape, torch.sum(revers_mask))
    xyshift[revers_mask] = 0.5
    xyshift = xyshift * 2 - 1
    rgbs = torch.nn.functional.grid_sample(rgb_train, xyshift, mode='bilinear', align_corners=True)[..., 0, 0]
    # print("rgbs", rgbs.shape)
    inds = ijs + xyshift[:,0,0,:]
    directions = torch.stack([(inds[...,0] - cent[0]) / focal, (inds[...,1] - cent[1]) / focal, torch.ones_like(inds[...,0])], -1)  # (H, W, 3)
    directions /= torch.norm(directions, dim=-1, keepdim=True)
    rays_d = torch.matmul(directions[:,None,:], c2ws).squeeze(1)
    return torch.cat([rays_o, rays_d], dim=1), rgbs
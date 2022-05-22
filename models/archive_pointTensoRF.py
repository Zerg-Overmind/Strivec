import torch
import torch.nn
import torch.nn.functional as F
from .sh import eval_sh_bases
import numpy as np
import time
import os
from torch_scatter import segment_coo
from .apparatus import *
from .tensorBase import TensorBase
from .pointTensoRF import PointTensorBase
from tqdm import tqdm

class PointTensorVMSplit(PointTensorBase):
    def __init__(self, aabb, gridSize, device, **kargs):
        super(PointTensorVMSplit, self).__init__(aabb, gridSize, device, **kargs)


    def init_svd_volume(self, local_dims, device):
        self.density_plane, self.density_line = self.init_one_svd(self.density_n_comp, local_dims[:3], 0.1, device)
        self.app_plane, self.app_line = self.init_one_svd(self.app_n_comp, local_dims[:3], 0.1, device)
        self.basis_mat = torch.nn.Linear(sum(self.app_n_comp), self.app_dim, bias=False).to(device)


    def init_one_svd(self, n_component, local_dims, scale, device):
        plane_coef, line_coef = [], []
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef.append(torch.nn.Parameter(
                scale * torch.randn((len(self.geo), n_component[i], local_dims[mat_id_1]+1, local_dims[mat_id_0]+1))))  #
            line_coef.append(
                torch.nn.Parameter(scale * torch.randn((len(self.geo), n_component[i], local_dims[vec_id]+1))))

        return torch.nn.ParameterList(plane_coef).to(device), torch.nn.ParameterList(line_coef).to(device)
    
    
    def get_optparam_groups(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001, skip_zero_grad = True):
        grad_vars = [{'params': self.density_line, 'lr': lr_init_spatialxyz, 'skip_zero_grad': (skip_zero_grad)}, {'params': self.density_plane, 'lr': lr_init_spatialxyz, 'skip_zero_grad': (skip_zero_grad)},
                     {'params': self.app_line, 'lr': lr_init_spatialxyz, 'skip_zero_grad': (skip_zero_grad)}, {'params': self.app_plane, 'lr': lr_init_spatialxyz, 'skip_zero_grad': (skip_zero_grad)},
                         {'params': self.basis_mat.parameters(), 'lr':lr_init_network, 'skip_zero_grad': (False)}]
        if isinstance(self.renderModule, torch.nn.Module):
            grad_vars += [{'params':self.renderModule.parameters(), 'lr':lr_init_network, 'skip_zero_grad': (False)}]
        return grad_vars


    def vectorDiffs(self, vector_comps):
        total = 0
        
        for idx in range(len(vector_comps)):
            n_comp, n_size = vector_comps[idx].shape[1:-1]
            
            dotp = torch.matmul(vector_comps[idx].view(n_comp,n_size), vector_comps[idx].view(n_comp,n_size).transpose(-1,-2))
            non_diagonal = dotp.view(-1)[1:].view(n_comp-1, n_comp+1)[...,:-1]
            total = total + torch.mean(torch.abs(non_diagonal))
        return total

    def vector_comp_diffs(self):
        return self.vectorDiffs(self.density_line) + self.vectorDiffs(self.app_line)
    
    def density_L1(self):
        total = 0
        for idx in range(len(self.density_plane)):
            total = total + torch.mean(torch.abs(self.density_plane[idx])) + torch.mean(torch.abs(self.density_line[idx]))# + torch.mean(torch.abs(self.app_plane[idx])) + torch.mean(torch.abs(self.density_plane[idx]))
        return total
    
    def TV_loss_density(self, reg):
        total = 0
        for idx in range(len(self.density_plane)):
            total = total + reg(self.density_plane[idx]) * 1e-2 + reg(self.density_line[idx]) * 1e-3
        return total
        
    def TV_loss_app(self, reg):
        total = 0
        for idx in range(len(self.app_plane)):
            total = total + reg(self.app_plane[idx]) * 1e-2 + reg(self.app_line[idx]) * 1e-3
        return total


    def ind_intrp_plane_map(self, matMode, density_plane_cur, local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, tensoRF_id):
        plane_nw = density_plane_cur[tensoRF_id, :, local_gindx_s[..., matMode[0]],
                   local_gindx_s[..., matMode[1]]]
        plane_se = density_plane_cur[tensoRF_id, :, local_gindx_l[..., matMode[0]],
                   local_gindx_l[..., matMode[1]]]
        plane_ne = density_plane_cur[tensoRF_id, :, local_gindx_s[..., matMode[0]],
                   local_gindx_l[..., matMode[1]]]
        plane_sw = density_plane_cur[tensoRF_id, :, local_gindx_l[..., matMode[0]],
                   local_gindx_s[..., matMode[1]]]

        plane_nw_gweight = local_gweight_s[..., matMode[0]] * local_gweight_s[
            ..., matMode[1]]
        plane_se_gweight = local_gweight_l[..., matMode[0]] * local_gweight_l[
            ..., matMode[1]]
        plane_ne_gweight = local_gweight_s[..., matMode[0]] * local_gweight_l[
            ..., matMode[1]]
        plane_sw_gweight = local_gweight_l[..., matMode[0]] * local_gweight_s[
            ..., matMode[1]]

        return plane_nw * plane_nw_gweight[:, None, ...] + plane_se * plane_se_gweight[:, None, ...] + plane_ne * plane_ne_gweight[:, None, ...] + plane_sw * plane_sw_gweight[:, None, ...]


    def ind_intrp_line_map(self, vecMode, density_line_cur, local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, tensoRF_id):

        line_s = density_line_cur[tensoRF_id, :, local_gindx_s[..., vecMode]]
        line_l = density_line_cur[tensoRF_id, :, local_gindx_l[..., vecMode]]

        line_s_gweight = local_gweight_s[:, None, vecMode]
        line_l_gweight = local_gweight_l[:, None, vecMode]

        return line_s * line_s_gweight + line_l * line_l_gweight

    def compute_densityfeature_geo(self, local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, local_kernel_dist, tensoRF_id, agg_id):
        # plane + line basis

        sigma_feature = torch.zeros((local_gweight_s.shape[0],), device=local_gweight_s.device, dtype=local_gweight_s.dtype)

        for idx_plane in range(len(self.density_plane)):
            plane_coef_point = self.ind_intrp_plane_map(self.matMode[idx_plane], self.density_plane[idx_plane], local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, tensoRF_id)

            line_coef_point = self.ind_intrp_line_map(self.vecMode[idx_plane], self.density_line[idx_plane], local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, tensoRF_id)
            sigma_feature = sigma_feature + torch.sum(plane_coef_point * line_coef_point, dim=1)
        sigma_feature = self.agg_tensoRF_at_samples(local_kernel_dist, agg_id, sigma_feature[..., None]).squeeze(-1)
        return sigma_feature


    def compute_appfeature_geo(self, local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, local_kernel_dist, tensoRF_id, agg_id):

        # plane + line basis

        plane_coef_point,line_coef_point = [],[]
        for idx_plane in range(len(self.app_plane)):
            plane_coef_point.append(self.ind_intrp_plane_map(self.matMode[idx_plane], self.app_plane[idx_plane], local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, tensoRF_id))
            line_coef_point.append(self.ind_intrp_line_map(self.vecMode[idx_plane], self.app_line[idx_plane], local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, tensoRF_id))
        plane_coef_point, line_coef_point = torch.cat(plane_coef_point, dim=-1), torch.cat(line_coef_point, dim=-1)
        # print("local_kernel_dist, agg_id, (plane_coef_point * line_coef_point).T", local_kernel_dist.shape, agg_id.shape, (plane_coef_point * line_coef_point).T.shape, line_coef_point.shape, plane_coef_point.shape)
        app_feat = self.agg_tensoRF_at_samples(local_kernel_dist, agg_id, plane_coef_point * line_coef_point)
        return self.basis_mat(app_feat)


    @torch.no_grad()
    def up_sampling_VM(self, plane_coef, line_coef, res_target):
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef[i] = torch.nn.Parameter(
                F.interpolate(plane_coef[i].data, size=(res_target[mat_id_1]+1, res_target[mat_id_0]+1), mode='bilinear', align_corners=True))
            line_coef[i] = torch.nn.Parameter(
                F.interpolate(line_coef[i].data, size=(res_target[vec_id]+1), mode='linear', align_corners=True))


        return plane_coef, line_coef

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        print(f'upsamping to local dims: {self.local_dims} to {res_target}')
        self.local_dims = torch.as_tensor(res_target, device=self.local_dims.device, dtype=self.local_dims.dtype)
        self.app_plane, self.app_line = self.up_sampling_VM(self.app_plane, self.app_line, res_target)
        self.density_plane, self.density_line = self.up_sampling_VM(self.density_plane, self.density_line, res_target)
        self.update_stepSize(self.local_dims)



class PointTensorCPD(PointTensorBase):
    def __init__(self, aabb, gridSize, device, **kargs):
        super(PointTensorCPR, self).__init__(aabb, gridSize, device, **kargs)


    def init_svd_volume(self, local_dims, device):
        self.density_line = self.init_one_svd(self.density_n_comp[0], local_dims[:3], 0.2, device)
        self.app_line = self.init_one_svd(self.app_n_comp[0], local_dims[:3], 0.2, device)

        self.theta_line = self.init_angle_svd(self.app_n_comp[0], local_dims[3], 0.2, device)
        self.phi_line = self.init_angle_svd(self.app_n_comp[0], local_dims[4], 0.2, device)
        self.basis_mat = torch.nn.Linear(self.app_n_comp[0], self.app_dim, bias=False).to(device)


    def init_angle_svd(self, n_component, local_dim, scale, device):
        return torch.nn.Parameter(scale * torch.randn((len(self.geo), n_component, local_dim), device=device), requires_grad=True).to(device)


    def init_one_svd(self, n_component, local_dims, scale, device):
        line_coef = []
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            line_coef.append(
                torch.nn.Parameter(scale * torch.randn((len(self.geo), n_component, local_dims[vec_id] + 1))))
        return torch.nn.ParameterList(line_coef).to(device)


    def get_optparam_groups(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001, skip_zero_grad=True):
        grad_vars = [{'params': self.density_line, 'lr': lr_init_spatialxyz, 'skip_zero_grad': (skip_zero_grad)},
                     {'params': self.app_line, 'lr': lr_init_spatialxyz, 'skip_zero_grad': (skip_zero_grad)},
                     {'params': self.theta_view, 'lr': lr_init_spatialxyz, 'skip_zero_grad': (skip_zero_grad)},
                     {'params': self.phi_view, 'lr': lr_init_spatialxyz, 'skip_zero_grad': (skip_zero_grad)},
                     {'params': self.basis_mat.parameters(), 'lr':lr_init_network, 'skip_zero_grad': (False)}]
        if isinstance(self.renderModule, torch.nn.Module):
            grad_vars += [{'params':self.renderModule.parameters(), 'lr':lr_init_network, 'skip_zero_grad': (False)}]
        return grad_vars


    def ind_intrp_line_map_batch(self, vecModes, density_lines, local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, tensoRF_id):

        line_s = torch.stack([density_lines[0][tensoRF_id, :, local_gindx_s[..., vecModes[0]]], density_lines[1][tensoRF_id, :, local_gindx_s[..., vecModes[1]]], density_lines[2][tensoRF_id, :, local_gindx_s[..., vecModes[2]]]], dim=-1)
        line_l = torch.stack([density_lines[0][tensoRF_id, :, local_gindx_l[..., vecModes[0]]], density_lines[1][tensoRF_id, :, local_gindx_l[..., vecModes[1]]], density_lines[2][tensoRF_id, :, local_gindx_l[..., vecModes[2]]]], dim=-1)

        line_s_gweight = torch.stack([local_gweight_s[:, None, vecModes[0]], local_gweight_s[:, None, vecModes[1]], local_gweight_s[:, None, vecModes[2]]], dim=-1)
        line_l_gweight = torch.stack([local_gweight_l[:, None, vecModes[0]], local_gweight_l[:, None, vecModes[1]], local_gweight_l[:, None, vecModes[2]]], dim=-1)

        return line_s * line_s_gweight + line_l * line_l_gweight


    def ind_intrp_line_map_batch_prod(self, vecModes, density_lines, local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, tensoRF_id):
        return (density_lines[0][tensoRF_id, :, local_gindx_s[..., vecModes[0]]] * local_gweight_s[:, None, vecModes[0]] + density_lines[0][tensoRF_id, :, local_gindx_l[..., vecModes[0]]] * local_gweight_l[:, None, vecModes[0]]) *  (density_lines[1][tensoRF_id, :, local_gindx_s[..., vecModes[1]]] * local_gweight_s[:, None, vecModes[1]] + density_lines[1][tensoRF_id, :, local_gindx_l[..., vecModes[1]]] * local_gweight_l[:, None, vecModes[1]]) * (density_lines[2][tensoRF_id, :, local_gindx_s[..., vecModes[2]]] * local_gweight_s[:, None, vecModes[2]] + density_lines[2][tensoRF_id, :, local_gindx_l[..., vecModes[2]]] * local_gweight_l[:, None, vecModes[2]])


    @torch.no_grad()
    def up_sampling_Vector(self, density_line_coef, app_line_coef, res_target):
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            density_line_coef[i] = torch.nn.Parameter(
                F.interpolate(density_line_coef[i].data, size=(res_target[vec_id]+1), mode='linear',
                              align_corners=True))
            app_line_coef[i] = torch.nn.Parameter(
                F.interpolate(app_line_coef[i].data, size=(res_target[vec_id]+1), mode='linear', align_corners=True))

        return density_line_coef, app_line_coef

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        print(f'upsamping to local dims: {self.local_dims} to {res_target}')
        self.local_dims = torch.as_tensor(res_target, device=self.local_dims.device, dtype=self.local_dims.dtype)
        self.density_line, self.app_line = self.up_sampling_Vector(self.density_line, self.app_line, res_target)
        self.update_stepSize(self.local_dims)


    def compute_densityfeature_geo(self, local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, local_kernel_dist, tensoRF_id, agg_id):
        # plane + line basis

        # sigma_feature =  torch.sum(torch.prod(self.ind_intrp_line_map_batch(self.vecMode, self.density_line, local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, tensoRF_id), dim=-1), dim=1, keepdim=True)

        sigma_feature =  torch.sum(self.ind_intrp_line_map_batch_prod(self.vecMode, self.density_line, local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, tensoRF_id), dim=1, keepdim=True)

        sigma_feature = self.agg_tensoRF_at_samples(local_kernel_dist, agg_id, sigma_feature).squeeze(-1)
        return sigma_feature


    def compute_appfeature_geo(self, local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, local_kernel_dist, tensoRF_id, agg_id, ray_id, theta_gindx_s, theta_gindx_l, phi_gindx_s, phi_gindx_l, theta_gweight_l, phi_gweight_l):
        # plane + line basis
        # line_coef_point = torch.prod(self.ind_intrp_line_map_batch(self.vecMode, self.app_line, local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, tensoRF_id), dim=-1)

        line_coef_point = self.ind_intrp_line_map_batch_prod(self.vecMode, self.app_line, local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, tensoRF_id)

        app_feat = self.agg_tensoRF_at_samples(local_kernel_dist, agg_id, line_coef_point)
        return self.basis_mat(app_feat)


    def density_L1(self):
        total = 0
        for idx in range(len(self.density_line)):
            total = total + torch.mean(torch.abs(self.density_line[idx]))
        return total

    def TV_loss_density(self, reg):
        total = 0
        for idx in range(len(self.density_line)):
            total = total + reg(self.density_line[idx]) * 1e-3
        return total

    def TV_loss_app(self, reg):
        total = 0
        for idx in range(len(self.app_line)):
            total = total + reg(self.app_line[idx]) * 1e-3
        return total

    def forward(self, rays_chunk, white_bg=True, is_train=False, ray_type=0, N_samples=-1, return_depth=0, tensoRF_per_ray=None, eval=False):

        # sample points
        viewdirs = rays_chunk[:, 3:6]
        N, _ = rays_chunk.shape
        theta_gindx_s, theta_gindx_l, phi_gindx_s, phi_gindx_l, theta_gweight_l, phi_gweight_l = self.view_decompose(viewdirs)
        xyz_sampled, t_min, ray_id, step_id = self.sample_ray_cvrg_cuda(rays_chunk[:, :3], viewdirs, use_mask=True, N_samples=N_samples, random=False)
        mask_any = True
        if self.alphaMask is not None:
            mask = self.alphaMask.sample_alpha(xyz_sampled) > 0;
            mask_any = mask.any()
            # print("mask", mask.shape, torch.sum(mask))
            if mask_any:
                xyz_sampled = xyz_sampled[mask]
                ray_id = ray_id[mask]
                if return_depth:
                    step_id = step_id[mask]

        if ray_id is None or len(ray_id) == 0 or not mask_any:
            return torch.full([N, 3], 1.0 if (white_bg or (is_train and torch.rand((1,)) < 0.5)) else 0.0, device="cuda", dtype=torch.float32), rays_chunk[..., -1].detach()

        local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, local_kernel_dist, tensoRF_id, agg_id = self.sample_2_tensoRF_cvrg(xyz_sampled)
        sigma_feature = self.compute_densityfeature_geo(local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, local_kernel_dist, tensoRF_id, agg_id)

        alpha = Raw2Alpha.apply(sigma_feature.flatten(), self.density_shift, self.stepSize * self.distance_scale).reshape(sigma_feature.shape)
        weights, bg_weight = Alphas2Weights.apply(alpha, ray_id, N)
        mask = weights > self.rayMarch_weight_thres
        # print("weights",weights.shape,torch.min(weights), torch.max(weights))
        if mask.any() and (~mask).any():
            tensor_mask = mask[agg_id]
            weights = weights[mask]
            ray_id = ray_id[mask]
            agg_id = self.remask_aggid(agg_id, tensor_mask, mask)
            tensoRF_id = tensoRF_id[tensor_mask]
            local_gindx_s = local_gindx_s[tensor_mask]
            local_gindx_l = local_gindx_l[tensor_mask]
            local_gweight_s = local_gweight_s[tensor_mask]
            local_gweight_l = local_gweight_l[tensor_mask]
            local_kernel_dist = local_kernel_dist[tensor_mask]
            if return_depth:
                step_id = step_id[mask]
        app_features = self.compute_appfeature_geo(local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, local_kernel_dist, tensoRF_id, agg_id, ray_id, theta_gindx_s, theta_gindx_l, phi_gindx_s, phi_gindx_l, theta_gweight_l, phi_gweight_l)
        rgb = self.renderModule(None, viewdirs[ray_id], app_features)
        # print("rgb",rgb.shape, torch.max(rgb,dim=0)[0])

        rgb_map = segment_coo(
            src=(weights.unsqueeze(-1) * rgb),
            index=ray_id,
            out=torch.zeros([N, 3], device=weights.device, dtype=torch.float32),
            reduce='sum')
        # print("rgb_map",rgb_map.shape, torch.max(rgb_map,dim=0)[0])
        if white_bg or (is_train and torch.rand((1,)) < 0.5):
            rgb_map += (bg_weight.unsqueeze(-1))

        if return_depth:
            with torch.no_grad():
                z_val = t_min[ray_id] + step_id * self.stepSize
                depth_map = segment_coo(
                    src=(weights.unsqueeze(-1) * z_val[..., None]),
                    index=ray_id,
                    out=torch.zeros([N, 1], device=weights.device, dtype=torch.float32),
                    reduce='sum')[..., 0]
                depth_map = depth_map + bg_weight * rays_chunk[..., -1]
        else:
            depth_map = None

        rgb_map = rgb_map.clamp(0, 1)
        return rgb_map, depth_map  # rgb, sigma, alpha, weight, bg_weight



class PointTensorCPB(PointTensorBase):
    def __init__(self, aabb, gridSize, device, **kargs):
        super(PointTensorCPB, self).__init__(aabb, gridSize, device, **kargs)


    def init_svd_volume(self, local_dims, device):
        self.density_line = self.init_one_svd(self.density_n_comp[0], local_dims, 0.2, device)
        self.app_line = self.init_one_svd(self.app_n_comp[0], local_dims, 0.2, device)
        # stdv = 1. / math.sqrt(self.weight.size(1))
        # self.weight.data.uniform_(-stdv, stdv)
        # self.basis_mat = torch.nn.Parameter(2. / np.sqrt(self.app_dim) * (torch.randn(len(self.geo), self.app_n_comp[0], self.app_dim, device=device)-0.5), requires_grad=True).to(device)
        self.basis_mat = torch.nn.Parameter(2. / np.sqrt(self.app_dim) * (torch.randn(len(self.geo), self.app_n_comp[0], self.app_dim, device=device)-0.5), requires_grad=True).to(device)


    def init_one_svd(self, n_component, local_dims, scale, device):
        line_coef = []
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            line_coef.append(
                torch.nn.Parameter(scale * torch.randn((len(self.geo), n_component, local_dims[vec_id] + 1))))
        return torch.nn.ParameterList(line_coef).to(device)


    def get_optparam_groups(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001, skip_zero_grad=True):
        grad_vars = [{'params': self.density_line, 'lr': lr_init_spatialxyz, 'skip_zero_grad': (skip_zero_grad)},
                     {'params': self.app_line, 'lr': lr_init_spatialxyz, 'skip_zero_grad': (skip_zero_grad)},
                     {'params': self.basis_mat, 'lr':lr_init_spatialxyz, 'skip_zero_grad': (skip_zero_grad)}]
        if isinstance(self.renderModule, torch.nn.Module):
            grad_vars += [{'params':self.renderModule.parameters(), 'lr':lr_init_network, 'skip_zero_grad': (False)}]
        return grad_vars


    def ind_intrp_line_map_batch(self, vecModes, density_lines, local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, tensoRF_id):

        line_s = torch.stack([density_lines[0][tensoRF_id, :, local_gindx_s[..., vecModes[0]]], density_lines[1][tensoRF_id, :, local_gindx_s[..., vecModes[1]]], density_lines[2][tensoRF_id, :, local_gindx_s[..., vecModes[2]]]], dim=-1)
        line_l = torch.stack([density_lines[0][tensoRF_id, :, local_gindx_l[..., vecModes[0]]], density_lines[1][tensoRF_id, :, local_gindx_l[..., vecModes[1]]], density_lines[2][tensoRF_id, :, local_gindx_l[..., vecModes[2]]]], dim=-1)

        line_s_gweight = torch.stack([local_gweight_s[:, None, vecModes[0]], local_gweight_s[:, None, vecModes[1]], local_gweight_s[:, None, vecModes[2]]], dim=-1)
        line_l_gweight = torch.stack([local_gweight_l[:, None, vecModes[0]], local_gweight_l[:, None, vecModes[1]], local_gweight_l[:, None, vecModes[2]]], dim=-1)

        return line_s * line_s_gweight + line_l * line_l_gweight


    def ind_intrp_line_map_prod(self, vecModes, density_lines, local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, tensoRF_id):
        return (density_lines[0][tensoRF_id, :, local_gindx_s[..., vecModes[0]]] * local_gweight_s[:, None, vecModes[0]] + density_lines[0][tensoRF_id, :, local_gindx_l[..., vecModes[0]]] * local_gweight_l[:, None, vecModes[0]]) *  (density_lines[1][tensoRF_id, :, local_gindx_s[..., vecModes[1]]] * local_gweight_s[:, None, vecModes[1]] + density_lines[1][tensoRF_id, :, local_gindx_l[..., vecModes[1]]] * local_gweight_l[:, None, vecModes[1]]) * (density_lines[2][tensoRF_id, :, local_gindx_s[..., vecModes[2]]] * local_gweight_s[:, None, vecModes[2]] + density_lines[2][tensoRF_id, :, local_gindx_l[..., vecModes[2]]] * local_gweight_l[:, None, vecModes[2]])

    #
    def ind_intrp_line_B_map_prod(self, vecModes, density_lines, local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, tensoRF_id, Bmat):
        line_weights = ((density_lines[0][tensoRF_id, :, local_gindx_s[..., vecModes[0]]] * local_gweight_s[:, None, vecModes[0]] + density_lines[0][tensoRF_id, :, local_gindx_l[..., vecModes[0]]] * local_gweight_l[:, None, vecModes[0]]) *  (density_lines[1][tensoRF_id, :, local_gindx_s[..., vecModes[1]]] * local_gweight_s[:, None, vecModes[1]] + density_lines[1][tensoRF_id, :, local_gindx_l[..., vecModes[1]]] * local_gweight_l[:, None, vecModes[1]]) * (density_lines[2][tensoRF_id, :, local_gindx_s[..., vecModes[2]]] * local_gweight_s[:, None, vecModes[2]] + density_lines[2][tensoRF_id, :, local_gindx_l[..., vecModes[2]]] * local_gweight_l[:, None, vecModes[2]]))[..., None]
        return torch.sum(line_weights * Bmat[tensoRF_id, ...], dim=1)



    @torch.no_grad()
    def up_sampling_Vector(self, density_line_coef, app_line_coef, res_target):
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            density_line_coef[i] = torch.nn.Parameter(
                F.interpolate(density_line_coef[i].data, size=(res_target[vec_id]+1), mode='linear',
                              align_corners=True))
            app_line_coef[i] = torch.nn.Parameter(
                F.interpolate(app_line_coef[i].data, size=(res_target[vec_id]+1), mode='linear', align_corners=True))

        return density_line_coef, app_line_coef

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        print(f'upsamping to local dims: {self.local_dims} to {res_target}')
        self.local_dims = torch.as_tensor(res_target, device=self.local_dims.device, dtype=self.local_dims.dtype)
        self.density_line, self.app_line = self.up_sampling_Vector(self.density_line, self.app_line, res_target)
        self.update_stepSize(self.local_dims)


    def compute_densityfeature_geo(self, local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, local_kernel_dist, tensoRF_id, agg_id):
        # plane + line basis

        # sigma_feature =  torch.sum(torch.prod(self.ind_intrp_line_map_batch(self.vecMode, self.density_line, local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, tensoRF_id), dim=-1), dim=1, keepdim=True)

        sigma_feature =  torch.sum(self.ind_intrp_line_map_prod(self.vecMode, self.density_line, local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, tensoRF_id), dim=1, keepdim=True)

        sigma_feature = self.agg_tensoRF_at_samples(local_kernel_dist, agg_id, sigma_feature).squeeze(-1)
        return sigma_feature


    def compute_appfeature_geo_nochunk(self, local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, local_kernel_dist, tensoRF_id, agg_id):
        # plane + line basis
        # line_coef_point = torch.prod(self.ind_intrp_line_map_batch(self.vecMode, self.app_line, local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, tensoRF_id), dim=-1)

        line_coef_feat = self.ind_intrp_line_B_map_prod(self.vecMode, self.app_line, local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, tensoRF_id, self.basis_mat)

        return self.agg_tensoRF_at_samples(local_kernel_dist, agg_id, line_coef_feat)


    def compute_appfeature_geo(self, local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, local_kernel_dist, tensoRF_id, agg_id):
        # plane + line basis
        # line_coef_point = torch.prod(self.ind_intrp_line_map_batch(self.vecMode, self.app_line, local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, tensoRF_id), dim=-1)

        chunksize=4096
        line_coef_feat_cat = None
        start = 0
        for i in range(len(local_gindx_s) // chunksize + 1):
            end = min((i+1) * chunksize, len(local_gindx_s))
            # print("len(local_gindx_s), end ", len(local_gindx_s), end)
            if line_coef_feat_cat is None:
                line_coef_feat_cat = self.ind_intrp_line_B_map_prod(self.vecMode, self.app_line, local_gindx_s[start:end,...], local_gindx_l[start:end,...], local_gweight_s[start:end,...], local_gweight_l[start:end,...], tensoRF_id[start:end,...], self.basis_mat)
            else:
                line_coef_feat_cat = torch.cat([line_coef_feat_cat, self.ind_intrp_line_B_map_prod(
                                                                    self.vecMode, self.app_line,
                                                                    local_gindx_s[start:end, ...],
                                                                    local_gindx_l[start:end, ...],
                                                                    local_gweight_s[start:end, ...],
                                                                    local_gweight_l[start:end, ...],
                                                                    tensoRF_id[start:end, ...], self.basis_mat)],                                                                     dim=0)
            start = end

        return self.agg_tensoRF_at_samples(local_kernel_dist, agg_id, line_coef_feat_cat)



    def density_L1(self):
        total = 0
        for idx in range(len(self.density_line)):
            total = total + torch.mean(torch.abs(self.density_line[idx]))
        return total

    def TV_loss_density(self, reg):
        total = 0
        for idx in range(len(self.density_line)):
            total = total + reg(self.density_line[idx]) * 1e-3
        return total

    def TV_loss_app(self, reg):
        total = 0
        for idx in range(len(self.app_line)):
            total = total + reg(self.app_line[idx]) * 1e-3
        return total
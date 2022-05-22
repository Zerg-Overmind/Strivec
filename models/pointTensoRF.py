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
from tqdm import tqdm

class PointTensorBase(TensorBase):
    def __init__(self, aabb, gridSize, device, density_n_comp=8, appearance_n_comp=24, app_dim=27, shadingMode='MLP_PE', alphaMask=None, near_far=[2.0, 6.0], density_shift=-10, alphaMask_thres=0.001, distance_scale=25, rayMarch_weight_thres=0.0001, pos_pe=6, view_pe=6, fea_pe=6, featureC=128, step_ratio=2.0, fea2denseAct='softplus', local_dims=None, geo=None, args=None):
        super(TensorBase, self).__init__()
        assert geo is not None, "No geo loaded, when using pointTensorBase"
        self.args = args
        self.geo=geo
        self.pnt_xyz = geo[..., :3].cuda().contiguous()
        self.density_n_comp = density_n_comp
        self.app_n_comp = appearance_n_comp
        self.app_dim = app_dim
        self.aabb = aabb
        self.alphaMask = alphaMask
        self.device = device
        self.local_range = torch.as_tensor(args.local_range, device=device, dtype=torch.float32)
        self.local_dims = torch.as_tensor(local_dims, device=device, dtype=torch.int64)
        self.density_shift = density_shift
        self.alphaMask_thres = alphaMask_thres
        self.distance_scale = distance_scale
        self.rayMarch_weight_thres = rayMarch_weight_thres
        self.fea2denseAct = fea2denseAct
        self.max_tensoRF = args.rot_max_tensoRF if args.rot_max_tensoRF is not None else args.max_tensoRF
        self.near_far = near_far
        self.step_ratio = step_ratio

        self.update_stepSize(self.local_dims)

        self.matMode = [[0, 1], [0, 2], [1, 2]]
        self.vecMode = [2, 1, 0]
        self.comp_w = [1, 1, 1]
        self.init_svd_volume(local_dims, device)

        self.shadingMode, self.pos_pe, self.view_pe, self.fea_pe, self.featureC = shadingMode, pos_pe, view_pe, fea_pe, featureC
        self.init_render_func(shadingMode, pos_pe, view_pe, fea_pe, featureC, device)

    def update_stepSize(self, local_dims):
        print("aabb", self.aabb.view(-1))
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invaabbSize = 2.0/self.aabbSize
        if self.args.tensoRF_shape == "cube":
            self.units = (2 * self.local_range / local_dims[:3])
            self.stepSize = torch.mean(self.units) * self.step_ratio
            print(torch.mean(self.units) , self.step_ratio)
            self.gridSize = torch.ceil(self.aabbSize / self.units).long().to(self.device)
            self.radius = torch.sqrt(torch.sum(torch.square(self.local_range))).cpu().item()
        else:
            self.units = torch.stack([(self.local_range[1] - self.local_range[0]) / local_dims[0], torch.as_tensor(np.pi * 2, device=self.local_range.device, dtype=self.local_range.dtype) / local_dims[1], torch.as_tensor(np.pi, device=self.local_range.device, dtype=self.local_range.dtype) / local_dims[2]], dim=0)
            self.units_3 = torch.stack([self.units[0], self.units[0], self.units[0]], dim=0)
            self.radiush = self.local_range[1].cpu().item()
            self.radiusl = self.local_range[0].cpu().item()
            self.stepSize = self.units[0] * self.step_ratio
            self.gridSize = torch.ceil(self.aabbSize / self.units[0]).long().to(self.device)

        if len(local_dims) > 3:
            self.view_units = torch.stack([torch.as_tensor(np.pi * 2, device=self.local_range.device, dtype=self.local_range.dtype) / local_dims[3], torch.as_tensor(np.pi, device=self.local_range.device, dtype=self.local_range.dtype) / local_dims[4]], dim=0)

        self.aabbDiag = torch.sqrt(torch.sum(torch.square(self.aabbSize)))
        self.nSamples=int((self.aabbDiag / self.stepSize).item()) + 1
        print("sampling step size: ", self.stepSize)
        print("sampling grid size: ", self.gridSize)
        print("nSamples: ", self.nSamples)
        self.create_sample_map()


    def compute_alpha(self, xyz_locs, length=1, pnt_rmatrix=None):
        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(xyz_locs)
            alpha_mask = alphas > 0
        else:
            alpha_mask = torch.ones_like(xyz_locs[:, 0], dtype=bool)

        alpha_mask = torch.logical_and(alpha_mask, self.filter_xyz_cvrg(xyz_locs, pnt_rmatrix=pnt_rmatrix))
        sigma = torch.zeros(xyz_locs.shape[:-1], device=xyz_locs.device)
        # print("alpha_mask", self.alphaMask is not None, torch.sum(alpha_mask))
        if alpha_mask.any():
            # self.cvrg_inds_center2pnts(self.tensoRF_cvrg_inds)
            # print("xyz_sampled.contiguous()", xyz_locs[alpha_mask])
            local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, local_kernel_dist, tensoRF_id, agg_id = self.sample_2_tensoRF_cvrg(xyz_locs[alpha_mask], pnt_rmatrix=pnt_rmatrix, rotgrad=False)
            # if len(local_gindx_s) == 0:
            #     self.cvrg_inds_center2pnts(self.tensoRF_cvrg_inds)
            #     np.savetxt("xyz_locs.txt", xyz_locs[alpha_mask].cpu().numpy(), delimiter=";")

            sigma_feature = self.compute_densityfeature_geo(local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, local_kernel_dist, tensoRF_id, agg_id)
            validsigma = self.feature2density(sigma_feature)
            # print("sigma_feature", sigma_feature.shape)
            # print("validsigma", validsigma.shape)
            # print("sigma", sigma.shape)
            # print("alpha_mask", alpha_mask.shape)
            sigma[alpha_mask] = validsigma
        alpha = 1 - torch.exp(-sigma * length).view(xyz_locs.shape[:-1])
        return alpha

    def agg_weight_func(self, dist):
        if self.args.intrp_mthd == "linear":
            return (1 / (dist + 1e-6))[..., None]
        elif self.args.intrp_mthd == "avg":
            return torch.ones_like(dist[..., None])
        elif self.args.intrp_mthd == "quadric":
            return (1 / (dist*dist + 1e-6))[..., None]

    def agg_tensoRF_at_samples(self, dist, agg_id, value):
        weights = self.agg_weight_func(dist)
        weight_sum = torch.clamp(segment_coo(
            src=weights,
            index=agg_id,
            reduce='sum'), min=1)
        agg_value = segment_coo(
            src=weights * value,
            index=agg_id,
            reduce='sum')
        return agg_value / weight_sum

    @torch.no_grad()
    def filtering_rays(self, all_rays, all_rgbs, N_samples=-1, chunk=10240 * 1, cvrg=True, bbox_only=False, apply_filter=True):
        if apply_filter: print('========> filtering rays ...')
        tt = time.time()

        N = torch.tensor(all_rays.shape[:-1]).prod()

        mask_filtered = []
        idx_chunks = torch.split(torch.arange(N), chunk)
        tensoRF_per_ray_lst = []
        def passfunc(a):
            return a
        func = tqdm if apply_filter else passfunc
        for idx_chunk in func(idx_chunks):#img_list:#
            # print("all_rays[idx_chunk]", all_rays.shape, idx_chunk.shape)
            rays_chunk = all_rays[idx_chunk].to(self.device)

            rays_o, rays_d = rays_chunk[..., :3], rays_chunk[..., 3:6]
            xyz_sampled, _, xyz_inbbox = self.sample_ray(rays_o, rays_d, N_samples=N_samples, is_train=False)

            if bbox_only:

                vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
                rate_a = (self.aabb[1] - rays_o) / vec
                rate_b = (self.aabb[0] - rays_o) / vec
                t_min = torch.minimum(rate_a, rate_b).amax(-1)  # .clamp(min=near, max=far)
                t_max = torch.maximum(rate_a, rate_b).amin(-1)  # .clamp(min=near, max=far)
                mask_inbbox = t_max > t_min

            else:
                mask_inbbox = (self.alphaMask.sample_alpha(xyz_sampled).view(xyz_sampled.shape[:-1]) > 0).any(-1)
            # mask_inrange = filter_ray_by_points(xyz_sampled, self.geo[...,:3], self.local_range)
            if cvrg:
                mask_inrange = filter_ray_by_cvrg(xyz_sampled, xyz_inbbox, self.units if self.args.tensoRF_shape == "cube" else self.units_3, self.aabb[0], self.aabb[1], self.tensoRF_cvrg_inds)
                mask_filtered.append(mask_inrange.view(xyz_sampled.shape[:-1]).any(-1).cpu())
                # print("mask_inrange", mask_inrange.shape, mask_inrange)
            else:
                tensoRF_per_ray = filter_ray_by_projection(rays_o, rays_d, self.pnt_xyz, self.local_range)
                mask_inrange = tensoRF_per_ray > 0
                mask_filtered.append((mask_inbbox * mask_inrange).cpu())
                tensoRF_per_ray_lst.append(tensoRF_per_ray.cpu())

        mask_filtered = torch.cat(mask_filtered).view(all_rays.shape[:-1])
        tensoRF_per_ray = torch.cat(tensoRF_per_ray_lst).view(all_rays.shape[:-1]) if len(tensoRF_per_ray_lst) > 0 else None

        if apply_filter:
            all_rays, all_rgbs, tensoRF_per_ray = all_rays[mask_filtered], all_rgbs[mask_filtered], None if tensoRF_per_ray is None else tensoRF_per_ray[mask_filtered]
            print(f'Ray filtering done! takes {time.time() - tt} s. ray mask ratio: {torch.sum(mask_filtered) / N}')

        return all_rays, all_rgbs, tensoRF_per_ray


    @torch.no_grad()
    def updateAlphaMask(self):
        gridSize = self.gridSize
        alpha, dense_xyz = self.getDenseAlpha(gridSize)
        dense_xyz = dense_xyz.transpose(0, 2).contiguous()
        alpha = alpha.clamp(0, 1).transpose(0, 2).contiguous()[None, None]
        total_voxels = gridSize[0] * gridSize[1] * gridSize[2]

        ks = 3
        alpha = F.max_pool3d(alpha, kernel_size=ks, padding=ks // 2, stride=1).view((gridSize[2], gridSize[1], gridSize[0])) #.view(gridSize[::-1])
        alpha[alpha >= self.alphaMask_thres] = 1
        alpha[alpha < self.alphaMask_thres] = 0

        self.alphaMask = AlphaGridMask(self.device, self.aabb, alpha, mask_cache_thres=self.alphaMask_thres)

        valid_xyz = dense_xyz[alpha > 0.5]

        xyz_min = valid_xyz.amin(0)
        xyz_max = valid_xyz.amax(0)

        new_aabb = torch.stack((xyz_min, xyz_max))

        total = torch.sum(alpha)
        print(f"bbox: {xyz_min, xyz_max} alpha rest %%%f" % (total / total_voxels * 100))
        return new_aabb


    @torch.no_grad()
    def getDenseAlpha(self,gridSize=None):
        gridSize = self.gridSize if gridSize is None else gridSize

        samples = torch.stack(torch.meshgrid(
            torch.linspace(0, 1, gridSize[0]),
            torch.linspace(0, 1, gridSize[1]),
            torch.linspace(0, 1, gridSize[2]),
        ), -1).to(self.device)
        samples = self.aabb[0] * (1-samples) + self.aabb[1] * samples
        # dense_xyz = dense_xyz
        # print(self.stepSize, self.distance_scale*self.aabbDiag)
        alpha = torch.zeros_like(samples[...,0])
        pnt_rmatrix = None if self.args.rot_init is None else self.rot2m(self.pnt_rot)
        for i in range(gridSize[0]):
            alpha[i] = self.compute_alpha(samples[i].view(-1,3), self.stepSize, pnt_rmatrix).view((gridSize[1], gridSize[2])) #  * self.distance_scale
        return alpha, samples


    def filter_by_points(self, xyz_sampled):
        xyz_dist = torch.abs(xyz_sampled[..., None, :] - self.pnt_xyz[None, ...]) # raysampleN * 4096 * 3
        mask_inrange = torch.all(xyz_dist <= self.local_range[None,None, :], dim=-1) # raysampleN * 4096
        mask_inrange = torch.any(mask_inrange.view(mask_inrange.shape[0], -1), dim=-1)
        return mask_inrange


    def select_by_points(self, xyz_sampled, ray_id, step_id=None):
        xyz_dist = torch.abs(xyz_sampled[..., None, :] - self.pnt_xyz[None, ...]) # raysampleN * 4096 * 3
        mask_inrange = torch.all(xyz_dist <= self.local_range[None,None, :], dim=-1) # raysampleN * 4096
        mask_inds = torch.nonzero(mask_inrange)
        # print("mask_inds", mask_inds.shape)
        xyz_sampled = xyz_sampled[mask_inds[..., 0], ...]
        ray_id = ray_id[mask_inds[..., 0], ...]
        step_id = step_id[mask_inds[..., 0], ...] if step_id is not None else None
        return xyz_sampled, ray_id, step_id, mask_inds[..., 1]

    def create_sample_map(self):
        print("start create mapping")
        print("bself.pnt_xyz", self.pnt_xyz.shape, self.pnt_xyz.device)
        if self.args.tensoRF_shape == "cube":
            if self.args.rot_init is None:
                self.tensoRF_cvrg_inds, self.tensoRF_count, self.tensoRF_topindx, self.geo_xyz_recenter = search_geo_cuda.build_tensoRF_map(self.pnt_xyz, self.gridSize, self.aabb[0], self.aabb[1], self.units, self.local_range, self.local_dims[:3], self.max_tensoRF)
                self.geo_xyz_recenter = self.pnt_xyz if self.args.align_center == 0 else self.geo_xyz_recenter.contiguous()
            else:
                self.tensoRF_cvrg_inds, self.tensoRF_count, self.tensoRF_topindx = search_geo_cuda.build_sphere_tensoRF_map(self.pnt_xyz, self.gridSize, self.aabb[0], self.aabb[1], self.units, 0.0, self.radius, self.local_dims[:3], self.max_tensoRF)

        elif self.args.tensoRF_shape == "sphere":
            self.tensoRF_cvrg_inds, self.tensoRF_count, self.tensoRF_topindx = search_geo_cuda.build_sphere_tensoRF_map(self.pnt_xyz, self.gridSize, self.aabb[0], self.aabb[1], self.units, self.radiusl, self.radiush, self.local_dims[:3], self.max_tensoRF)

        self.tensoRF_cvrg_inds, self.tensoRF_count, self.tensoRF_topindx = self.tensoRF_cvrg_inds.contiguous(), self.tensoRF_count.contiguous(), self.tensoRF_topindx.contiguous()


        # self.cvrg_inds_center2pnts(self.tensoRF_cvrg_inds)

        # print("tensoRF_cvrg_inds, tensoRF_count, tensoRF_topindx, max_tensoRF_count", self.tensoRF_cvrg_inds.shape, self.tensoRF_count.shape, self.tensoRF_topindx.shape, torch.max(self.tensoRF_count))
        # print("tensoRF_cvrg_inds", self.tensoRF_cvrg_inds.numel(), torch.max(self.tensoRF_cvrg_inds), torch.sum(self.tensoRF_cvrg_inds >= 0), self.tensoRF_count.shape, self.tensoRF_topindx.shape)


    def sample_ray_geo_cuda(self, rays_o, rays_d, geo, tensoRF_per_ray, use_mask=True, N_samples=-1, random=False):
        '''Sample query points on rays.
        All the output points are sorted from near to far.
        Input:
            rays_o, rayd_d:   both in [N, 3] indicating ray configurations.
        Output:
            ray_pts:          [M, 3] storing all the sampled points.
            ray_id:           [M]    the index of the ray of each point.
            step_id:          [M]    the i'th step on a ray of each point.
        '''
        near, far = self.near_far
        rays_o = rays_o.view(-1,3).contiguous()
        rays_d = rays_d.view(-1,3).contiguous()
        if torch.sum(tensoRF_per_ray) > 0:
            xyz_sampled, sample_dir, local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, local_kernel_dist, ray_id, step_id, tensoRF_id, agg_id, t_min, t_max = render_utils_cuda.sample_pts_on_rays_geo(rays_o, rays_d, self.pnt_xyz, tensoRF_per_ray.contiguous(), torch.square(self.local_range), self.aabb[0], self.aabb[1], near, far, self.stepSize, self.local_range, self.local_dims[:3], self.stepSize)
        else:
            return None, None, None, None, None, None, None, None, None, None, None, None

        return xyz_sampled, sample_dir, local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, local_kernel_dist, t_min, ray_id, step_id, tensoRF_id, agg_id


    def sample_ray_cvrg_cuda(self, rays_o, rays_d, use_mask=True, N_samples=-1, random=False, ji=False):
        '''Sample query points on rays.
        All the output points are sorted from near to far.
        Input:
            rays_o, rayd_d:   both in [N, 3] indicating ray configurations.
        Output:
            ray_pts:          [M, 3] storing all the sampled points.
            ray_id:           [M]    the index of the ray of each point.
            step_id:          [M]    the i'th step on a ray of each point.
        '''
        near, far = self.near_far
        rays_o = rays_o.view(-1, 3).contiguous()
        rays_d = rays_d.view(-1, 3).contiguous()
        shift = None
        pnt_rmatrix = None
        if self.args.tensoRF_shape == "cube":
            if random:
                shift = (1 + self.args.shp_rand * (torch.rand(rays_o.shape[0], dtype=rays_d.dtype, device=rays_d.device) - 0.5)) * self.stepSize
                ray_pts, mask_valid, ray_id, step_id, N_steps, t_min, t_max = search_geo_cuda.sample_pts_on_rays_dist_cvrg( rays_o, rays_d, self.tensoRF_cvrg_inds >= 0, self.units, self.aabb[0], self.aabb[1], near, far, self.stepSize, shift)
            elif ji:
                ray_pts, mask_valid, ray_id, step_id, N_steps, t_min, t_max = search_geo_cuda.sample_pts_on_rays_ji_cvrg(rays_o, rays_d, self.tensoRF_cvrg_inds >= 0, self.units, self.aabb[0], self.aabb[1], near, far, self.stepSize, torch.rand(rays_o.shape[0], dtype=rays_d.dtype, device=rays_d.device) * self.stepSize)
            elif self.args.rot_init is not None:
                pnt_rmatrix = self.rot2m(self.pnt_rot)
                ray_pts, mask_valid, ray_id, step_id, N_steps, t_min, t_max = search_geo_cuda.sample_pts_on_rays_rot_cvrg(rays_o, rays_d, self.pnt_xyz, pnt_rmatrix, self.tensoRF_cvrg_inds, self.tensoRF_count, self.tensoRF_topindx, self.units, self.local_range, self.aabb[0], self.aabb[1], near, far, self.stepSize)
            else:
                ray_pts, mask_valid, ray_id, step_id, N_steps, t_min, t_max = search_geo_cuda.sample_pts_on_rays_cvrg(rays_o, rays_d, self.tensoRF_cvrg_inds >= 0, self.units, self.aabb[0], self.aabb[1], near, far, self.stepSize)
        elif self.args.tensoRF_shape == "sphere":
           # assert not random, "random has no implementation for sphere!!!"
           # assert not ji, "ji has no implementation for sphere!!!"
           # assert args.rot_init is None, "args.rot_init has no implementation for sphere!!!"
           ray_pts, mask_valid, ray_id, step_id, N_steps, t_min, t_max = search_geo_cuda.sample_pts_on_rays_sphere_cvrg(rays_o, rays_d, self.pnt_xyz, self.tensoRF_cvrg_inds, self.tensoRF_count, self.tensoRF_topindx, self.units, self.radiusl, self.radiush, self.aabb[0], self.aabb[1], near, far, self.stepSize)
        if use_mask:
            ray_pts = ray_pts[mask_valid]
            ray_id = ray_id[mask_valid]
            step_id = step_id[mask_valid]
        # print("t_min", t_min.shape, step_id.shape, ray_pts.shape)
        return ray_pts, t_min, ray_id, step_id, shift, pnt_rmatrix


    def filter_xyz_cvrg(self, xyz_sampled, pnt_rmatrix=None):
        if self.args.tensoRF_shape == "cube":
            if self.args.rot_init is None:
                mask = search_geo_cuda.filter_xyz_cvrg(xyz_sampled.contiguous(), self.aabb[0], self.aabb[1], self.units,self.tensoRF_cvrg_inds >= 0)
            else:
                mask = search_geo_cuda.filter_xyz_rot_cvrg(self.pnt_xyz, pnt_rmatrix, xyz_sampled.contiguous(), self.aabb[0], self.aabb[1], self.units,self.tensoRF_cvrg_inds, self.tensoRF_count, self.tensoRF_topindx, self.local_range)
        elif self.args.tensoRF_shape == "sphere":
            mask = search_geo_cuda.filter_xyz_sphere_cvrg(xyz_sampled.contiguous(), self.aabb[0], self.aabb[1], self.units, self.radiusl, self.radiush, self.tensoRF_cvrg_inds, self.tensoRF_count, self.tensoRF_topindx, self.pnt_xyz)
        return mask

    def normalize_coord_tsrf(self, sample_tensoRF_pos):
        return sample_tensoRF_pos / self.local_range[None, ...]


    def cvrg_inds_center2pnts(self, tensoRF_cvrg_inds):
        inds = torch.nonzero(tensoRF_cvrg_inds>=0)
        pnts = self.aabb[0][None, ...] + (inds+0.5) * self.units[None, ...]
        print("center pnts", torch.min(pnts, dim=0)[0], torch.max(pnts, dim=0)[0])
        np.savetxt("cvrg_inds_center2pnts.txt", pnts.cpu().numpy(), delimiter=";")

    def rot2m(self, rot):
        sin_roll = torch.sin(rot[:, 0])
        cos_roll = torch.cos(rot[:, 0])
        sin_pitch = torch.sin(rot[:, 1])
        cos_pitch = torch.cos(rot[:, 1])
        sin_yaw = torch.sin(rot[:, 2])
        cos_yaw = torch.cos(rot[:, 2])

        tensor_0 = torch.zeros_like(sin_roll)
        tensor_1 = torch.ones_like(sin_roll)

        RX = torch.stack([
            torch.stack([tensor_1, tensor_0, tensor_0], dim=-1),
            torch.stack([tensor_0, cos_roll, -sin_roll], dim=-1),
            torch.stack([tensor_0, sin_roll, cos_roll], dim=-1)], dim=-2)

        RY = torch.stack([
            torch.stack([cos_pitch, tensor_0, sin_pitch], dim=-1),
            torch.stack([tensor_0, tensor_1, tensor_0], dim=-1),
            torch.stack([-sin_pitch, tensor_0, cos_pitch], dim=-1)], dim=-2)

        RZ = torch.stack([
            torch.stack([cos_yaw, -sin_yaw, tensor_0], dim=-1),
            torch.stack([sin_yaw, cos_yaw, tensor_0], dim=-1),
            torch.stack([tensor_0, tensor_0, tensor_1], dim=-1)], dim=-2)

        # R = RZ @ RY @ RX
        R = torch.matmul(torch.matmul(RZ, RY), RX)
        return R


    def sample_2_tensoRF_cvrg(self, xyz_sampled, pnt_rmatrix=None, rotgrad=False):

        if self.args.tensoRF_shape == "cube":
            if self.args.rot_init is not None and not rotgrad:
                mask, local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, local_kernel_dist, tensoRF_id, agg_id = search_geo_cuda.sample_2_rot_tensoRF_cvrg(xyz_sampled.contiguous(), self.aabb[0], self.aabb[1], self.units, self.local_range, self.local_dims[:3], self.tensoRF_cvrg_inds, self.tensoRF_count, self.tensoRF_topindx, pnt_rmatrix, self.pnt_xyz)
                local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, local_kernel_dist, tensoRF_id = local_gindx_s[mask, :], local_gindx_l[mask, :], local_gweight_s[mask, :], local_gweight_l[mask, :], local_kernel_dist[mask], tensoRF_id[mask]
            elif self.args.rot_init is not None and rotgrad:
                mask, dist, local_kernel_dist, tensoRF_id, agg_id = search_geo_cuda.sample_2_rotdist_tensoRF_cvrg(xyz_sampled.contiguous(), self.aabb[0], self.aabb[1], self.units, self.local_range, self.local_dims[:3], self.tensoRF_cvrg_inds, self.tensoRF_count, self.tensoRF_topindx, pnt_rmatrix, self.pnt_xyz)
                dist, local_kernel_dist, tensoRF_id  = dist[mask, :], local_kernel_dist[mask], tensoRF_id[mask]
                rot_dist = torch.matmul(dist[:,None,:], pnt_rmatrix[tensoRF_id,:,:]).squeeze(1) + self.local_range[None, :]
                soft_inds = rot_dist / self.units[None, :];
                local_gindx_s = torch.minimum(torch.clamp(torch.floor(soft_inds).long(), min=0), (self.local_dims-1)[None,:])
                local_gindx_l = local_gindx_s + 1
                local_gweight_l = soft_inds - local_gindx_s
                local_gweight_s = 1 - local_gweight_l
            else:
                local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, local_kernel_dist, tensoRF_id, agg_id = search_geo_cuda.sample_2_tensoRF_cvrg(xyz_sampled.contiguous(), self.aabb[0], self.aabb[1], self.units, self.local_range, self.local_dims[:3], self.tensoRF_cvrg_inds, self.tensoRF_count, self.tensoRF_topindx, self.geo_xyz_recenter, self.pnt_xyz)

        elif self.args.tensoRF_shape == "sphere":

            mask, local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, local_kernel_dist, tensoRF_id, agg_id = search_geo_cuda.sample_2_sphere_tensoRF_cvrg(xyz_sampled.contiguous(), self.aabb[0], self.aabb[1], self.units, self.radiusl, self.radiush, self.local_dims[:3], self.tensoRF_cvrg_inds, self.tensoRF_count, self.tensoRF_topindx, self.pnt_xyz)

            local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, local_kernel_dist, tensoRF_id = local_gindx_s[mask, :], local_gindx_l[mask, :], local_gweight_s[mask, :], local_gweight_l[mask, :], local_kernel_dist[mask], tensoRF_id[mask]

        return local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, local_kernel_dist, tensoRF_id, agg_id


    def inds_cvrg(self, xyz_sampled):
        global_gindx, local_gindx, tensoRF_id = search_geo_cuda.inds_cvrg(xyz_sampled.contiguous(), self.aabb[0], self.aabb[1], self.units, self.local_range, self.local_dims[:3], self.tensoRF_cvrg_inds, self.tensoRF_count, self.tensoRF_topindx, self.geo_xyz_recenter, self.pnt_xyz)
        return global_gindx, local_gindx, tensoRF_id


    def view_decompose(self, viewdirs):

        theta = torch.atan2(viewdirs[..., 1], viewdirs[...,0]) + np.pi
        phi = torch.acos(viewdirs[..., 2])
        soft_theta_gindx_s = theta / self.view_units[0]
        soft_phi_gindx_s = phi / self.view_units[1]

        theta_gindx_s = torch.floor(soft_theta_gindx_s)
        phi_gindx_s = torch.floor(soft_phi_gindx_s)

        theta_gweight_l = soft_theta_gindx_s - theta_gindx_s
        phi_gweight_l = soft_phi_gindx_s - phi_gindx_s


        theta_gindx_s = torch.remainder(theta_gindx_s, self.local_dims[3])
        phi_gindx_s = torch.remainder(phi_gindx_s, self.local_dims[4])

        theta_gindx_l = torch.remainder(theta_gindx_s+1, self.local_dims[3])
        phi_gindx_l = torch.remainder(phi_gindx_s+1, self.local_dims[4])

        return torch.stack([theta_gindx_s, phi_gindx_s], dim=0).long(), torch.stack([theta_gindx_l, phi_gindx_l], dim=0).long(), torch.stack([theta_gweight_l, phi_gweight_l], dim=0)



    def forward(self, rays_chunk, white_bg=True, is_train=False, ray_type=0, N_samples=-1, return_depth=0, tensoRF_per_ray=None, eval=False, rot_step=False):

        # sample points
        viewdirs = rays_chunk[:, 3:6]
        if len(self.local_dims) > 3:
            dir_gindx_s, dir_gindx_l, dir_gweight_l = self.view_decompose(viewdirs)
        else:
            dir_gindx_s, dir_gindx_l, dir_gweight_l = None, None, None

        N, _ = rays_chunk.shape
        shp_rand = (self.args.shp_rand > 0) and (not eval)
        ji = (self.args.ji > 0) and (not eval)
        # xyz_sampled, viewdirs, local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, local_kernel_dist, t_min, ray_id, step_id, tensoRF_id, agg_id = self.sample_ray_geo_cuda(rays_chunk[:, :3], viewdirs, self.geo, tensoRF_per_ray, use_mask=True, N_samples=N_samples, random=False)
        xyz_sampled, t_min, ray_id, step_id, shift, pnt_rmatrix = self.sample_ray_cvrg_cuda(rays_chunk[:, :3], viewdirs, use_mask=True, N_samples=N_samples, random=shp_rand, ji=ji)
        # print("xyz_sampled, ", xyz_sampled.shape, ji, shp_rand)
        # self.cvrg_inds_center2pnts(self.tensoRF_cvrg_inds)
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

        local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, local_kernel_dist, tensoRF_id, agg_id = self.sample_2_tensoRF_cvrg(xyz_sampled, pnt_rmatrix=pnt_rmatrix, rotgrad=rot_step)
        sigma_feature = self.compute_densityfeature_geo(local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, local_kernel_dist, tensoRF_id, agg_id)
        if shift is None:
            alpha = Raw2Alpha.apply(sigma_feature.flatten(), self.density_shift, self.stepSize * self.distance_scale).reshape(sigma_feature.shape)
        else:
            alpha = Raw2Alpha_randstep.apply(sigma_feature.flatten(), self.density_shift, (shift * self.distance_scale)[ray_id].contiguous()).reshape(sigma_feature.shape)
        # print("alpha", alpha.shape, ray_id.shape, len(torch.unique(ray_id)), torch.unique(ray_id))

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

        app_features = self.compute_appfeature_geo(local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, local_kernel_dist, tensoRF_id, agg_id, ray_id, dir_gindx_s=dir_gindx_s, dir_gindx_l=dir_gindx_l, dir_gweight_l=dir_gweight_l)
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
                z_val = t_min[ray_id] + step_id * (shift[ray_id] if shift is not None else self.stepSize)
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

    def remask_aggid(self, agg_id, tensor_mask, main_mask):
        holder = torch.zeros((len(main_mask)), device=main_mask.device, dtype=torch.int64)
        agg_id = agg_id[tensor_mask]
        holder[main_mask] = torch.arange(0, torch.sum(main_mask).cpu().item(), device=main_mask.device, dtype=torch.int64)
        return holder[agg_id]

    def sample_ray_cuda(self, rays_o, rays_d, use_mask=True, random=False, N_samples=-1):
        '''Sample query points on rays.
        All the output points are sorted from near to far.
        Input:
            rays_o, rayd_d:   both in [N, 3] indicating ray configurations.
        Output:
            ray_pts:          [M, 3] storing all the sampled points.
            ray_id:           [M]    the index of the ray of each point.
            step_id:          [M]    the i'th step on a ray of each point.
        '''
        near, far = self.near_far
        rays_o = rays_o.view(-1,3).contiguous()
        rays_d = rays_d.view(-1,3).contiguous()
        if random:
            shift = (torch.rand(rays_o.shape[0], dtype=rays_d.dtype, device=rays_d.device) - 0.5) * self.stepSize
            ray_pts, mask_outbbox, ray_id, step_id, N_steps, t_min, t_max = render_utils_cuda.sample_pts_on_rays_dist(
                rays_o, rays_d, self.aabb[0], self.aabb[1], near, far, self.stepSize, shift)
        else:
            ray_pts, mask_outbbox, ray_id, step_id, N_steps, t_min, t_max = render_utils_cuda.sample_pts_on_rays(rays_o, rays_d, self.aabb[0], self.aabb[1], near, far, self.stepSize)

        mask_inbbox = ~mask_outbbox
        if use_mask:
            ray_pts = ray_pts[mask_inbbox]
            ray_id = ray_id[mask_inbbox]
            step_id = step_id[mask_inbbox]

        # print("t_min", t_min.shape, step_id.shape, ray_pts.shape)
        return ray_pts, t_min, ray_id, step_id

    @torch.no_grad()
    def shrink(self, new_aabb):
        print("====> shrinking ...")
        xyz_min, xyz_max = new_aabb
        t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (xyz_max - self.aabb[0]) / self.units
        # print(new_aabb, self.aabb)
        # print(t_l, b_r,self.alphaMask.alpha_volume.shape)
        t_l, b_r = torch.round(torch.round(t_l)).long(), torch.round(b_r).long() + 1
        b_r = torch.stack([b_r, self.gridSize]).amin(0)
        if not torch.all(self.alphaMask.gridSize == self.gridSize):
            t_l_r, b_r_r = t_l / (self.gridSize - 1), (b_r - 1) / (self.gridSize - 1)
            correct_aabb = torch.zeros_like(new_aabb)
            correct_aabb[0] = (1 - t_l_r) * self.aabb[0] + t_l_r * self.aabb[1]
            correct_aabb[1] = (1 - b_r_r) * self.aabb[0] + b_r_r * self.aabb[1]
            print("aabb", new_aabb, "\ncorrect aabb", correct_aabb)
            new_aabb = correct_aabb

        self.aabb = new_aabb
        self.update_stepSize(self.local_dims[:3])



class PointTensorCP(PointTensorBase):
    def __init__(self, aabb, gridSize, device, **kargs):
        super(PointTensorCP, self).__init__(aabb, gridSize, device, **kargs)


    def init_svd_volume(self, local_dims, device, init=True):
        self.density_line = self.init_one_svd(self.density_n_comp[0], local_dims[:3], 0.2, device)
        self.app_line = self.init_one_svd(self.app_n_comp[0], local_dims[:3], 0.2, device)
        if len(self.local_dims) > 3:
            self.theta_line = self.init_angle_svd(self.app_n_comp[0], local_dims[3], 0.2, device)
            self.phi_line = self.init_angle_svd(self.app_n_comp[0], local_dims[4], 0.2, device)
        else:
            self.theta_line, self.phi_line = None, None
        if self.args.rot_init is not None and init:
            self.pnt_rot = torch.nn.Parameter(torch.as_tensor(self.args.rot_init, device="cuda", dtype=torch.float32).repeat(len(self.geo), 1), requires_grad=self.args.rotgrad>0)
        self.basis_mat = torch.nn.Linear(self.app_n_comp[0], self.app_dim, bias=False).to(device)

    def init_angle_svd(self, n_component, local_dim, scale, device):
        return torch.nn.Parameter(scale * torch.randn((1, n_component, local_dim), device=device), requires_grad=True).to(device)

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
                     {'params': self.basis_mat.parameters(), 'lr':lr_init_network, 'skip_zero_grad': (False)}]
        if len(self.local_dims) > 3:
            grad_vars += [
                {'params': self.theta_line, 'lr': lr_init_spatialxyz, 'skip_zero_grad': (skip_zero_grad)},
                {'params': self.phi_line, 'lr': lr_init_spatialxyz, 'skip_zero_grad': (skip_zero_grad)}
            ]

        if isinstance(self.renderModule, torch.nn.Module):
            grad_vars += [{'params':self.renderModule.parameters(), 'lr':lr_init_network, 'skip_zero_grad': (False)}]
        return grad_vars

    def get_geoparam_groups(self, lr_init_geo=0.03):
        grad_vars = [
            {'params': self.pnt_rot, 'lr': lr_init_geo, 'skip_zero_grad': (False), "weight_decay": 0.0}
        ]
        return grad_vars



    def ind_intrp_line_map_batch(self, vecModes, density_lines, local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, tensoRF_id):

        line_s = torch.stack([density_lines[0][tensoRF_id, :, local_gindx_s[..., vecModes[0]]], density_lines[1][tensoRF_id, :, local_gindx_s[..., vecModes[1]]], density_lines[2][tensoRF_id, :, local_gindx_s[..., vecModes[2]]]], dim=-1)
        line_l = torch.stack([density_lines[0][tensoRF_id, :, local_gindx_l[..., vecModes[0]]], density_lines[1][tensoRF_id, :, local_gindx_l[..., vecModes[1]]], density_lines[2][tensoRF_id, :, local_gindx_l[..., vecModes[2]]]], dim=-1)

        line_s_gweight = torch.stack([local_gweight_s[:, None, vecModes[0]], local_gweight_s[:, None, vecModes[1]], local_gweight_s[:, None, vecModes[2]]], dim=-1)
        line_l_gweight = torch.stack([local_gweight_l[:, None, vecModes[0]], local_gweight_l[:, None, vecModes[1]], local_gweight_l[:, None, vecModes[2]]], dim=-1)

        return line_s * line_s_gweight + line_l * line_l_gweight


    def ind_intrp_line_map_batch_prod(self, vecModes, density_lines, local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, tensoRF_id):
        return (density_lines[0][tensoRF_id, :, local_gindx_s[..., vecModes[0]]] * local_gweight_s[:, None, vecModes[0]] + density_lines[0][tensoRF_id, :, local_gindx_l[..., vecModes[0]]] * local_gweight_l[:, None, vecModes[0]]) *  (density_lines[1][tensoRF_id, :, local_gindx_s[..., vecModes[1]]] * local_gweight_s[:, None, vecModes[1]] + density_lines[1][tensoRF_id, :, local_gindx_l[..., vecModes[1]]] * local_gweight_l[:, None, vecModes[1]]) * (density_lines[2][tensoRF_id, :, local_gindx_s[..., vecModes[2]]] * local_gweight_s[:, None, vecModes[2]] + density_lines[2][tensoRF_id, :, local_gindx_l[..., vecModes[2]]] * local_gweight_l[:, None, vecModes[2]])


    def ind_intrp_line_batch(self, density_line, local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l):
        # print("density_line", density_line.shape)
        # print("local_gindx_s", torch.min(local_gindx_s), torch.max(local_gindx_s))
        # print("local_gindx_l", torch.min(local_gindx_l), torch.max(local_gindx_l))
        # return (density_line[0, :, local_gindx_s] * local_gweight_s[None, :] + density_line[0, :, local_gindx_l] * local_gweight_l[None, :]).permute((1,0))
        b_inds = torch.zeros([len(local_gindx_s)], device=density_line.device, dtype=torch.int64)
        return density_line[b_inds, :, local_gindx_s] * local_gweight_s[:, None] + density_line[b_inds, :, local_gindx_l] * local_gweight_l[:, None]

    @torch.no_grad()
    def up_sampling_Vector(self, density_line_coef, app_line_coef, theta_line_coef, phi_line_coef, res_target):
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            density_line_coef[i] = torch.nn.Parameter(
                F.interpolate(density_line_coef[i].data, size=(res_target[vec_id]+1), mode='linear',
                              align_corners=True))
            app_line_coef[i] = torch.nn.Parameter(
                F.interpolate(app_line_coef[i].data, size=(res_target[vec_id]+1), mode='linear', align_corners=True))
        if len(self.local_dims) > 3:
            theta_line_coef = torch.nn.Parameter(F.interpolate(theta_line_coef.data, size=(res_target[3]), mode='linear', align_corners=True))
            phi_line_coef = torch.nn.Parameter(F.interpolate(phi_line_coef.data, size=(res_target[4]), mode='linear', align_corners=True))

        return density_line_coef, app_line_coef, theta_line_coef, phi_line_coef

    @torch.no_grad()
    def upsample_volume_grid(self, res_target, reset_feat=False):
        print(f'upsamping to local dims: {self.local_dims} to {res_target}')
        self.local_dims = torch.as_tensor(res_target, device=self.local_dims.device, dtype=self.local_dims.dtype)
        if reset_feat:
            self.init_svd_volume(self.local_dims, self.local_dims.device, init=False)
        else:
            self.up_sampling_Vector(self.density_line, self.app_line, self.theta_line, self.phi_line, res_target)
        self.update_stepSize(self.local_dims)


    def compute_densityfeature_geo(self, local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, local_kernel_dist, tensoRF_id, agg_id):
        # plane + line basis

        # sigma_feature =  torch.sum(torch.prod(self.ind_intrp_line_map_batch(self.vecMode, self.density_line, local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, tensoRF_id), dim=-1), dim=1, keepdim=True)

        sigma_feature =  torch.sum(self.ind_intrp_line_map_batch_prod(self.vecMode, self.density_line, local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, tensoRF_id), dim=1, keepdim=True)

        sigma_feature = self.agg_tensoRF_at_samples(local_kernel_dist, agg_id, sigma_feature).squeeze(-1)
        return sigma_feature


    def compute_appfeature_geo(self, local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, local_kernel_dist, tensoRF_id, agg_id, ray_id, dir_gindx_s=None, dir_gindx_l=None, dir_gweight_l=None):
        # plane + line basis
        # line_coef_point = torch.prod(self.ind_intrp_line_map_batch(self.vecMode, self.app_line, local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, tensoRF_id), dim=-1)

        line_coef_point = self.ind_intrp_line_map_batch_prod(self.vecMode, self.app_line, local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, tensoRF_id)

        app_feat = self.agg_tensoRF_at_samples(local_kernel_dist, agg_id, line_coef_point)
        if dir_gindx_s is not None:
            app_feat = app_feat * self.ind_intrp_line_batch(self.theta_line, dir_gindx_s[0, ray_id], dir_gindx_l[0, ray_id], (1 - dir_gweight_l[0])[ray_id], dir_gweight_l[0, ray_id]) * self.ind_intrp_line_batch(self.phi_line, dir_gindx_s[1, ray_id], dir_gindx_l[1,ray_id], (1 - dir_gweight_l[1])[ray_id], dir_gweight_l[1, ray_id])

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


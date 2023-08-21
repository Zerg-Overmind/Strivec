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

def vis_box_pca(cluster_raw_pnts, geo, pca_cluster_newpnts, cluster_raw_mean, local_ranges, args, pnt_rmatrix, sep=False, subdir="rot_tensoRF"):
    for l in range(len(geo)):
        if not sep:
            draw_box_pca(geo[l][..., :3], None, local_ranges[l], f'{args.basedir}/{args.expname}', l+1000, args, pnt_rmatrix[l].cuda(), subdir=subdir) 
        else:
            draw_sep_box_pca(cluster_raw_pnts[l], geo[l][..., :3], None, local_ranges[l],
                         f'{args.basedir}/{args.expname}', l, args, pnt_rmatrix[l].cuda(), subdir=subdir)
def vis_box(geo, args):
    for l in range(len(geo)):
        draw_box(geo[l][..., :3], args.local_range[l], f'{args.basedir}/{args.expname}', l)

class StrivecBase_hier(TensorBase):
    def __init__(self, aabb, gridSize, device, density_n_comp=8, appearance_n_comp=24, app_dim=27, shadingMode='MLP_PE', alphaMask=None, near_far=[2.0, 6.0], density_shift=-10, alphaMask_thres=0.001, distance_scale=25, rayMarch_weight_thres=0.0001, pos_pe=6, view_pe=6, fea_pe=6, featureC=128, step_ratio=2.0, fea2denseAct='softplus', local_dims=None, cluster_dict=None, geo=None, args=None, up_stage=0):
        super(TensorBase, self).__init__()
        assert geo is not None, "No geo loaded"
        self.args = args
        self.geo = geo
        self.pnt_xyz = [geo_lvl[..., :3].cuda().contiguous() for geo_lvl in self.geo]
        self.density_n_comp = density_n_comp
        self.app_n_comp = appearance_n_comp
        self.app_dim = app_dim
        self.aabb = aabb
        self.alphaMask = alphaMask
        self.device = device
        self.local_range = torch.as_tensor(args.local_range, device=device, dtype=torch.float32)
        self.local_dims = torch.as_tensor(local_dims, device=device, dtype=torch.int64)
        self.lvl = len(self.local_dims)
        self.density_shift = density_shift
        self.alphaMask_thres = alphaMask_thres
        self.distance_scale = distance_scale
        self.rayMarch_weight_thres = rayMarch_weight_thres
        self.fea2denseAct = fea2denseAct
        self.max_tensoRF = args.rot_max_tensoRF if args.rot_max_tensoRF is not None else args.max_tensoRF
        self.K_tensoRF = args.rot_K_tensoRF if args.rot_K_tensoRF is not None else args.K_tensoRF
        self.K_tensoRF = self.max_tensoRF if self.K_tensoRF is None else self.K_tensoRF
        self.KNN = (args.rot_KNN > 0) if args.rot_KNN is not None else (args.KNN > 0)
        self.near_far = near_far
        self.step_ratio = step_ratio 
        self.update_stepSize(self.local_dims)
        self.vecMode = [2, 1, 0]
        self.init_svd_volume(local_dims, device)
        self.shadingMode, self.pos_pe, self.view_pe, self.fea_pe, self.featureC = shadingMode, pos_pe, view_pe, fea_pe, featureC
        self.init_render_func(shadingMode, pos_pe, view_pe, fea_pe, featureC, device, app_dim=app_dim[0] if args.radiance_add > 0 else None)

        if cluster_dict is not None:
            if len(cluster_dict["cluster_xyz"][0]) > 1:
               self.geo_xyz = [torch.as_tensor(cluster_dict["cluster_xyz"][l][..., :3], device=self.device).contiguous() for l in range(len(cluster_dict["cluster_xyz"]))]
               self.box_length = cluster_dict["box_length"]
               self.stds = cluster_dict["stds"]
            else:
               self.geo_xyz = [torch.as_tensor(cluster_dict["cluster_xyz"][l][0][..., :3], device=self.device).contiguous() for l in range(len(cluster_dict["cluster_xyz"]))]
               self.box_length = cluster_dict["box_length"][0]
               self.stds = cluster_dict["stds"][0]
            if self.lvl < len(self.geo_xyz):
                self.geo_xyz = [torch.cat(self.geo_xyz, dim=0).contiguous()]
                self.box_length = [np.concatenate(self.box_length)]
                self.stds = [np.concatenate(self.stds)]
            self.pnt_xyz = self.geo_xyz
          

            # self.local_range, self.local_dims = self.set_local_range_dim(self.geo_xyz, self.box_length, geo_cluster, pca_cluster_newpnts, pca_axis, self.stds)
            #self.local_range, self.local_dims = self.set_local_range_dim(self.geo_xyz, self.box_length, None, None, None, self.stds)
            self.pnt_rmatrix = [torch.transpose(torch.as_tensor(cluster_dict['pca_axis'][l], dtype=torch.float32, device=self.device), 1, 2).contiguous() for l in range(self.lvl)]
            # vis_box_pca(geo_cluster, self.geo_xyz, pca_cluster_newpnts, cluster_raw_mean, self.local_range, args, self.pnt_rmatrix, sep=False)
            ####
            vis_box_pca(cluster_dict["cluster_pnts"], self.geo_xyz, None, None, self.local_range, args, self.pnt_rmatrix, sep=False)
            #vis_box_pca(cluster_dict["cluster_pnts"], self.geo_xyz, None, None, self.local_range, args, self.pnt_rmatrix, sep=True)
            ###
 

        draw_hier_box(self.pnt_xyz, self.local_range, os.path.join(args.basedir, args.expname), step=0, rot_m=None)

 #       draw_box(self.pnt_xyz[0], self.local_range[0], logfolder, 0, rot_m=None)
        #exit()
    def set_local_range_dim(self, geo_xyz, box_length, geo_cluster, pca_cluster_newpnts, pca_axis, stds):
        local_dims, local_range = [], []
       
        for l in range(self.lvl):
            unit = self.args.local_unit[l]
            print("unit of level {} is {}".format(self.lvl, unit))
            dims = np.ceil((box_length[l] * self.args.dilation_ratio[l] - unit) * 0.5  / unit).astype(np.int16)
            local_dims.append((torch.as_tensor(dims, dtype=torch.int16, device=self.device) * 2 + 1).contiguous())
            local_range.append(torch.as_tensor((dims+0.5) * unit, dtype=torch.float32, device=self.device).contiguous())

        return local_range, local_dims

    def update_stepSize(self, local_dims):
        print("aabb", self.aabb.view(-1))
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invaabbSize = 2.0/self.aabbSize
        if self.args.tensoRF_shape == "cube":
            self.lvl_units = [(2 * self.local_range[l] / local_dims[l]) for l in range(self.lvl)]
            self.units = self.lvl_units[self.args.unit_lvl]
            self.stepSize = torch.mean(self.units) * self.step_ratio
            print(torch.mean(self.units) , self.step_ratio)
            self.gridSize = torch.ceil(self.aabbSize / self.units).long().to(self.device)
            self.radius = torch.norm(self.local_range, dim=-1).cpu().tolist()
            print("radius, furthest shading to tensoRF distance: ", self.radius)
        else:
            print("not implemented")
            exit()
            self.units = torch.stack([(self.local_range[1] - self.local_range[0]) / local_dims[0], torch.as_tensor(np.pi * 2, device=self.local_range.device, dtype=self.local_range.dtype) / local_dims[1], torch.as_tensor(np.pi, device=self.local_range.device, dtype=self.local_range.dtype) / local_dims[2]], dim=0)
            self.units_3 = torch.stack([self.units[0], self.units[0], self.units[0]], dim=0)
            self.radiush = self.local_range[1].cpu().item()
            self.radiusl = self.local_range[0].cpu().item()
            self.stepSize = self.units[0] * self.step_ratio
            self.gridSize = torch.ceil(self.aabbSize / self.units[0]).long().to(self.device)

        if len(local_dims[0]) > 3:
            self.view_units = [torch.stack([torch.as_tensor(np.pi * 2, device=self.local_range.device, dtype=self.local_range.dtype) / local_dims[i][3], torch.as_tensor(np.pi, device=self.local_range.device, dtype=self.local_range.dtype) / local_dims[i][4]], dim=0) for i in range(self.lvl)]

        self.aabbDiag = torch.norm(self.aabbSize)
        print("self.aabbDiag", self.aabbDiag)
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
        if alpha_mask.any():
            filtered_xyz = xyz_locs[alpha_mask]
            local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, local_kernel_dist, tensoRF_id, agg_id = self.sample_2_tensoRF_cvrg_hier(filtered_xyz, pnt_rmatrix=pnt_rmatrix, rotgrad=False)

            sigma_feature = self.compute_densityfeature_geo(local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, local_kernel_dist, tensoRF_id, agg_id, sample_num=len(filtered_xyz))
            validsigma = self.feature2density(sigma_feature)
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

    def agg_tensoRF_at_samples(self, dist, agg_id, value, out=None, outweight=None):
        weights = self.agg_weight_func(dist)
        weight_sum = segment_coo(
            src=weights,
            index=agg_id,
            out=outweight,
            reduce='sum')
        agg_value = segment_coo(
            src=weights * value,
            index=agg_id,
            out=out,
            reduce='sum')
        return agg_value / torch.clamp(weight_sum, min=1), weight_sum > 0

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
        for idx_chunk in func(idx_chunks):
            rays_chunk = all_rays[idx_chunk].to(self.device)

            rays_o, rays_d = rays_chunk[..., :3], rays_chunk[..., 3:6]
            xyz_sampled, _, xyz_inbbox = self.sample_ray(rays_o, rays_d, N_samples=N_samples, is_train=False)

            if bbox_only:
                if self.args.ub360 ==1:
                   vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
                   rate_a = (self.aabb[1] - rays_o) / vec
                   rate_b = (self.aabb[0] - rays_o) / vec
                   t_min = torch.minimum(rate_a, rate_b).amax(-1)  # .clamp(min=near, max=far)
                   t_max = torch.maximum(rate_a, rate_b).amin(-1)  # .clamp(min=near, max=far)
                   mask_inbbox = t_max > t_min
    
                else:
                   vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
                   rate_a = (self.aabb[1] - rays_o) / vec
                   rate_b = (self.aabb[0] - rays_o) / vec
                   t_min = torch.minimum(rate_a, rate_b).amax(-1)  # .clamp(min=near, max=far)
                   t_max = torch.maximum(rate_a, rate_b).amin(-1)  # .clamp(min=near, max=far)
                   mask_inbbox = t_max > t_min

            else:
                mask_inbbox = (self.alphaMask.sample_alpha(xyz_sampled).view(xyz_sampled.shape[:-1]) > 0).any(-1)
            if cvrg:
                mask_inrange = filter_ray_by_cvrg(xyz_sampled, xyz_inbbox, self.units if self.args.tensoRF_shape == "cube" else self.units_3, self.aabb[0], self.aabb[1], self.tensoRF_cvrg_filter)
                mask_filtered.append(mask_inrange.view(xyz_sampled.shape[:-1]).any(-1).cpu())
            else:
                tensoRF_per_ray = filter_ray_by_projection(rays_o, rays_d, self.pnt_xyz, self.local_range)
                mask_inrange = tensoRF_per_ray > 0
                mask_filtered.append((mask_inbbox * mask_inrange).cpu())
                tensoRF_per_ray_lst.append(tensoRF_per_ray.cpu())
        mask_filtered = torch.cat(mask_filtered).view(all_rays.shape[:-1])

        ### here
        #mask_filtered = torch.ones_like(mask_filtered)

        tensoRF_per_ray = torch.cat(tensoRF_per_ray_lst).view(all_rays.shape[:-1]) if len(tensoRF_per_ray_lst) > 0 else None

        if apply_filter:
            tensoRF_per_ray = None if tensoRF_per_ray is None else tensoRF_per_ray[mask_filtered]
            print(f'Ray filtering done! takes {time.time() - tt} s. ray mask ratio: {torch.sum(mask_filtered) / N}')

        return mask_filtered, tensoRF_per_ray


    @torch.no_grad()
    def updateAlphaMask(self):
        gridSize = self.gridSize
        alpha, dense_xyz = self.getDenseAlpha(gridSize)
        dense_xyz = dense_xyz.transpose(0, 2).contiguous()
        alpha = alpha.clamp(0, 1).transpose(0, 2).contiguous()[None, None]
        total_voxels = gridSize[0] * gridSize[1] * gridSize[2]

        ks = 3
        alpha = F.max_pool3d(alpha, kernel_size=ks, padding=ks // 2, stride=1).view((gridSize[2], gridSize[1], gridSize[0])) 
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
            alpha[i] = self.compute_alpha(samples[i].view(-1,3), self.stepSize, pnt_rmatrix).view((gridSize[1], gridSize[2])) 
        return alpha, samples


    def filter_by_points(self, xyz_sampled):
        xyz_dist = torch.abs(xyz_sampled[..., None, :] - self.pnt_xyz[None, ...])
        mask_inrange = torch.all(xyz_dist <= self.local_range[None, None, :], dim=-1) 
        mask_inrange = torch.any(mask_inrange.view(mask_inrange.shape[0], -1), dim=-1)
        return mask_inrange


    def select_by_points(self, xyz_sampled, ray_id, step_id=None):
        xyz_dist = torch.abs(xyz_sampled[..., None, :] - self.pnt_xyz[None, ...]) 
        mask_inrange = torch.all(xyz_dist <= self.local_range[None, None, :], dim=-1) 
        mask_inds = torch.nonzero(mask_inrange)
        xyz_sampled = xyz_sampled[mask_inds[..., 0], ...]
        ray_id = ray_id[mask_inds[..., 0], ...]
        step_id = step_id[mask_inds[..., 0], ...] if step_id is not None else None
        return xyz_sampled, ray_id, step_id, mask_inds[..., 1]

    def create_sample_map(self):
        # initialize local tensors acoording to initial geometry
        print("start create mapping")
        self.tensoRF_cvrg_inds, self.tensoRF_count, self.tensoRF_topindx = [], [], []
        for l in range(self.lvl):
            if self.args.tensoRF_shape == "cube":
                if self.args.rot_init is None:
                    # torch::Tensor pnt_xyz,
                    #     torch::Tensor gridSize,
                    #     torch::Tensor xyz_min,
                    #     torch::Tensor xyz_max,
                    #     torch::Tensor units,
                    #     torch::Tensor local_range,
                    #     torch::Tensor local_dims, const int max_tensoRF
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()           
                    tensoRF_cvrg_inds, tensoRF_count, tensoRF_topindx = search_geo_hier_cuda.build_tensoRF_map_hier(self.pnt_xyz[l], self.gridSize, self.aabb[0], self.aabb[1], self.units, self.local_range[l], self.local_dims[l], self.max_tensoRF[l])
                else:
                    print("no implementation")
                    exit()
                    tensoRF_cvrg_inds, tensoRF_count, tensoRF_topindx = search_geo_cuda.build_sphere_tensoRF_map(self.pnt_xyz[l], self.gridSize, self.aabb[0], self.aabb[1], self.units, 0.0, self.radius[l], self.local_dims[l, :3], self.max_tensoRF[l])
            elif self.args.tensoRF_shape == "sphere":
                print("no implementation")
                exit()
                tensoRF_cvrg_inds, tensoRF_count, tensoRF_topindx = search_geo_cuda.build_sphere_tensoRF_map(self.pnt_xyz[l], self.gridSize, self.aabb[0], self.aabb[1], self.units, self.radiusl, self.radiush, self.local_dims[l, :3], self.max_tensoRF[l])
            tensoRF_cvrg_inds, tensoRF_count, tensoRF_topindx = tensoRF_cvrg_inds.contiguous(), tensoRF_count.contiguous(), tensoRF_topindx.contiguous()
            self.tensoRF_cvrg_inds.append(tensoRF_cvrg_inds)
            self.tensoRF_count.append(tensoRF_count)
            self.tensoRF_topindx.append(tensoRF_topindx)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        if self.args.filterall == 0:
            self.tensoRF_cvrg_filter = torch.any(torch.stack(self.tensoRF_cvrg_inds, dim=-1) >= 0, dim=-1).contiguous() if len(self.tensoRF_cvrg_inds) > 0 else (self.tensoRF_cvrg_inds[0] >= 0).contiguous()
        else:
            self.tensoRF_cvrg_filter = torch.all(torch.stack(self.tensoRF_cvrg_inds, dim=-1) >= 0, dim=-1).contiguous() if len(self.tensoRF_cvrg_inds) > 0 else (self.tensoRF_cvrg_inds[0] >= 0).contiguous()



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

            if self.args.ub360 == 1: 
               # roughly following DVGO
               scene_center = (self.aabb[0] + self.aabb[1]) * 0.5
               scene_radius = (self.aabb[1] - self.aabb[0]) * 0.5

               rays_o_aft = (rays_o - scene_center) / scene_radius
               rays_d_aft = rays_d / rays_d.norm(dim=-1, keepdim=True)
     
               bg_len = 0.4
               world_size = sum((self.aabb[1]-self.aabb[0])**2)
               N_inner = 160
               N_outer = N_inner
               b_inner = torch.linspace(0, 2, N_inner+1)
               b_outer = 2 / torch.linspace(1, 1/128, N_outer+1)
               t = torch.cat([ 
                      (b_inner[1:] + b_inner[:-1]) * 0.5,
                      (b_outer[1:] + b_outer[:-1]) * 0.5,
               ])
              
               ray_pts = (rays_o_aft[:,None,:].cpu() + rays_d_aft[:,None,:].cpu() * t[None,:,None]).to(dtype=torch.float32)      
               norm = ray_pts.abs().amax(dim=-1, keepdim=True)
               mask_valid_1 = (norm<=1)
               ray_pts = torch.where(
               mask_valid_1,
               ray_pts,
               ray_pts / norm * ((1+bg_len) - bg_len/norm)
               )
              
               
               ray_id = torch.arange(ray_pts.shape[:2][0]).view(-1,1).expand(ray_pts.shape[:2]).flatten().cuda()
               step_id = torch.arange(ray_pts.shape[:2][1]).view(1,-1).expand(ray_pts.shape[:2]).flatten().cuda()
               
               ray_pts = ray_pts.reshape(-1,3).cuda()
               t_min = torch.zeros_like(ray_id).cuda()
               mask_valid = torch.ones_like(step_id,dtype=torch.bool).cuda()
               stepSize = 0.02
            
               ### our cuda implementation of nonuniform ray sampling (scene warping) and find TopK for each sample point (considering tensor cvrg), but is not good if the initial geometry is not good.
               #ray_pts, mask_valid, ray_id, step_id, N_steps, t_min, t_max = search_geo_cuda.sample_pts_on_rays_cvrg_360(rays_o_aft, rays_d_aft, self.tensoRF_cvrg_filter, self.units, self.aabb[0], self.aabb[1], near, far, stepSize)

               
            else:
               ray_pts, mask_valid, ray_id, step_id, N_steps, t_min, t_max = search_geo_cuda.sample_pts_on_rays_cvrg(rays_o, rays_d, self.tensoRF_cvrg_filter, self.units, self.aabb[0], self.aabb[1], near, far, self.stepSize)
               
        elif self.args.tensoRF_shape == "sphere":
            print("no implementation")
            exit() 
            # assert not random, "random has no implementation for sphere!!!"
            # assert not ji, "ji has no implementation for sphere!!!"
            # assert args.rot_init is None, "args.rot_init has no implementation for sphere!!!"
            ray_pts, mask_valid, ray_id, step_id, N_steps, t_min, t_max = search_geo_cuda.sample_pts_on_rays_sphere_cvrg(rays_o, rays_d, self.pnt_xyz, self.tensoRF_cvrg_inds, self.tensoRF_count, self.tensoRF_topindx, self.units, self.radiusl, self.radiush, self.aabb[0], self.aabb[1], near, far, self.stepSize)

        
        if use_mask:
            ray_pts = ray_pts[mask_valid]
            ray_id = ray_id[mask_valid]
            step_id = step_id[mask_valid]
        
        return ray_pts, t_min, ray_id, step_id, shift, pnt_rmatrix


    def filter_xyz_cvrg(self, xyz_sampled, pnt_rmatrix=None):
        if self.args.tensoRF_shape == "cube":
            if self.args.rot_init is None:
                mask = search_geo_cuda.filter_xyz_cvrg(xyz_sampled.contiguous(), self.aabb[0], self.aabb[1], self.units, self.tensoRF_cvrg_filter)
            else:
                print("no implementation")
                exit()
                mask = search_geo_cuda.filter_xyz_rot_cvrg(self.pnt_xyz, pnt_rmatrix, xyz_sampled.contiguous(), self.aabb[0], self.aabb[1], self.units,self.tensoRF_cvrg_inds, self.tensoRF_count, self.tensoRF_topindx, self.local_range)
        elif self.args.tensoRF_shape == "sphere":
            print("no implementation")
            exit()
            mask = search_geo_cuda.filter_xyz_sphere_cvrg(xyz_sampled.contiguous(), self.aabb[0], self.aabb[1], self.units, self.radiusl, self.radiush, self.tensoRF_cvrg_inds, self.tensoRF_count, self.tensoRF_topindx, self.pnt_xyz)
        return mask

    def normalize_coord_tsrf(self, sample_tensoRF_pos):
        return sample_tensoRF_pos / self.local_range[None, ...]


    def cvrg_inds_center2pnts(self, tensoRF_cvrg_inds):
        for l in range(self.lvl):
            inds = torch.nonzero(tensoRF_cvrg_inds[l]>=0)
            pnts = self.aabb[0][None, ...] + (inds+0.5) * self.units[None, ...]
            print("center pnts", torch.min(pnts, dim=0)[0], torch.max(pnts, dim=0)[0])
            np.savetxt("log/ship_hier_try/cvrg_inds_center2pnts_lvl{}.txt".format(l), pnts.cpu().numpy(), delimiter=";")


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


    def sample_2_tensoRF_cvrg_hier(self, xyz_sampled, pnt_rmatrix=None, rotgrad=False):
        # to find the grid indeces of each sample point's TopK tensors for feature interpolation and aggregation 
        local_gindx_s_lst, local_gindx_l_lst, local_gweight_s_lst, local_gweight_l_lst, local_kernel_dist_lst, tensoRF_id_lst, agg_id_lst = [], [], [], [], [], [], []
        for l in range(self.lvl):
            mask = None
            if self.args.tensoRF_shape == "cube":
                local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, local_kernel_dist, tensoRF_id, agg_id = search_geo_hier_cuda.sample_2_tensoRF_cvrg_hier(xyz_sampled.contiguous(), self.aabb[0], self.aabb[1], self.units, self.lvl_units[l], self.local_range[l], self.local_dims[l,:3], self.tensoRF_cvrg_inds[l], self.tensoRF_count[l], self.tensoRF_topindx[l], self.pnt_xyz[l], self.K_tensoRF[l], self.KNN)
            elif self.args.tensoRF_shape == "sphere":
                print("no implementation")
                exit()
                mask, local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, local_kernel_dist, tensoRF_id, agg_id = search_geo_cuda.sample_2_sphere_tensoRF_cvrg(xyz_sampled.contiguous(), self.aabb[0], self.aabb[1], self.units, self.radiusl, self.radiush, self.local_dims[l,:3], self.tensoRF_cvrg_inds[l], self.tensoRF_count[l], self.tensoRF_topindx[l], self.pnt_xyz[l], self.K_tensoRF[l], self.KNN)
            if mask is not None:
                local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, local_kernel_dist, tensoRF_id = local_gindx_s[mask, :], local_gindx_l[mask, :], local_gweight_s[mask, :], local_gweight_l[mask, :], local_kernel_dist[mask], tensoRF_id[mask]

            local_gindx_s_lst.append(local_gindx_s)
            local_gindx_l_lst.append(local_gindx_l)
            local_gweight_s_lst.append(local_gweight_s)
            local_gweight_l_lst.append(local_gweight_l)
            local_kernel_dist_lst.append(local_kernel_dist)
            tensoRF_id_lst.append(tensoRF_id)
            agg_id_lst.append(agg_id)
        return local_gindx_s_lst, local_gindx_l_lst, local_gweight_s_lst, local_gweight_l_lst, local_kernel_dist_lst, tensoRF_id_lst, agg_id_lst


    def inds_cvrg(self, xyz_sampled):
        global_gindx, local_gindx, tensoRF_id = search_geo_cuda.inds_cvrg(xyz_sampled.contiguous(), self.aabb[0], self.aabb[1], self.units, self.local_range, self.local_dims[:3], self.tensoRF_cvrg_inds, self.tensoRF_count, self.tensoRF_topindx, self.pnt_xyz)
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



    def forward(self, rays_chunk, white_bg=True, is_train=False, ray_type=0, N_samples=-1, return_depth=0,
                tensoRF_per_ray=None, eval=False, rot_step=False, depth_bg=True):

        # sample points
        viewdirs = rays_chunk[:, 3:6]
        if len(self.local_dims[0]) > 3:
            print("not implemented")
            exit()
            dir_gindx_s, dir_gindx_l, dir_gweight_l = self.view_decompose(viewdirs)
        else:
            dir_gindx_s, dir_gindx_l, dir_gweight_l = None, None, None

        N, _ = rays_chunk.shape
        shp_rand = (self.args.shp_rand > 0) and (not eval)
        ji = (self.args.ji > 0) and (not eval)
        xyz_sampled, t_min, ray_id, step_id, shift, pnt_rmatrix = self.sample_ray_cvrg_cuda(rays_chunk[:, :3], viewdirs, use_mask=True, N_samples=N_samples, random=shp_rand, ji=ji)

        mask_any = True
        if self.alphaMask is not None:
            mask = self.alphaMask.sample_alpha(xyz_sampled) > 0
            mask_any = mask.any()
            if mask_any:
                xyz_sampled = xyz_sampled[mask]
                ray_id = ray_id[mask]
                if return_depth:
                    step_id = step_id[mask]

        if ray_id is None or len(ray_id) == 0 or not mask_any:
            return torch.full([N, 3], 1.0 if (white_bg or (is_train and torch.rand((1,)) < 0.5)) else 0.0, device="cuda", dtype=torch.float32), rays_chunk[..., -1].detach(), None, None, None
        
        local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, local_kernel_dist, tensoRF_id, agg_id = self.sample_2_tensoRF_cvrg_hier(xyz_sampled, pnt_rmatrix=pnt_rmatrix, rotgrad=rot_step)
        sigma_feature = self.compute_densityfeature_geo(local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, local_kernel_dist, tensoRF_id, agg_id, sample_num=len(ray_id))
      
        if shift is None:
            alpha = Raw2Alpha.apply(sigma_feature.flatten(), self.density_shift, self.stepSize * self.distance_scale).reshape(sigma_feature.shape)
        else:
            alpha = Raw2Alpha_randstep.apply(sigma_feature.flatten(), self.density_shift, (shift * self.distance_scale)[ray_id].contiguous()).reshape(sigma_feature.shape)
        weights, bg_weight = Alphas2Weights.apply(alpha, ray_id, N) #
        mask = weights > self.rayMarch_weight_thres
        if mask.any() and (~mask).any():
            if return_depth:
                step_id = step_id[mask]
            weights = weights[mask]
            ray_id = ray_id[mask]
            holder = torch.zeros((len(mask)), device=mask.device, dtype=torch.int64)
            holder[mask] = torch.arange(0, torch.sum(mask).cpu().item(), device=mask.device, dtype=torch.int64)
            for l in range(self.lvl):
                tensor_mask = mask[agg_id[l]]
                agg_id[l] = holder[agg_id[l][tensor_mask]]
                tensoRF_id[l] = tensoRF_id[l][tensor_mask]
                local_gindx_s[l] = local_gindx_s[l][tensor_mask]
                local_gindx_l[l] = local_gindx_l[l][tensor_mask]
                local_gweight_s[l] = local_gweight_s[l][tensor_mask]
                local_gweight_l[l] = local_gweight_l[l][tensor_mask]
                local_kernel_dist[l] = local_kernel_dist[l][tensor_mask]

        app_features = self.compute_appfeature_geo(local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, local_kernel_dist, tensoRF_id, agg_id, sample_num=len(ray_id), dir_gindx_s=dir_gindx_s, dir_gindx_l=dir_gindx_l, dir_gweight_l=dir_gweight_l)
        rgb = self.renderModule(None, viewdirs[ray_id], app_features)

        rgb_map = segment_coo(
            src=(weights.unsqueeze(-1) * rgb),
            index=ray_id,
            out=torch.zeros([N, 3], device=weights.device, dtype=torch.float32),
            reduce='sum')
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
                depth_map += (bg_weight * 1000) if depth_bg else 0
        else:
            depth_map = None
        rgb_map = rgb_map.clamp(0, 1)
        # print("rgb_map",  rgb_map[0])
        return rgb_map, depth_map, rgb, ray_id, weights # rgb, sigma, alpha, weight, bg_weight

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

        return ray_pts, t_min, ray_id, step_id

    @torch.no_grad()
    def shrink(self, new_aabb):
        print("====> shrinking ...")
        xyz_min, xyz_max = new_aabb
        t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (xyz_max - self.aabb[0]) / self.units
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
        self.update_stepSize(self.local_dims)



class StrivecCP_hier(StrivecBase_hier):
    def __init__(self, aabb, gridSize, device, **kargs):
        super(StrivecCP_hier, self).__init__(aabb, gridSize, device, **kargs)

    def init_svd_volume(self, local_dims, device, init=True):
        self.density_line = self.init_one_svd(self.geo, self.density_n_comp, local_dims, 0.2, device, self.lvl)
        self.app_line = self.init_one_svd(self.geo, self.app_n_comp, local_dims, 0.2, device, self.lvl)
        if len(self.local_dims[0]) > 3:
            self.theta_line = self.init_angle_svd(self.app_n_comp, local_dims, 3, 0.2, device, self.lvl)
            self.phi_line = self.init_angle_svd(self.app_n_comp, local_dims, 4, 0.2, device, self.lvl)
        else:
            self.theta_line, self.phi_line = None, None

        if self.args.rot_init is not None and init:
            self.pnt_rot = torch.nn.ParameterList([torch.nn.Parameter(torch.as_tensor(self.args.rot_init, device="cuda", dtype=torch.float32).repeat(len(geo), 1), requires_grad=self.args.rotgrad>0) for geo in self.geo]).to(device)

        self.basis_mat = torch.nn.ModuleList([torch.nn.Linear(self.app_n_comp[l][0], self.app_dim[l], bias=False).to(device) for l in range(len(self.app_dim))]).to(device)


    def init_angle_svd(self, n_component, local_dim, dimpos, scale, device, lvl):
        angle_coef = []
        for l in range(lvl):
            angle_coef.append(torch.nn.Parameter(scale * torch.randn((1, n_component[l][0], local_dim[l][dimpos]), device=device), requires_grad=True).to(device))
        return torch.nn.ParameterList(angle_coef).to(device)


    def init_one_svd(self, geo, n_component, local_dims, scale, device, lvl):
        line_coef = []
        for l in range(lvl):
            for i in range(3):
                line_coef.append(torch.nn.Parameter(scale * torch.randn((len(geo[l]), n_component[l][0], local_dims[l][i]+1))))
        return torch.nn.ParameterList(line_coef).to(device)


    def get_kwargs(self):
        return {
            'aabb': self.aabb,
            'gridSize':self.gridSize.tolist(),
            'density_n_comp': self.density_n_comp,
            'appearance_n_comp': self.app_n_comp,
            'app_dim': self.app_dim,

            'density_shift': self.density_shift,
            'alphaMask_thres': self.alphaMask_thres,
            'distance_scale': self.distance_scale,
            'rayMarch_weight_thres': self.rayMarch_weight_thres,
            'fea2denseAct': self.fea2denseAct,

            'near_far': self.near_far,
            'step_ratio': self.step_ratio,

            'shadingMode': self.shadingMode,
            'pos_pe': self.pos_pe,
            'view_pe': self.view_pe,
            'fea_pe': self.fea_pe,
            'featureC': self.featureC
        }



    def get_optparam_groups(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001, skip_zero_grad=True):
        grad_vars = [{'params': self.density_line, 'lr': lr_init_spatialxyz, 'skip_zero_grad': (skip_zero_grad)}, {'params': self.app_line, 'lr': lr_init_spatialxyz, 'skip_zero_grad': (skip_zero_grad)}]

        grad_vars += [{'params': self.basis_mat[i].parameters(), 'lr':lr_init_network, 'skip_zero_grad': (False)} for i in range(self.lvl)]
        if len(self.local_dims[0]) > 3:
            grad_vars += [
                {'params': self.theta_line, 'lr': lr_init_spatialxyz, 'skip_zero_grad': (skip_zero_grad)},
                {'params': self.phi_line, 'lr': lr_init_spatialxyz, 'skip_zero_grad': (skip_zero_grad)}
            ]
        if isinstance(self.renderModule, torch.nn.Module):
            grad_vars += [{'params':self.renderModule.parameters(), 'lr':lr_init_network, 'skip_zero_grad': (False)}]
        return grad_vars

    def get_geoparam_groups(self, lr_init_geo=0.03):
        grad_vars = []
        for l in range(self.lvl):
            grad_vars += [
                {'params': self.pnt_rot[l], 'lr': lr_init_geo, 'skip_zero_grad': (False), "weight_decay": 0.0}
            ]
        return grad_vars



    def ind_intrp_line_map_batch(self, vecModes, density_lines, local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, tensoRF_id):

        line_s = torch.stack([density_lines[0][tensoRF_id, :, local_gindx_s[..., 0]], density_lines[1][tensoRF_id, :, local_gindx_s[..., 1]], density_lines[2][tensoRF_id, :, 2]], dim=-1)
        line_l = torch.stack([density_lines[0][tensoRF_id, :, local_gindx_l[..., 0]], density_lines[1][tensoRF_id, :, local_gindx_l[..., 1]], density_lines[2][tensoRF_id, :, local_gindx_l[..., 2]]], dim=-1)

        line_s_gweight = torch.stack([local_gweight_s[:, None, 0], local_gweight_s[:, None, 1], local_gweight_s[:, None, 2]], dim=-1)
        line_l_gweight = torch.stack([local_gweight_l[:, None, 0], local_gweight_l[:, None, 1], local_gweight_l[:, None, 2]], dim=-1)

        return line_s * line_s_gweight + line_l * line_l_gweight


    def ind_intrp_line_map_batch_prod(self, vecModes, density_lines, local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, tensoRF_id):
        return (density_lines[0][tensoRF_id, :, local_gindx_s[..., 0]] * local_gweight_s[:, None, 0] + density_lines[0][tensoRF_id, :, local_gindx_l[..., 0]] * local_gweight_l[:, None, 0]) *  (density_lines[1][tensoRF_id, :, local_gindx_s[..., 1]] * local_gweight_s[:, None, 1] + density_lines[1][tensoRF_id, :, local_gindx_l[..., 1]] * local_gweight_l[:, None, 1]) * (density_lines[2][tensoRF_id, :, local_gindx_s[..., 2]] * local_gweight_s[:, None, 2] + density_lines[2][tensoRF_id, :, local_gindx_l[..., 2]] * local_gweight_l[:, None, 2])


    def ind_intrp_line_batch(self, density_line, local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l):
        b_inds = torch.zeros([len(local_gindx_s)], device=density_line.device, dtype=torch.int64)
        return density_line[b_inds, :, local_gindx_s] * local_gweight_s[:, None] + density_line[b_inds, :, local_gindx_l] * local_gweight_l[:, None]

    @torch.no_grad()
    def up_sampling_Vector(self, density_line_coef, app_line_coef, theta_line_coef, phi_line_coef, res_target):
        for l in range(self.lvl):
            for i in range(len(self.vecMode)):
                density_line_coef[3*l+i] = torch.nn.Parameter(
                    F.interpolate(density_line_coef[3*l+i].data, size=(res_target[l][i]+1), mode='linear', align_corners=True))
                app_line_coef[3*l+i] = torch.nn.Parameter(
                    F.interpolate(app_line_coef[3*l+i].data, size=(res_target[l][i]+1), mode='linear', align_corners=True))
            if len(self.local_dims[0]) > 3:
                print("no implementation")
                exit()

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


    def compute_densityfeature_geo(self, local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, local_kernel_dist, tensoRF_id, agg_id, sample_num=None):
        # plane + line basis
        # sigma_feature =  torch.sum(torch.prod(self.ind_intrp_line_map_batch(self.vecMode, self.density_line, local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, tensoRF_id), dim=-1), dim=1, keepdim=True)
        sigma_feature_acc = torch.zeros([sample_num], device=local_gindx_s[0].device, dtype=torch.float32)
        num_lvl_exist = torch.zeros([sample_num, 1], device="cuda", dtype=torch.float32)
        for l in range(self.lvl):
            if len(local_gindx_s[l]) > 0:
                sigma_feature = torch.sum(self.ind_intrp_line_map_batch_prod(self.vecMode, self.density_line[3*l:3*l+3], local_gindx_s[l], local_gindx_l[l], local_gweight_s[l], local_gweight_l[l], tensoRF_id[l]), dim=1, keepdim=True)
                sigma_feature, has_tensorf = self.agg_tensoRF_at_samples(local_kernel_dist[l], agg_id[l], sigma_feature, out=torch.zeros([sample_num, 1], device=local_gindx_s[l].device, dtype=torch.float32), outweight=torch.zeros([sample_num, 1], device=local_gindx_s[l].device, dtype=torch.float32))
                if self.args.den_lvl_norm > 0:
                    num_lvl_exist += has_tensorf
                sigma_feature_acc += sigma_feature.squeeze(-1)
        return sigma_feature_acc / torch.clamp(num_lvl_exist, min=1).squeeze(-1)


    def compute_appfeature_geo(self, local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, local_kernel_dist, tensoRF_id, agg_id, sample_num=None, dir_gindx_s=None, dir_gindx_l=None, dir_gweight_l=None):
        # plane + line basis
        # line_coef_point = torch.prod(self.ind_intrp_line_map_batch(self.vecMode, self.app_line, local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, tensoRF_id), dim=-1)
        infeat = torch.zeros([sample_num, 0 if self.args.radiance_add == 0 else self.app_dim[0]], device=local_gindx_s[0].device, dtype=torch.float32)
        num_lvl_exist = torch.zeros([sample_num, 1], device="cuda", dtype=torch.float32)
        for l in range(self.lvl):
            if len(local_gindx_s[l]) > 0:
                line_coef_point = self.ind_intrp_line_map_batch_prod(self.vecMode, self.app_line[3*l:3*l+3], local_gindx_s[l], local_gindx_l[l], local_gweight_s[l], local_gweight_l[l], tensoRF_id[l])
                app_feat, has_tensorf = self.agg_tensoRF_at_samples(local_kernel_dist[l], agg_id[l], line_coef_point, out=torch.zeros([sample_num, self.app_n_comp[l][0]], device=local_gindx_s[l].device, dtype=torch.float32), outweight=torch.zeros([sample_num, 1], device=local_gindx_s[l].device, dtype=torch.float32))
                if dir_gindx_s is not None:
                    print("not implemented")
                    exit()
                if self.args.radiance_add == 0:
                    infeat = torch.cat([infeat, self.basis_mat[l](app_feat)], dim=-1)
                else:
                    infeat += self.basis_mat[l](app_feat)
                    if self.args.rad_lvl_norm > 0: 
                        num_lvl_exist += has_tensorf 
            else:
                infeat = torch.cat([infeat, torch.zeros([sample_num, self.app_dim[l]], device=infeat.device, dtype=torch.float32)], dim=-1) if self.args.radiance_add == 0 else infeat
        return infeat / torch.clamp(num_lvl_exist, min=1)


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

    # if cfg_train.weight_rgbper > 0:
    #             rgbper = (render_result['raw_rgb'] - target[render_result['ray_id']]).pow(2).sum(-1)
    #             rgbper_loss = (rgbper * render_result['weights'].detach()).sum() / len(rays_o)
    #             loss += cfg_train.weight_rgbper * rgbper_loss
    #         loss.backward()
    # def

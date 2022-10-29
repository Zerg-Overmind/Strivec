import os, sys, copy, glob, json, time, random, argparse
from shutil import copyfile
from tqdm import tqdm, trange

# import mmcv
import imageio
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import utils, dvgo
# from lib.load_data import load_data

# from torch_efficient_distloss import flatten_eff_distloss
sys.path.append("dataLoader/")
from ray_utils import SimpleSampler


def _compute_bbox_by_cam_frustrm_bounded(cfg, HW, K, poses, near, far):
    xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min
    H, W = HW
    for c2w in poses:
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w,
                ndc=cfg["ndc"], inverse_y=cfg["inverse_y"],
                flip_x=cfg["flip_x"], flip_y=cfg["flip_y"])
        if cfg["ndc"]:
            pts_nf = torch.stack([rays_o+rays_d*near, rays_o+rays_d*far])
        else:
            pts_nf = torch.stack([rays_o+viewdirs*near, rays_o+viewdirs*far])
        xyz_min = torch.minimum(xyz_min, pts_nf.amin((0,1,2)))
        xyz_max = torch.maximum(xyz_max, pts_nf.amax((0,1,2)))
    return xyz_min, xyz_max

def _compute_bbox_by_cam_frustrm_unbounded(cfg, HW, Ks, poses, near_clip):
    # Find a tightest cube that cover all camera centers
    xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min
    for (H, W), K, c2w in zip(HW[i_train], Ks[i_train], poses[i_train]):
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w,
                ndc=cfg["ndc"], inverse_y=cfg["inverse_y"],
                flip_x=cfg["flip_x"], flip_y=cfg["flip_y"])
        pts = rays_o + rays_d * near_clip
        xyz_min = torch.minimum(xyz_min, pts.amin((0,1)))
        xyz_max = torch.maximum(xyz_max, pts.amax((0,1)))
    center = (xyz_min + xyz_max) * 0.5
    radius = (center - xyz_min).max() * cfg["unbounded_inner_r"]
    xyz_min = center - radius
    xyz_max = center + radius
    return xyz_min, xyz_max

def compute_bbox_by_cam_frustrm(args, cfg, HW, Ks, poses, near, far, **kwargs):
    print('compute_bbox_by_cam_frustrm: start')
    if cfg["unbounded_inward"]:
        xyz_min, xyz_max = _compute_bbox_by_cam_frustrm_unbounded(
                cfg, HW, Ks, poses, kwargs.get('near_clip', None))
    else:
        xyz_min, xyz_max = _compute_bbox_by_cam_frustrm_bounded(
                cfg, HW, Ks, poses, near, far)
    print('compute_bbox_by_cam_frustrm: xyz_min', xyz_min)
    print('compute_bbox_by_cam_frustrm: xyz_max', xyz_max)
    print('compute_bbox_by_cam_frustrm: finish')
    return xyz_min, xyz_max

def create_new_model(cfg, args, xyz_min, xyz_max, stage):
    num_voxels = int(args.pre_num_voxels / (2**len(args.pre_pg_scale)))

    if cfg["ndc"]:
        print(f'scene_rep_reconstruction ({stage}): \033[96muse multiplane images\033[0m')
        model = dmpigo.DirectMPIGO(
            xyz_min=xyz_min, xyz_max=xyz_max,
            num_voxels=num_voxels, num_voxels_base=num_voxels, alpha_init=args.pre_alpha_init,
            )
    elif cfg["unbounded_inward"]:
        print(f'scene_rep_reconstruction ({stage}): \033[96muse contraced voxel grid (covering unbounded)\033[0m')
        model = dcvgo.DirectContractedVoxGO(
            xyz_min=xyz_min, xyz_max=xyz_max,
            num_voxels=num_voxels, num_voxels_base=num_voxels, alpha_init=args.pre_alpha_init,
            )
    else:
        print(f'scene_rep_reconstruction ({stage}): \033[96muse dense voxel grid\033[0m')
        model = dvgo.DirectVoxGO(
            xyz_min=xyz_min, xyz_max=xyz_max,
            num_voxels=num_voxels, num_voxels_base=num_voxels,
            mask_cache_path=None, alpha_init=args.pre_alpha_init,
            )
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = utils.create_optimizer_or_freeze_model(model, args, global_step=0)
    return model, optimizer

def scene_rep_reconstruction(args, cfg, xyz_min, xyz_max, dataset, stage, coarse_ckpt_path=None):
    # init
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if abs(args.world_bound_scale - 1) > 1e-9:
        xyz_shift = (xyz_max - xyz_min) * (args.world_bound_scale - 1) / 2
        xyz_min -= xyz_shift
        xyz_max += xyz_shift
    HW = [dataset.img_wh[1], dataset.img_wh[0]]
    Ks = dataset.intrinsics
    poses = dataset.raw_poses
    near, far = dataset.near_far

    # init model and optimizer
    print(f'scene_rep_reconstruction ({stage}): train density from scratch')
    model, optimizer = create_new_model(cfg, args, xyz_min, xyz_max, stage)
    start = 0
    if args.pre_maskout_near_cam_vox:
        model.maskout_near_cam_vox(poses[:,:3,3], near)

    # init rendering setup
    render_kwargs = {
        'near': near,
        'far': far,
        'bg': 1 if cfg["white_bkgd"] else 0,
        'rand_bkgd': cfg["rand_bkgd"],
        'stepsize': args.step_ratio,
        'inverse_y': cfg["inverse_y"],
        'flip_x': cfg["flip_x"],
        'flip_y': cfg["flip_y"],
    }
    # gather rays
    allrays, allrgbs = dataset.all_rays, dataset.all_rgbs
    rays_o_tr, rays_d_tr = allrays[..., :3].to(device), allrays[..., 3:].to(device)
    # print("rays_d_tr", rays_d_tr[0], poses[0], dataset.poses[0])
    # exit()
    trainingSampler = SimpleSampler(allrays.shape[0], args.pre_batch_size)
    imsz = [HW[0] * HW[1] for i in range(len(poses))]
    
    # view-count-based learning rate
    # if args.pervoxel_lr:
    #     def per_voxel_init():
    #         cnt = model.voxel_count_views(
    #             rays_o_tr=rays_o_tr, rays_d_tr=rays_d_tr, imsz=imsz,
    #             near=near, far=far, stepsize=args.step_ratio, downrate=1,
    #             irregular_shape=dataset.irregular_shape)
    #         optimizer.set_pervoxel_lr(cnt)
    #         model.mask_cache.mask[cnt.squeeze() <= 2] = False
    #     per_voxel_init()

    # if args.pre_maskout_lt_nviews > 0:
    #     model.update_occupancy_cache_lt_nviews(
    #             rays_o_tr, rays_d_tr, imsz, render_kwargs, args.pre_maskout_lt_nviews)
    # GOGO
    torch.cuda.empty_cache()
    psnr_lst = []
    time0 = time.time()
    global_step = -1
    for global_step in trange(1+start, 1+args.pre_N_iters):
        # renew occupancy grid
        # if model.mask_cache is not None and (global_step + 500) % 1000 == 0:
        #     model.update_occupancy_cache()

        # progress scaling checkpoint
        if global_step in args.pre_pg_scale:
            n_rest_scales = len(args.pre_pg_scale)-args.pre_pg_scale.index(global_step)-1
            cur_voxels = int(args.pre_num_voxels / (2**n_rest_scales))
            if isinstance(model, (dvgo.DirectVoxGO, dcvgo.DirectContractedVoxGO)):
                model.scale_volume_grid(cur_voxels)
            elif isinstance(model, dmpigo.DirectMPIGO):
                model.scale_volume_grid(cur_voxels, model.mpi_depth)
            else:
                raise NotImplementedError
            optimizer = utils.create_optimizer_or_freeze_model(model, args, global_step=0)
            model.act_shift -= args.decay_after_scale
            torch.cuda.empty_cache()

        # random sample rays
        # if cfg_train.ray_sampler in ['flatten', 'in_maskcache']:
        #     sel_i = batch_index_sampler()
        #     target = rgb_tr[sel_i]
        #     rays_o = rays_o_tr[sel_i]
        #     rays_d = rays_d_tr[sel_i]
        #     viewdirs = viewdirs_tr[sel_i]
        # elif cfg_train.ray_sampler == 'random':
        #     sel_b = torch.randint(rgb_tr.shape[0], [cfg_train.N_rand])
        #     sel_r = torch.randint(rgb_tr.shape[1], [cfg_train.N_rand])
        #     sel_c = torch.randint(rgb_tr.shape[2], [cfg_train.N_rand])
        #     target = rgb_tr[sel_b, sel_r, sel_c]
        #     rays_o = rays_o_tr[sel_b, sel_r, sel_c]
        #     rays_d = rays_d_tr[sel_b, sel_r, sel_c]
        #     viewdirs = viewdirs_tr[sel_b, sel_r, sel_c]
        # else:
        #     raise NotImplementedError

        # sample ray index
        ray_idx = trainingSampler.nextids()
        rays_o, rays_d, target = rays_o_tr[ray_idx].to(device), rays_d_tr[ray_idx].to(
            device), allrgbs[ray_idx].to(device)

        # volume rendering
        render_result = model(
            rays_o, rays_d, rays_d,
            global_step=global_step, is_train=True,
            **render_kwargs)

        # gradient descent step
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(render_result['rgb_marched'], target)
        psnr = utils.mse2psnr(loss.detach())

        if args.pre_weight_entropy_last > 0:
            pout = render_result['alphainv_last'].clamp(1e-6, 1-1e-6)
            entropy_last_loss = -(pout*torch.log(pout) + (1-pout)*torch.log(1-pout)).mean()
            loss += args.pre_weight_entropy_last * entropy_last_loss

        if args.pre_weight_rgbper > 0:
            rgbper = (render_result['raw_rgb'] - target[render_result['ray_id']]).pow(2).sum(-1)
            rgbper_loss = (rgbper * render_result['weights'].detach()).sum() / len(rays_o)
            loss += args.pre_weight_rgbper * rgbper_loss
        loss.backward()

        optimizer.step()
        psnr_lst.append(psnr.item())

        # update lr
        decay_steps = args.pre_lrate_decay * 1000
        decay_factor = 0.1 ** (1/decay_steps)
        for i_opt_g, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = param_group['lr'] * decay_factor

        # check log & save
        if global_step % 500==0:
            eps_time = time.time() - time0
            eps_time_str = f'{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}'
            tqdm.write(f'scene_rep_reconstruction ({stage}): iter {global_step:6d} / '
                       f'Loss: {loss.item():.9f} / PSNR: {np.mean(psnr_lst):5.2f} / '
                       f'Eps: {eps_time_str}')
            psnr_lst = []
    return model

def get_density_pnts(args, train_dataset, bounded=True):
    # args, cfg, HW, Ks, poses, i_train, near, far
    HW = [train_dataset.img_wh[1], train_dataset.img_wh[0]]
    Ks = train_dataset.intrinsics
    poses = train_dataset.raw_poses
    near, far = train_dataset.near_far
    cfg = {
        "unbounded_inward" : train_dataset.unbounded_inward,
        "unbounded_inner_r" : train_dataset.unbounded_inner_r,
        "white_bkgd" : train_dataset.white_bg,
        "rand_bkgd" : False,
        "flip_y" : train_dataset.flip_y,
        "flip_x" : train_dataset.flip_x,
        "inverse_y" : train_dataset.inverse_y,
        "ndc" : train_dataset.ndc,
    }
    xyz_min_coarse, xyz_max_coarse = compute_bbox_by_cam_frustrm(args, cfg, HW, Ks, poses, near, far)
    
    model = scene_rep_reconstruction(
        args=args, cfg=cfg, xyz_min=xyz_min_coarse, xyz_max=xyz_max_coarse,
        dataset=train_dataset, stage='coarse')
    # coarse_ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'coarse_last.tar')
    interp = torch.stack(torch.meshgrid(
        torch.linspace(0, 1, model.world_size[0], device="cuda"),
        torch.linspace(0, 1, model.world_size[1], device="cuda"),
        torch.linspace(0, 1, model.world_size[2], device="cuda"),
    ), -1)
    dense_xyz = model.xyz_min * (1-interp) + model.xyz_max * interp
    density = model.density(dense_xyz)
    alpha = model.activate_density(density)
    thres = 1e-4
    mask = (alpha > thres)
    geo = dense_xyz[mask]
    # print("active_xyz", active_xyz.shape)
    return geo
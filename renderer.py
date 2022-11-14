import torch,os,imageio,sys
from tqdm.auto import tqdm
from dataLoader.ray_utils import get_rays
from models.tensoRF import TensorVM, TensorCP, raw2alpha, TensorVMSplit, AlphaGridMask
from models.pointTensoRF import PointTensorCP
from models.pointTensoRF_hier import PointTensorCP_hier
from models.pointTensoRF_adapt import PointTensorCP_adapt
from models.archive_pointTensoRF import PointTensorCPB, PointTensorCPD, PointTensorVMSplit
from utils import *
from dataLoader.ray_utils import ndc_rays_blender
import random
 
 
def OctreeRender_trilinear_fast(rays, tensorf, chunk=4096, N_samples=-1, ray_type=0, white_bg=True, is_train=False,
    device='cuda', return_depth=0, tensoRF_per_ray=None, eval=False, rot_step=False, depth_bg=True):

    rgbs, alphas, depth_maps, weights, uncertainties, rgbpers, ray_ids = [], [], [], [], [], [], []
    N_rays_all = rays.shape[0]
    
    for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):

        rays_chunk = rays[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)
        tensoRF_per_ray_chunk = tensoRF_per_ray[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device) if tensoRF_per_ray is not None else None
        # print("tensoRF_per_ray shape ", rays.shape, rays_chunk.shape, tensoRF_per_ray.shape, tensoRF_per_ray_chunk.shape)
        rgb_map, depth_map, rgbper, ray_id, weight = tensorf(rays_chunk, is_train=is_train, white_bg=white_bg,
                                                             ray_type=ray_type, N_samples=N_samples,
                                                             return_depth=return_depth,
                                                              tensoRF_per_ray=tensoRF_per_ray_chunk, eval=eval, rot_step=rot_step, depth_bg=depth_bg)
        
        rgbs.append(rgb_map)
        depth_maps.append(depth_map)
        if rgbper is not None:
            rgbpers.append(rgbper)
            ray_ids.append(ray_id)
            weights.append(weight) 

    return torch.cat(rgbs), torch.cat(weights) if len(weights) > 0 else None, torch.cat(depth_maps) if return_depth else None, torch.cat(rgbpers) if len(rgbpers) > 0 else None, torch.cat(ray_ids) if len(ray_ids) > 0 else None


# def den_eval(geo, dataset, allrays, allrgbs, tensorf, args, renderer, N_samples=-1, white_bg=True, ray_type=0,
#              device="cuda", den_thresh=0.7):
#     xyz_input, alphas = tensorf.get_grid_centers(), []
#
#     for chunk_idx in range(len(xyz_input) // chunk + int(len(xyz_input) % chunk > 0)):
#         alphas = tensorf.probe_density(xyz_input)
#
#     return torch.cat(rgbs), torch.cat(weights), torch.cat(depth_maps) if return_depth else None, torch.cat(
#         rgbpers) if len(rgbpers) > 0 else None, torch.cat(ray_ids) if len(ray_ids) > 0 else None


@torch.no_grad()
def evaluation(test_dataset, tensorf, args, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
               white_bg=False, ray_type=0, compute_extra_metrics=True, device='cuda'):
    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims,l_alex,l_vgg=[],[],[]
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath+"/rgbd", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    if len(test_dataset.img_wh) > 2:
       img_eval_interval = 1 if N_vis < 0 else max(len(test_dataset.all_rays) // N_vis,1)
       idxs = list(range(0, len(test_dataset.all_rays), img_eval_interval))
       img_eval_itv = 1 if N_vis < 0 else max(len(test_dataset.img_wh) // N_vis,1)
       idxs_img_lst = list(range(0, len(test_dataset.img_wh), img_eval_itv))
       idxs_img_num = [test_dataset.img_wh[:idxs_img_lst[bkb]] for bkb in range(len(idxs_img_lst))]
       idxs_img = []
    else:
       img_eval_interval = 1 if N_vis < 0 else max(test_dataset.all_rays.shape[0] // N_vis,1)
       idxs = list(range(0, test_dataset.all_rays.shape[0], img_eval_interval))

    if len(test_dataset.img_wh) > 2:
       for idx_k in range(len(idxs_img_num)):
           if idx_k == 0:
              idxs_img.append(0)
           else:
              idxs_img.append((np.array(idxs_img_num[idx_k])[:,0]*np.array(idxs_img_num[idx_k])[:,1]).sum())
    
    img_kk = 0
    for idx, samples in tqdm(enumerate(test_dataset.all_rays[0::img_eval_interval]), file=sys.stdout):
     
        if len(test_dataset.img_wh) > 2:

           W, H = test_dataset.img_wh[idxs[idx]]
        else:
            W, H = test_dataset.img_wh
           
         
        W, H = W - 2 * args.test_margin, H - 2 * args.test_margin

        rays = samples.view(-1, samples.shape[-1])
            
        if len(test_dataset.img_wh) > 2:
            gt_rgb = test_dataset.all_rgbs_stack[idxs_img_lst[img_kk]]
        else:
            gt_rgb = test_dataset.all_rgbs[idxs[idx]].view(H, W, 3)
            
        
        gt_rgb = margin_restore(gt_rgb, args.test_margin)
        img_kk += 1
        cur_tensoRF_per_ray = None
        # _, _, cur_tensoRF_per_ray = tensorf.filtering_rays(rays, None, bbox_only=True, apply_filter=False)
    
        rgb_map, _, depth_map, _, _ = renderer(rays, tensorf, chunk=args.batch_size, N_samples=N_samples, ray_type=ray_type, white_bg = white_bg, device=device, return_depth=1, tensoRF_per_ray = None if cur_tensoRF_per_ray is None else cur_tensoRF_per_ray.cuda() , eval=True)
        rgb_map = rgb_map.clamp(0.0, 1.0)

        rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()
        rgb_map = margin_restore(rgb_map, args.test_margin)
        depth_map = margin_restore(depth_map, args.test_margin)
        depth_map, _ = visualize_depth_numpy(depth_map.numpy(),near_far)
        if len(test_dataset.all_rgbs):
            loss = torch.mean((rgb_map - gt_rgb) ** 2)
            PSNRs.append(-10.0 * np.log(loss.item()) / np.log(10.0))

            if compute_extra_metrics:
                ssim = rgb_ssim(rgb_map, gt_rgb, 1)
                l_a = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'alex', tensorf.device)
                l_v = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'vgg', tensorf.device)
                ssims.append(ssim)
                l_alex.append(l_a)
                l_vgg.append(l_v)

        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        # rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', rgb_map)
            rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            imageio.imwrite(f'{savePath}/rgbd/{prtx}{idx:03d}.png', rgb_map)

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr, ssim, l_a, l_v]))
            print(f'{savePath}/{prtx}mean.txt', " psnr:{}, ssim:{}".format(psnr, ssim))
        else:
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr]))
            print(f'{savePath}/{prtx}mean.txt', " psnr:{}".format(psnr))

    if img_eval_interval == 1:
        imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(rgb_maps), fps=30, quality=10)
        imageio.mimwrite(f'{savePath}/{prtx}depthvideo.mp4', np.stack(depth_maps), fps=30, quality=10)

    return PSNRs

@torch.no_grad()
def ray_evaluation(dataset, all_rays, all_rgbs, tensorf, args, renderer, N_samples=-1, white_bg=False, ray_type=0,
                   device='cuda', worse_thresh=0.7):
    PSNRs, depth_xyz = [], []
    # try:
    #     tqdm._instances.clear()
    # except Exception:
    #     pass

    near_far = dataset.near_far
    W, H = dataset.img_wh
    W, H = W - 2 * args.test_margin, H - 2 * args.test_margin
    idxs = list(range(0, len(all_rays)))
    random.shuffle(idxs)
    idxs = np.asarray(idxs)
    chunk = args.batch_size
    N_rays_all = all_rays.shape[0]

    for chunk_idx in tqdm(range(N_rays_all // chunk + int(N_rays_all % chunk > 0)), file=sys.stdout):
        batch_inds = idxs[chunk_idx * chunk:(chunk_idx + 1) * chunk]
        rays_chunk = all_rays[batch_inds].to(device)
        rgb_chunk = all_rgbs[batch_inds].to(device)
        rgb_map, _, depth_map, _, _ = renderer(rays_chunk, tensorf, chunk=args.batch_size, N_samples=N_samples,
                                               ray_type=ray_type, white_bg=white_bg, device=device, return_depth=1,
                                               tensoRF_per_ray=None, eval=True, depth_bg=False)
        rgb_map = rgb_map.clamp(0.0, 1.0)
        # rgbs.append(rgb_map)
        depth_xyz_cur = rays_chunk[..., :3] + rays_chunk[..., 3:] * depth_map[..., None]
        depth_xyz.append(depth_xyz_cur)

        loss = torch.mean(((rgb_map - rgb_chunk) ** 2).reshape(-1, 3), dim=-1)
        PSNRs.append(-10.0 * torch.log(loss) / np.log(10.0))
    PSNRs = torch.cat(PSNRs)
    depth_xyz = torch.cat(depth_xyz, dim=0)
    max_PSNRs = torch.max(PSNRs)
    thresh_mask = PSNRs < max_PSNRs * worse_thresh
    PSNRs = PSNRs[thresh_mask]
    np.savetxt("log/ship_adapt_full_0.4_0.2_0.1_2222/depth_xyz_all.txt", depth_xyz.cpu().numpy(), delimiter=";")
    depth_xyz = depth_xyz[thresh_mask, :]
    print("PSNRs, depth_xyz", PSNRs.shape, depth_xyz.shape)
    np.savetxt("log/ship_adapt_full_0.4_0.2_0.1_2222/depth_xyz_threshed.txt", depth_xyz.cpu().numpy(), delimiter=";")
    return PSNRs, depth_xyz

def margin_restore(map, margin):
    if margin != 0:
        if map.dim() >= 3:
            new_map = torch.zeros([map.shape[0] + 2 * margin, map.shape[1] + 2 * margin, map.shape[2]], device=map.device, dtype=map.dtype)
            new_map[margin:-margin, margin:-margin,:] = map
        else:
            new_map = torch.zeros([map.shape[0] + 2 * margin, map.shape[1] + 2 * margin],
                                  device=map.device, dtype=map.dtype)
            new_map[margin:-margin, margin:-margin] = map
        map = new_map
    return map


@torch.no_grad()
def evaluation_path(test_dataset,tensorf, c2ws, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
                    white_bg=False, ray_type=0, compute_extra_metrics=True, device='cuda'):
    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims,l_alex,l_vgg=[],[],[]
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath+"/rgbd", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    for idx, c2w in tqdm(enumerate(c2ws)):

        W, H = test_dataset.img_wh

        c2w = torch.FloatTensor(c2w)
        rays_o, rays_d = get_rays(test_dataset.directions, c2w)  # both (h*w, 3)
        if ray_type == 1:
            rays_o, rays_d = ndc_rays_blender(H, W, test_dataset.focal[0], 1.0, rays_o, rays_d)
        rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)

        rgb_map, _, depth_map, _, _ = renderer(rays, tensorf, chunk=8192, N_samples=N_samples,
                                        ray_type=ray_type, white_bg = white_bg, device=device)
        rgb_map = rgb_map.clamp(0.0, 1.0)

        rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()

        depth_map, _ = visualize_depth_numpy(depth_map.numpy(),near_far)

        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        # rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', rgb_map)
            rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            imageio.imwrite(f'{savePath}/rgbd/{prtx}{idx:03d}.png', rgb_map)

    imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(rgb_maps), fps=30, quality=8)
    imageio.mimwrite(f'{savePath}/{prtx}depthvideo.mp4', np.stack(depth_maps), fps=30, quality=8)

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr, ssim, l_a, l_v]))
        else:
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr]))


    return PSNRs


import sys
import os
import pathlib
sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), '..'))
sys.path.append(os.path.join(pathlib.Path(__file__).parent.parent.absolute(), '..'))
import torch
import torch_cluster
import numpy as np
from mvs import mvs_utils, filter_utils
torch.manual_seed(0)
np.random.seed(0)
from tqdm import tqdm
from utils import masking, create_mvs_model
from dataLoader import mvs_dataset_dict
from preprocessing.cluster import cluster
from collections import defaultdict
from preprocessing.boxing import find_tensorf_box, filter_cluster_n_pnts

def load(pointfile):
    if os.path.exists(pointfile):
        return torch.as_tensor(np.loadtxt(pointfile, delimiter=";"), dtype=torch.float32)
    else:
        return None


def gen_pnts(args):
    if args.pointfile.endswith('ply'):
        # points_path = os.path.join(self.root_dir, "exported/pcd.ply")
        geo = mvs_utils.load_ply_points(args)
    elif args.pointfile == 'depth':
        colordir = os.path.join(args.datadir, "exported/color")
        image_paths = [f for f in os.listdir(colordir) if os.path.isfile(os.path.join(colordir, f))]
        image_paths = [os.path.join(args.datadir, "exported/color/{}.jpg".format(i)) for i in
                            range(len(image_paths))]
        all_id_list = mvs_utils.filter_valid_id(args, list(range(len(image_paths))))
        depth_intrinsic = np.loadtxt(
            os.path.join(args.datadir, "exported/intrinsic/intrinsic_depth.txt")).astype(np.float32)[:3, :3]
        geo = mvs_utils.load_init_depth_points(args, all_id_list, depth_intrinsic, device="cuda")
        # np.savetxt(os.path.join(args.basedir, args.expname, "depth.txt"), geo.cpu().numpy(), delimiter=";")
    else:
        geo = load(args.pointfile)
    return geo

def gen_geo(args, pnts=None):

    if pnts is None:
        print("Do MVS to create pointfile at ", args.pointfile)
        dataset = mvs_dataset_dict[args.dataset_name]
        mvs_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=False)
        model = create_mvs_model(args)
        xyz_world_all, confidence_filtered_all, rgb_all = gen_points_filter(mvs_dataset, args, model)
        # geo = torch.cat([xyz_world_all, confidence_filtered_all], dim=-1)
        pnts = torch.cat([xyz_world_all, torch.as_tensor(rgb_all * 255, device=xyz_world_all.device)], dim=-1)
        os.makedirs(os.path.dirname(args.pointfile), exist_ok=True)
        np.savetxt(args.pointfile, pnts.cpu().numpy(), delimiter=";")
    else:
        print("successfully loaded args.pointfile at : ", args.pointfile, pnts.shape)
        if args.ranges[0] > -90.0:
            spacemin = torch.as_tensor(args.ranges[:3], device=pnts.device)
            spacemax = torch.as_tensor(args.ranges[3:], device=pnts.device)
            mask = (pnts[...,:3] - spacemin[None, ...].to(pnts.device)) >= 0
            mask *= (spacemax[None, ...].to(pnts.device) - pnts[...,:3]) >= 0
            mask = torch.prod(mask, dim=-1) > 0
            pnts = pnts[mask]
    lvl = len(args.vox_range) if args.vox_range is not None else len(args.cluster_method)
    cluster_dict = {
        "cluster_xyz": [[] for l in range(lvl)],
        "cluster_pnts": [[] for l in range(lvl)],
        "box_length": [[] for l in range(lvl)],
        "pca_axis": [[] for l in range(lvl)],
        "stds": [[] for l in range(lvl)],
    }

    for l in range(lvl):
        if args.vox_res > 0:
            _, _, sampled_pnt_idx = mvs_utils.construct_vox_points_closest(
                pnts[..., :3] if len(pnts) < 99999999 else pnts[::(len(pnts) // 99999999 + 1), ...].cuda(), args.vox_res)
            pnts = pnts[sampled_pnt_idx, :]

        cluster_dim = pnts.shape[-1] if pnts.shape[-1] == 6 else 3
        lvl_pnts = pnts[..., :cluster_dim]
        if cluster_dim == 6:
            lvl_pnts[..., 3:] = lvl_pnts[..., 3:] / torch.max(lvl_pnts[..., 3:] + 1, dim=-1, keepdims=True)[0] / 100

        count=0
        count_max = 10
        if args.vox_range is not None: # 1
            cluster_xyz, lvl_pnts, sparse_grid_idx, cluster_inds = mvs_utils.construct_voxrange_points_mean(lvl_pnts, torch.as_tensor(args.vox_range[l], dtype=torch.float32, device=pnts.device), vox_center=args.vox_center[l]>0)
            cluster_xyz, lvl_pnts, sparse_grid_idx, cluster_inds = cluster_xyz.cpu().numpy(), lvl_pnts.cpu().numpy(), sparse_grid_idx.cpu().numpy(), cluster_inds.cpu().numpy()
            # np.savetxt(args.pointfile[:-4] + "_{}_{}_vox".format(args.datadir.split("/")[-1], args.vox_range[l][0]) + ".txt", geo_lvl.cpu().numpy(), delimiter=";")
            cluster_pnts, pca_cluster_newpnts, cluster_xyz, box_length, pca_axis, stds = find_tensorf_box(cluster_xyz, lvl_pnts, cluster_inds, None, "obb")
            cluster_xyz, cluster_pnts, box_length, pca_axis, stds, lvl_pnts = filter_cluster_n_pnts(cluster_xyz, cluster_pnts, pca_cluster_newpnts, box_length, pca_axis, stds, None, args.dilation_ratio[l], args, filter_thresh= 600, count=count) # 15000
            cluster_dict["cluster_xyz"][l].append(cluster_xyz[...,:3])
            cluster_dict["box_length"][l].append(box_length)
            cluster_dict["pca_axis"][l].append(pca_axis)
            cluster_dict["stds"][l].append(stds)
            cluster_dict["cluster_pnts"][l] += cluster_pnts
            count += 1
            # grid_idx_lst.append(sparse_grid_idx.cuda())
            # inv_idx_lst.append(inv_idx.cuda())
        lvl_pnts = lvl_pnts.cpu().numpy() if torch.is_tensor(lvl_pnts) else lvl_pnts
        if args.cluster_method is not None:
            for l in range(len(args.cluster_method)):
                while lvl_pnts is not None and count <= count_max and len(lvl_pnts) > 1:
                    cluster_xyz, cluster_inds, cluster_model = cluster(lvl_pnts, method=args.cluster_method[l], num=np.minimum(args.cluster_num[l]//min(count,2), len(lvl_pnts) // 2), vis=False, tol=0.000001 if count > 1 else 0.000001)
                    # cluster_xyz, cluster_inds, cluster_model = cluster(lvl_pnts, method=args.cluster_method[l], num=args.cluster_num[l], vis=False, tol=0.000001 if count > 1 else 0.000001)
                    cluster_pnts, pca_cluster_newpnts, cluster_xyz, box_length, pca_axis, stds = find_tensorf_box(cluster_xyz, lvl_pnts, cluster_inds, cluster_model, args.boxing_method[l])
                    cluster_xyz, cluster_pnts, box_length, pca_axis, stds, lvl_pnts = filter_cluster_n_pnts(cluster_xyz, cluster_pnts, pca_cluster_newpnts, box_length, pca_axis, stds, cluster_model, args.dilation_ratio[l], args, filter_thresh=500 if count < count_max else 50, count=count) # 10000 if count < count_max else 5000
                    count+=1
                    if len(cluster_xyz) == 0:
                        continue
                    cluster_dict["cluster_xyz"][l].append(cluster_xyz)
                    cluster_dict["box_length"][l].append(box_length)
                    cluster_dict["pca_axis"][l].append(pca_axis)
                    cluster_dict["stds"][l].append(stds)
                    cluster_dict["cluster_pnts"][l] += cluster_pnts
                cluster_dict["cluster_xyz"][l] = np.concatenate(cluster_dict["cluster_xyz"][l])
                cluster_dict["box_length"][l] = np.concatenate(cluster_dict["box_length"][l])
                cluster_dict["pca_axis"][l] = np.concatenate(cluster_dict["pca_axis"][l])
                cluster_dict["stds"][l] = np.concatenate(cluster_dict["stds"][l])

            
    # if args.fps_num is not None: # fps_num=[0]
    #     for i in range(len(args.fps_num)):
    #         if len(geo_lst[i]) > args.fps_num[i]:
    #             fps_inds = torch_cluster.fps(geo_lst[i][...,:3], ratio=args.fps_num[i]/len(geo_lst[i]), random_start=True)
    #             geo_lvl = geo_lst[i][fps_inds, ...]
    #             print("fps_inds", fps_inds.shape, geo_lvl.shape)
    #             np.savetxt(args.pointfile[:-4]+"_{}".format(args.fps_num)+".txt", geo_lvl.cpu().numpy(), delimiter=";")
    #             geo_lst[i] = geo_lvl.cuda()
                
    return cluster_dict, pnts[...,:3]



def gen_points_filter(dataset, args, model):
    cam_xyz_all = []
    intrinsics_all = []
    extrinsics_all = []
    confidence_all = []
    points_mask_all = []
    near_fars_all = []
    gpu_filter = True
    cpu2gpu= len(dataset.view_id_list) > 300
    rgb_all, HDWD_lst, c2ws_lst, w2cs_lst, intrinsics_lst = [],[],[],[],[]

    with torch.no_grad():
        for i in tqdm(range(0, len(dataset.view_id_list))):
            data = dataset.get_init_item(i)
            # intrinsics    1, 3, 3, 3
            points_xyz_lst, xyz_color_lst, photometric_confidence_lst, point_mask_lst, HDWD, data_mvs, intrinsics_lst, extrinsics_lst = model.gen_points(data)

            c2ws, w2cs, intrinsics, near_fars = data_mvs['c2ws'], data_mvs['w2cs'], data["intrinsics"], data["near_fars"]

            B, N, C, H, W, _ = points_xyz_lst[0].shape
            # print("points_xyz_lst",points_xyz_lst[0].shape)

            cam_xyz_all.append((points_xyz_lst[0].cpu() if cpu2gpu else points_xyz_lst[0]) if gpu_filter else points_xyz_lst[0].cpu().numpy())

            # intrinsics_lst[0] 1, 3, 3
            intrinsics_all.append(intrinsics_lst[0] if gpu_filter else intrinsics_lst[0])
            extrinsics_all.append(extrinsics_lst[0] if gpu_filter else extrinsics_lst[0].cpu().numpy())
            confidence_all.append((photometric_confidence_lst[0].cpu() if cpu2gpu else photometric_confidence_lst[0]) if gpu_filter else photometric_confidence_lst[0].cpu().numpy())
            points_mask_all.append((point_mask_lst[0].cpu() if cpu2gpu else point_mask_lst[0]) if gpu_filter else point_mask_lst[0].cpu().numpy())
            rgb_all.append(xyz_color_lst[0])
            HDWD_lst.append(HDWD)
            c2ws_lst.append(c2ws)
            w2cs_lst.append(w2cs)
            near_fars_all.append(near_fars[0,0])
            # visualizer.save_neural_points(i, points_xyz_lst[0], None, data, save_ref=args.load_points == 0)
            # #################### start query embedding ##################
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        if gpu_filter:
            _, xyz_world_all, confidence_filtered_all, final_mask_lst = filter_utils.filter_by_masks_gpu(cam_xyz_all, intrinsics_all, extrinsics_all, confidence_all, points_mask_all, args, vis=True, return_w=True, cpu2gpu=cpu2gpu, near_fars_all=near_fars_all)
            rgb_all = [rgb_all[i][:, final_mask_lst[i]].permute(1, 0).cpu().numpy() for i in range(len(rgb_all))]
        else:
            _, xyz_world_all, confidence_filtered_all, final_mask_lst = filter_utils.filter_by_masks(cam_xyz_all, [intr.cpu().numpy() for intr in intrinsics_all], extrinsics_all, confidence_all, points_mask_all, args)
            rgb_all = [np.transpose(rgb_all[i][:, final_mask_lst[i]]) for i in range(len(rgb_all))]
        # print("imgs_all", len(imgs_all), imgs_all[0].shape, len(xyz_world_all), xyz_world_all[0].shape)
        points_vid = torch.cat([torch.ones_like(xyz_world_all[i][...,0:1]) * i for i in range(len(xyz_world_all))], dim=0)
        xyz_world_all = torch.cat(xyz_world_all, dim=0) if gpu_filter else torch.as_tensor(
            np.concatenate(xyz_world_all, axis=0), device="cuda", dtype=torch.float32)
        rgb_all = np.concatenate(rgb_all, axis=0)
        confidence_filtered_all = torch.cat(confidence_filtered_all, dim=0) if gpu_filter else torch.as_tensor(np.concatenate(confidence_filtered_all, axis=0), device="cuda", dtype=torch.float32)
        # print("xyz_world_all", xyz_world_all.shape, points_vid.shape, confidence_filtered_all.shape, rgb_all.shape)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        print("%%%%%%%%%%%%%  getattr(dataset, spacemin, None)", getattr(dataset, "spacemin", None))
        if getattr(dataset, "spacemin", None) is not None or args.ranges[0] > -90.0:
            spacemin = dataset.spacemin if getattr(dataset, "spacemin", None) is not None else torch.as_tensor(args.ranges[:3], device=xyz_world_all.device)
            spacemax = dataset.spacemax if getattr(dataset, "spacemax", None) is not None else torch.as_tensor(args.ranges[3:], device=xyz_world_all.device)
            mask = (xyz_world_all - spacemin[None, ...].to(xyz_world_all.device)) >= 0
            mask *= (spacemax[None, ...].to(xyz_world_all.device) - xyz_world_all) >= 0
            mask = torch.prod(mask, dim=-1) > 0
            first_lst, second_lst = masking(mask, [xyz_world_all, points_vid, confidence_filtered_all], [])
            xyz_world_all, points_vid, confidence_filtered_all = first_lst
            rgb_all=  rgb_all[mask.cpu().numpy()]
        # visualizer.save_neural_points(50, xyz_world_all, None, None, save_ref=False)
        # print("vis 50")
        if getattr(dataset, "alphas", None) is not None:
            vishull_mask = mvs_utils.alpha_masking(xyz_world_all, dataset.alphas, dataset.intrinsics, dataset.cam2worlds, dataset.world2cams, dataset.near_far if args.ranges[0] < -90.0 and getattr(dataset,"spacemin",None) is None else None, args=args)
            first_lst, second_lst = masking(vishull_mask, [xyz_world_all, points_vid, confidence_filtered_all], [])
            xyz_world_all, points_vid, confidence_filtered_all = first_lst
            rgb_all=  rgb_all[vishull_mask.cpu().numpy()]
            print("alpha masking xyz_world_all", xyz_world_all.shape, points_vid.shape)
        # visualizer.save_neural_points(100, xyz_world_all, None, data, save_ref=args.load_points == 0)
        # print("vis 100")
        if args.vox_res > 0:
            xyz_world_all, _, sampled_pnt_idx = mvs_utils.construct_vox_points_closest(xyz_world_all.cuda() if len(xyz_world_all) < 99999999 else xyz_world_all[::(len(xyz_world_all)//99999999+1),...].cuda(), args.vox_res)
            points_vid = points_vid[sampled_pnt_idx,:]
            confidence_filtered_all = confidence_filtered_all[sampled_pnt_idx]
            rgb_all = rgb_all[sampled_pnt_idx.cpu().numpy()]
            print("after voxelize:", xyz_world_all.shape, points_vid.shape)
            xyz_world_all = xyz_world_all.cuda()

    return xyz_world_all, confidence_filtered_all[..., None], rgb_all
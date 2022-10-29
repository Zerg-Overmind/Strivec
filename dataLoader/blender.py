import torch,cv2
from torch.utils.data import Dataset
import json
from tqdm import tqdm
import os
from PIL import Image
from torchvision import transforms as T
from .data_utils import *

from .ray_utils import *


class BlenderDataset(Dataset):
    def __init__(self, datadir, split='train', downsample=1.0, is_stack=False, N_vis=-1, rnd_ray=False, args=None):
        self.rnd_ray = rnd_ray
        self.N_vis = N_vis
        self.root_dir = datadir
        self.split = split
        self.is_stack = is_stack
        self.img_wh = (int(800/downsample),int(800/downsample))
        self.define_transforms()

        self.scene_bbox = torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]]) if args.ranges[0] < -99.00 else torch.tensor([args.ranges[:3], args.ranges[3:]])
        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.read_meta()
        self.define_proj_mat()

        self.white_bg = True
        self.near_far = [2.0,6.0]
        
        self.center = torch.mean(self.scene_bbox, axis=0).float().view(1, 3)
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 3)
        self.downsample=downsample

        self.unbounded_inward = False
        self.unbounded_inner_r = 0.0
        self.flip_y = False
        self.flip_x = False
        self.inverse_y = False
        self.ndc = False
        self.near_clip = None
        self.irregular_shape = False


    def read_depth(self, filename):
        depth = np.array(read_pfm(filename)[0], dtype=np.float32)  # (800, 800)
        return depth
    
    def read_meta(self):

        with open(os.path.join(self.root_dir, f"transforms_{self.split}.json"), 'r') as f:
            self.meta = json.load(f)

        w, h = self.img_wh
        self.focal = 0.5 * 800 / np.tan(0.5 * self.meta['camera_angle_x'])  # original focal length
        self.focal *= self.img_wh[0] / 800  # modify focal length to match size self.img_wh

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.cent = [w / 2, h / 2]
        self.directions, self.ij = get_ray_directions(h, w, [self.focal,self.focal])  # (h, w, 3)
        self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True)
        self.intrinsics = torch.tensor([[self.focal,0,w/2],[0,self.focal,h/2],[0,0,1]]).float()

        self.image_paths = []
        self.poses = []
        self.all_rays = []
        self.all_rgbs = []
        self.all_alpha = []
        self.all_masks = []
        self.all_depth = []
        self.c2ws = []
        self.downsample=1.0
        self.raw_poses = []
        img_eval_interval = 1 if self.N_vis < 0 else len(self.meta['frames']) // self.N_vis
        idxs = list(range(0, len(self.meta['frames']), img_eval_interval))
        for i in tqdm(idxs, desc=f'Loading data {self.split} ({len(idxs)})'):#img_list:#

            frame = self.meta['frames'][i]
            pose = np.array(frame['transform_matrix']) @ self.blender2opencv
            raw_poses = np.array(frame['transform_matrix'])
            c2w = torch.FloatTensor(pose)
            self.poses += [c2w]
            self.raw_poses += [torch.FloatTensor(raw_poses)]

            image_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
            self.image_paths += [image_path]
            img = Image.open(image_path)
            
            if self.downsample!=1.0:
                img = img.resize(self.img_wh, Image.Resampling.LANCZOS)
            img = self.transform(img)  # (4, h, w)
            img = img.view(4, -1).permute(1, 0)  # (h*w, 4) RGBA
            alpha_img = img[:, -1:]
            img = img[:, :3] * alpha_img + (1 - alpha_img)  # blend A to RGB
            if self.rnd_ray > 0:
                img, alpha_img = self.pix_2_patch(img, alpha_img, self.img_wh[1], self.img_wh[0])
                self.all_alpha += [alpha_img]
                self.c2ws += [(c2w[:3, :3].T)[None, ...].repeat(h*w, 1, 1)]
            self.all_rgbs += [img]
            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
            self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)


        self.poses = torch.stack(self.poses)
        self.raw_poses = torch.stack(self.raw_poses)
        if not self.is_stack:
            self.all_rays = torch.cat(self.all_rays, 0)  # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0)  # (len(self.meta['frames])*h*w, 3)
            if self.rnd_ray > 0:
                self.all_alpha = torch.cat(self.all_alpha, 0)
                self.ijs = self.ij.reshape(-1, 2).repeat(len(idxs), 1)
                self.c2ws = torch.cat(self.c2ws, dim=0)
                # print("!!!!!!!!!!!!!!!!!!!!!!!!!self.all_alpha", self.all_alpha.shape)
#             self.all_depth = torch.cat(self.all_depth, 0)  # (len(self.meta['frames])*h*w, 3)
        else:
            self.all_rays = torch.stack(self.all_rays, 0)  # (len(self.meta['frames]),h*w, 3)
            self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3)

            # (len(self.meta['frames]),h,w,3)
            # self.all_masks = torch.stack(self.all_masks, 0).reshape(-1,*self.img_wh[::-1])  # (len(self.meta['frames]),h,w,3)


    def pix_2_patch(self, img, alpha_img, h, w):
        thresh=0.01
        img = img.reshape(h, w, 3)
        hpad = torch.zeros_like(img[0:1, ...])
        img = torch.cat([hpad, img , hpad], dim=0)
        wpad = torch.zeros_like(img[:, 0:1, :])
        img = torch.cat([wpad, img , wpad], dim=1)
        stack_img = torch.stack([img[:-2, :-2, :], img[:-2, 1:-1, :], img[:-2, 2:, :], img[1:-1, :-2, :], img[1:-1, 1:-1, :], img[1:-1, 2:, :], img[2:, :-2, :], img[2:, 1:-1, :], img[2:, 2:, :]], dim=-1).reshape(-1, 3, 3, 3)

        alpha_img = alpha_img.reshape(h, w, 1)
        hpad = torch.zeros_like(alpha_img[0:1, ...])
        alpha_img = torch.cat([hpad, alpha_img, hpad], dim=0)
        wpad = torch.zeros_like(alpha_img[:, 0:1, :])
        alpha_img = torch.cat([wpad, alpha_img, wpad], dim=1)
        stack_alpha_img = torch.cat(
            [alpha_img[:-2, :-2, :], alpha_img[:-2, 1:-1, :], alpha_img[:-2, 2:, :], alpha_img[1:-1, :-2, :], alpha_img[1:-1, 1:-1, :],
             alpha_img[1:-1, 2:, :], alpha_img[2:, :-2, :], alpha_img[2:, 1:-1, :], alpha_img[2:, 2:, :]], dim=-1).reshape(-1, 3, 3)
        topleft = torch.max(stack_alpha_img[..., :2, :2].reshape(-1, 4), dim=-1)[0] - torch.min(stack_alpha_img[..., :2, :2].reshape(-1, 4), dim=-1)[0] > thresh
        bottomleft = torch.max(stack_alpha_img[..., 1:, :2].reshape(-1, 4), dim=-1)[0] - torch.min(stack_alpha_img[..., 1:, :2].reshape(-1, 4), dim=-1)[0] > thresh
        topright = torch.max(stack_alpha_img[..., :2, 1:].reshape(-1, 4), dim=-1)[0] - torch.min(stack_alpha_img[..., :2, 1:].reshape(-1, 4), dim=-1)[0] > thresh
        bottomright = torch.max(stack_alpha_img[..., 1:, 1:].reshape(-1, 4), dim=-1)[0] - torch.min(stack_alpha_img[..., 1:, 1:].reshape(-1, 4), dim=-1)[0] > thresh

        return stack_img, torch.stack([topleft, topright, bottomleft, bottomright], dim=-1).reshape(-1, 2, 2)


    def define_transforms(self):
        self.transform = T.ToTensor()
        
    def define_proj_mat(self):
        self.proj_mat = self.intrinsics.unsqueeze(0) @ torch.inverse(self.poses)[:,:3]

    def world2ndc(self,points,lindisp=None):
        device = points.device
        return (points - self.center.to(device)) / self.radius.to(device)
        
    def __len__(self):
        return len(self.all_rgbs)

    def __getitem__(self, idx):

        if self.split == 'train':  # use data in the buffers
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx]}

        else:  # create data for each image separately

            img = self.all_rgbs[idx]
            rays = self.all_rays[idx]
            mask = self.all_masks[idx] # for quantity evaluation

            sample = {'rays': rays,
                      'rgbs': img,
                      'mask': mask}
        return sample




class BlenderMVSDataset(Dataset):

    def __init__(self, datadir, split='train', downsample=1.0, is_stack=False, N_vis=-1):
        self.data_dir = datadir
        self.split = split

        self.img_wh = (int(800 * downsample), int(800 * downsample))
        self.downsample = downsample

        self.scale_factor = 1.0 / 1.0

        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.height, self.width = int(self.img_wh[1]), int(self.img_wh[0])

        self.white_bg = True

        self.define_transforms()
        with open(os.path.join(self.data_dir, f'transforms_{split}.json'), 'r') as f:
            self.meta = json.load(f)
        self.id_list = [i for i in range(len(self.meta["frames"]))]

        self.proj_mats, self.intrinsics, self.world2cams, self.cam2worlds = self.build_proj_mats()
        self.build_init_metas()
        self.read_meta()
        self.total = len(self.id_list)
        print("dataset total:", self.split, self.total)


    def define_transforms(self):
        self.transform = T.ToTensor()

    def build_init_metas(self):
        self.view_id_list = []
        cam_xyz_lst = [c2w[:3,3] for c2w in self.cam2worlds]

        if self.split=="train":
            cam_xyz = np.stack(cam_xyz_lst, axis=0)
            # triangles = triangluation_bpa(cam_xyz, test_pnts=None, full_comb=True)
            triangles = triangluation_bpa(cam_xyz, test_pnts=None, full_comb=False)
            self.view_id_list = [triangles[i] for i in range(len(triangles))]


    def load_init_points(self):
        if self.args.pcd_dir is not None:
            return self.load_txt_points()
        else:
            return self.load_ply_points()


    def load_txt_points(self):
        points_path = os.path.join(self.args.pcd_dir, self.scan + ".txt")
        assert os.path.exists(points_path)
        points_xyz = torch.as_tensor(np.loadtxt(points_path, delimiter=";"), dtype=torch.float32, device="cuda") # N X 3
        return points_xyz


    def load_ply_points(self):
        points_path = os.path.join(self.data_dir, self.scan, "colmap_results/dense/fused.ply")
        # points_path = os.path.join(self.data_dir, self.scan, "exported/pcd_te_1_vs_0.01_jit.ply")
        assert os.path.exists(points_path)
        plydata = PlyData.read(points_path)
        # plydata (PlyProperty('x', 'double'), PlyProperty('y', 'double'), PlyProperty('z', 'double'), PlyProperty('nx', 'double'), PlyProperty('ny', 'double'), PlyProperty('nz', 'double'), PlyProperty('red', 'uchar'), PlyProperty('green', 'uchar'), PlyProperty('blue', 'uchar'))
        print("plydata", plydata.elements[0])
        x,y,z=torch.as_tensor(plydata.elements[0].data["x"].astype(np.float32), device="cuda", dtype=torch.float32), torch.as_tensor(plydata.elements[0].data["y"].astype(np.float32), device="cuda", dtype=torch.float32), torch.as_tensor(plydata.elements[0].data["z"].astype(np.float32), device="cuda", dtype=torch.float32)
        points_xyz = torch.stack([x,y,z], dim=-1).to(torch.float32)

        # np.savetxt(os.path.join(self.data_dir, self.scan, "exported/pcd.txt"), points_xyz.cpu().numpy(), delimiter=";")
        if self.args.comb_file is not None:
            file_points = np.loadtxt(self.args.comb_file, delimiter=";")
            print("file_points", file_points.shape)
            comb_xyz = torch.as_tensor(file_points[...,:3].astype(np.float32), device=points_xyz.device, dtype=points_xyz.dtype)
            points_xyz = torch.cat([points_xyz, comb_xyz], dim=0)
        # np.savetxt("/home/xharlie/user_space/codes/testNr/checkpoints/pcolallship360_load_confcolordir_KNN8_LRelu_grid320_333_agg2_prl2e3_prune1e4/points/save.txt", points_xyz.cpu().numpy(), delimiter=";")
        return points_xyz



    def build_proj_mats(self, meta=None, list=None):
        proj_mats, intrinsics, world2cams, cam2worlds = [], [], [], []
        list = self.id_list if list is None else list
        meta = self.meta if meta is None else meta
        focal = 0.5 * 800 / np.tan(0.5 * self.meta['camera_angle_x'])  # original focal length
        focal *= self.img_wh[0] / 800  # modify focal length to match size self.img_wh
        self.focal = focal
        self.near_far = np.array([2.0, 6.0])
        for vid in list:
            frame = meta['frames'][vid]
            c2w = np.array(frame['transform_matrix']) @ self.blender2opencv
            w2c = np.linalg.inv(c2w)
            cam2worlds.append(c2w)
            world2cams.append(w2c)

            intrinsic = np.array([[focal, 0, self.width / 2], [0, focal, self.height / 2], [0, 0, 1]])
            intrinsics.append(intrinsic.copy().astype(np.float32))

            # multiply intrinsics and extrinsics to get projection matrix
            proj_mat_l = np.eye(4)
            intrinsic[:2] = intrinsic[:2] / 4
            proj_mat_l[:3, :4] = intrinsic @ w2c[:3, :4]
            proj_mats += [(proj_mat_l, self.near_far)]

        proj_mats, intrinsics = np.stack(proj_mats), np.stack(intrinsics)
        world2cams, cam2worlds = np.stack(world2cams), np.stack(cam2worlds)
        return proj_mats, intrinsics, world2cams, cam2worlds



    def define_transforms(self):
        self.transform = T.ToTensor()


    def read_meta(self):

        w, h = self.img_wh
        self.image_paths = []
        self.poses = []
        self.all_rays = []
        self.blackimgs = []
        self.whiteimgs = []
        self.depths = []
        self.alphas = []

        self.view_id_dict = {}
        self.directions = get_ray_directions(h, w, [self.focal, self.focal])  # (h, w, 3)

        count = 0
        for i, idx in enumerate(self.id_list):
            frame = self.meta['frames'][idx]

            image_path = os.path.join(self.data_dir,  f"{frame['file_path']}.png")
            self.image_paths += [image_path]
            img = Image.open(image_path)
            img = img.resize(self.img_wh, Image.Resampling.LANCZOS)
            img = self.transform(img)  # (4, h, w)
            self.depths += [(img[-1:, ...] > 0.1).numpy().astype(np.float32)]
            self.alphas += [img[-1:].numpy().astype(np.float32)]
            self.blackimgs += [img[:3] * img[-1:]]
            self.whiteimgs += [img[:3] * img[-1:] + (1 - img[-1:])]
            self.view_id_dict[idx] = i
        self.poses = self.cam2worlds


    def __len__(self):
        if self.split == 'train':
            return len(self.id_list) if self.max_len <= 0 else self.max_len
        return len(self.id_list) if self.max_len <= 0 else self.max_len


    def name(self):
        return 'NerfSynthFtDataset'


    def __del__(self):
        print("end loading")

    def get_init_item(self, idx, crop=False):
        sample = {}
        init_view_num = 3
        view_ids = self.view_id_list[idx]
        if self.split == 'train':
            view_ids = view_ids[:init_view_num]

        affine_mat, affine_mat_inv = [], []
        mvs_images, imgs, depths_h, alphas = [], [], [], []
        proj_mats, intrinsics, w2cs, c2ws, near_fars = [], [], [], [], []  # record proj mats between views
        for i in view_ids:
            vid = self.view_id_dict[i]
            mvs_images += [self.blackimgs[vid]]
            imgs += [self.whiteimgs[vid]]
            proj_mat_ls, near_far = self.proj_mats[vid]
            intrinsics.append(self.intrinsics[vid])
            w2cs.append(self.world2cams[vid])
            c2ws.append(self.cam2worlds[vid])

            affine_mat.append(proj_mat_ls)
            affine_mat_inv.append(np.linalg.inv(proj_mat_ls))
            depths_h.append(self.depths[vid])
            alphas.append(self.alphas[vid])
            near_fars.append(near_far)

        for i in range(len(affine_mat)):
            view_proj_mats = []
            ref_proj_inv = affine_mat_inv[i]
            for j in range(len(affine_mat)):
                if i == j:  # reference view
                    view_proj_mats += [np.eye(4)]
                else:
                    view_proj_mats += [affine_mat[j] @ ref_proj_inv]
            # view_proj_mats: 4, 4, 4
            view_proj_mats = np.stack(view_proj_mats)
            proj_mats.append(view_proj_mats[:, :3])
        # (4, 4, 3, 4)
        proj_mats = np.stack(proj_mats)
        imgs = np.stack(imgs).astype(np.float32)
        mvs_images = np.stack(mvs_images).astype(np.float32)
        depths_h = np.stack(depths_h)
        alphas = np.stack(alphas)
        affine_mat, affine_mat_inv = np.stack(affine_mat), np.stack(affine_mat_inv)
        intrinsics, w2cs, c2ws, near_fars = np.stack(intrinsics), np.stack(w2cs), np.stack(c2ws), np.stack(near_fars)
        # view_ids_all = [target_view] + list(src_views) if type(src_views[0]) is not list else [j for sub in src_views for j in sub]
        # c2ws_all = self.cam2worlds[self.remap[view_ids_all]]

        sample['images'] = imgs  # (V, 3, H, W)
        sample['mvs_images'] = mvs_images  # (V, 3, H, W)
        sample['depths_h'] = depths_h.astype(np.float32)  # (V, H, W)
        sample['alphas'] = alphas.astype(np.float32)  # (V, H, W)
        sample['w2cs'] = w2cs.astype(np.float32)  # (V, 4, 4)
        sample['c2ws'] = c2ws.astype(np.float32)  # (V, 4, 4)
        sample['near_fars_depth'] = near_fars.astype(np.float32)[0]
        sample['near_fars'] = np.tile(self.near_far.astype(np.float32)[None,...],(len(near_fars),1))
        sample['proj_mats'] = proj_mats.astype(np.float32)
        sample['intrinsics'] = intrinsics.astype(np.float32)  # (V, 3, 3)
        sample['view_ids'] = np.array(view_ids)
        # sample['light_id'] = np.array(light_idx)
        sample['affine_mat'] = affine_mat
        sample['affine_mat_inv'] = affine_mat_inv
        # sample['scan'] = scan
        # sample['c2ws_all'] = c2ws_all.astype(np.float32)


        for key, value in sample.items():
            if not isinstance(value, str):
                if not torch.is_tensor(value):
                    value = torch.as_tensor(value)
                    sample[key] = value.unsqueeze(0)
                sample[key] = sample[key].cuda()
        return sample

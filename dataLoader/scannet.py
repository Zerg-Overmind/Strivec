import os
import numpy as np
import cv2
import torch
from torchvision import transforms as T
import torchvision.transforms.functional as F
from kornia import create_meshgrid
import time
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
import os
from PIL import Image
import h5py
import mvs.mvs_utils as mvs_utils
import configparser
from .ray_utils import *

from os.path import join
import cv2
# import torch.nn.functional as F
from .data_utils import get_dtu_raydir
from plyfile import PlyData, PlyElement

FLIP_Z = np.asarray([
    [1,0,0],
    [0,1,0],
    [0,0,-1],
], dtype=np.float32)

def colorjitter(img, factor):
    # brightness_factor,contrast_factor,saturation_factor,hue_factor
    # img = F.adjust_brightness(img, factor[0])
    # img = F.adjust_contrast(img, factor[1])
    img = F.adjust_saturation(img, factor[2])
    img = F.adjust_hue(img, factor[3]-1.0)
    return img

def get_rays(directions, c2w):
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate
    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
    """
    # Rotate ray directions from camera coordinate to the world coordinate
    c2w = torch.FloatTensor(c2w)
    rays_d = directions @ c2w[:3, :3].T  # (H, W, 3)
    # rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:3, 3].expand(rays_d.shape)  # (H, W, 3)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d


# def get_ray_directions(H, W, focal, center=None):
#     """
#     Get ray directions for all pixels in camera coordinate.
#     Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
#                ray-tracing-generating-camera-rays/standard-coordinate-systems
#     Inputs:
#         H, W, focal: image height, width and focal length
#     Outputs:
#         directions: (H, W, 3), the direction of the rays in camera coordinate
#     """
#     grid = create_meshgrid(H, W, normalized_coordinates=False)[0]
#     i, j = grid.unbind(-1)
#     # the direction here is without +0.5 pixel centering as calibration is not so accurate
#     # see https://github.com/bmild/nerf/issues/24
#     cent = center if center is not None else [W / 2, H / 2]
#     directions = torch.stack([(i - cent[0]) / focal[0], (j - cent[1]) / focal[1], torch.ones_like(i)],
#                              -1)  # (H, W, 3)
#
#     return directions

class ScannetDataset(Dataset):
    def __init__(self, datadir, split='train', downsample=1.0, is_stack=False, N_vis=-1, rnd_ray=False,  max_len=-1, img_wh=[640, 480], args=None):
        self.white_bg = True
        self.args = args
        self.rnd_ray = rnd_ray
        self.N_vis = N_vis
        self.root_dir = datadir
        self.split = split
        self.is_stack = is_stack
        self.img_wh = (int(img_wh[0] * downsample), int(img_wh[1] * downsample))
        self.margin = self.args.margin if self.split=="train" else self.args.test_margin # self.opt.edge_filter

        self.max_len = max_len
        self.near_far = [0.5, 5.0] # [args.near_plane, args.far_plane]
        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.height, self.width = int(self.img_wh[1]), int(self.img_wh[0])
        self.define_transforms()
        # if not self.args.bg_color or self.args.bg_color == 'black':
        #     self.bg_color = (0, 0, 0)
        # elif self.args.bg_color == 'white':
        #     self.bg_color = (1, 1, 1)
        # elif self.args.bg_color == 'red':
        #     self.bg_color = (1, 0, 0)
        # elif self.args.bg_color == 'random':
        #     self.bg_color = 'random'
        # else:
        #     self.bg_color = [float(one) for one in self.args.bg_color.split(",")]

        self.read_meta()
        # print(self.intrinsic)
        self.total = len(self.id_list)
        print("dataset total:", self.split, self.total)
        # point_xyz = mvs_utils.load_ply_points(args)
        point_xyz = mvs_utils.load_init_depth_points(args, self.all_id_list, self.depth_intrinsic, device="cuda")
        self.scene_bbox = torch.stack([torch.min(point_xyz, dim=0)[0], torch.max(point_xyz, dim=0)[0]], dim=0)


    def normalize_cam(self, w2cs, c2ws):
        index = 0
        return w2cs[index], c2ws[index]

    def define_transforms(self):
        self.transform = T.ToTensor()

    def variance_of_laplacian(self, image):
        # compute the Laplacian of the image and then return the focus
        # measure, which is simply the variance of the Laplacian
        return cv2.Laplacian(image, cv2.CV_64F).var()

    def detect_blurry(self, list):
        blur_score = []
        for id in list:
            image_path = os.path.join(self.root_dir, "exported/color/{}.jpg".format(id))
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            fm = self.variance_of_laplacian(gray)
            blur_score.append(fm)
        blur_score = np.asarray(blur_score)
        ids = blur_score.argsort()[:150]
        allind = np.asarray(list)
        print("most blurry images", allind[ids])

    def remove_blurry(self, list):
        blur_path = os.path.join(self.root_dir, "exported/blur_list.txt")
        if os.path.exists(blur_path):
            blur_lst = []
            with open(blur_path) as f:
                lines = f.readlines()
                print("blur files", len(lines))
                for line in lines:
                    info = line.strip()
                    blur_lst.append(int(info))
            return [i for i in list if i not in blur_lst]
        else:
            print("no blur list detected, use all training frames!")
            return list


    def read_meta(self):
        w, h = self.img_wh
        colordir = os.path.join(self.root_dir, "exported/color")
        self.image_paths = [f for f in os.listdir(colordir) if os.path.isfile(os.path.join(colordir, f))]
        self.image_paths = [os.path.join(self.root_dir, "exported/color/{}.jpg".format(i)) for i in range(len(self.image_paths))]
        self.all_id_list = mvs_utils.filter_valid_id(self.args, list(range(len(self.image_paths))))
        # if len(self.all_id_list) > 2900: # neural point-based graphics' configuration
        #     self.test_id_list = self.all_id_list[::100]
        #     self.train_id_list = [self.all_id_list[i] for i in range(len(self.all_id_list)) if (((i % 100) > 19) and ((i % 100) < 81 or (i//100+1)*100>=len(self.all_id_list)))]
        # else:  # nsvf configuration
        step = 5
        if self.split == "train":
            self.id_list = self.all_id_list[::step]
            self.id_list = self.remove_blurry(self.id_list)
        else:
            # self.id_list = [self.all_id_list[i] for i in range(len(self.all_id_list)) if (i % step) !=0] if self.args.test_num_step != 1 else self.all_id_list
            self.id_list = self.all_id_list

        print("all_id_list",len(self.all_id_list))
        print("id_list",len(self.id_list), self.id_list)

        self.intrinsic = np.loadtxt(os.path.join(self.root_dir, "exported/intrinsic/intrinsic_color.txt")).astype(
            np.float32)[:3, :3]
        self.depth_intrinsic = np.loadtxt(
            os.path.join(self.root_dir, "exported/intrinsic/intrinsic_depth.txt")).astype(np.float32)[:3, :3]
        img = Image.open(self.image_paths[0])
        ori_img_shape = list(self.transform(img).shape)  # (4, h, w)
        self.intrinsic[0, :] *= (self.width / ori_img_shape[2])
        self.intrinsic[1, :] *= (self.height / ori_img_shape[1])

        self.directions, self.ij = get_ray_directions(h, w, [self.intrinsic[0,0], self.intrinsic[1,1]], center=[self.intrinsic[0,2], self.intrinsic[1,2]])  # (h, w, 3)
        if self.margin != 0:
            self.directions, self.ij = self.directions[self.margin:-self.margin, self.margin:-self.margin, :], self.ij[self.margin:-self.margin, self.margin:-self.margin, :]
        self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True)

        if self.split == "train":
            self.all_rays = []
            self.all_rgbs = []
            self.all_masks = []
            self.all_depth = []
            self.c2ws = []

            img_eval_interval = 1 if self.N_vis < 0 else len(self.meta['frames']) // self.N_vis
            idxs = list(range(0, len(self.id_list), img_eval_interval))
            for i in tqdm(idxs, desc=f'Loading data {self.split} ({len(idxs)})'):  # img_list:#

                # print("vid",vid)
                vid = self.id_list[i]
                c2w = np.loadtxt(os.path.join(self.root_dir, "exported/pose", "{}.txt".format(vid))).astype(np.float32)
                c2w = torch.FloatTensor(c2w)

                image_path = os.path.join(self.root_dir, "exported/color/{}.jpg".format(vid))
                img = Image.open(image_path)
                img = img.resize(self.img_wh, Image.Resampling.LANCZOS)
                img = self.transform(img)
                if self.margin != 0:
                    img = img[:, self.margin:-self.margin, self.margin:-self.margin]
                img = img.reshape(3, -1).permute(1, 0)  # (4, h, w)
                self.all_rgbs += [img]

                rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
                self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)

            if not self.is_stack:
                self.all_rays = torch.cat(self.all_rays, 0)  # (len(self.meta['frames])*h*w, 3)
                self.all_rgbs = torch.cat(self.all_rgbs, 0)  # (len(self.meta['frames])*h*w, 3)
                    # print("!!!!!!!!!!!!!!!!!!!!!!!!!self.all_alpha", self.all_alpha.shape)
            #             self.all_depth = torch.cat(self.all_depth, 0)  # (len(self.meta['frames])*h*w, 3)
            else:
                self.all_rays = torch.stack(self.all_rays, 0)  # (len(self.meta['frames]),h*w, 3)
                self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1, self.img_wh[-1] - 2 * self.margin, self.img_wh[-2] - 2 * self.margin, 3)
        else:
            self.all_rays = torch.arange(len(self.id_list))
            self.all_rgbs = self.all_rays


    def get_by_id(self, id):

        # print("vid",vid)
        vid = self.id_list[id]
        c2w = np.loadtxt(os.path.join(self.root_dir, "exported/pose", "{}.txt".format(vid))).astype(np.float32)
        c2w = torch.FloatTensor(c2w)

        image_path = os.path.join(self.root_dir, "exported/color/{}.jpg".format(vid))
        img = Image.open(image_path)
        img = img.resize(self.img_wh, Image.Resampling.LANCZOS)
        img = self.transform(img)
        if self.margin != 0:
            img = img[:, self.margin:-self.margin, self.margin:-self.margin]
        img = img.reshape(3, -1).permute(1, 0)  # (4, h, w)

        rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
        return torch.cat([rays_o, rays_d], 1), img.reshape(self.img_wh[-1] - 2 * self.margin, self.img_wh[-2] - 2 * self.margin, 3)


    def build_proj_mats(self, list=None, norm_w2c=None, norm_c2w=None):
        proj_mats, intrinsics, world2cams, cam2worlds = [], [], [], []
        list = self.id_list if list is None else list

        focal = 0.5 * 800 / np.tan(0.5 * self.meta['camera_angle_x'])  # original focal length
        focal *= self.img_wh[0] / 800  # modify focal length to match size self.img_wh
        self.focal = focal
        self.near_far = np.array([2.0, 6.0])
        for vid in list:
            frame = self.meta['frames'][vid]
            c2w = np.array(frame['transform_matrix']) # @ self.blender2opencv
            if norm_w2c is not None:
                c2w = norm_w2c @ c2w
            w2c = np.linalg.inv(c2w)
            cam2worlds.append(c2w)
            world2cams.append(w2c)

            intrinsic = np.array([[focal, 0, self.width / 2], [0, focal, self.height / 2], [0, 0, 1]])
            intrinsics.append(intrinsic.copy())

            # multiply intrinsics and extrinsics to get projection matrix
            proj_mat_l = np.eye(4)
            intrinsic[:2] = intrinsic[:2] / 4
            proj_mat_l[:3, :4] = intrinsic @ w2c[:3, :4]
            proj_mats += [(proj_mat_l, self.near_far)]

        proj_mats, intrinsics = np.stack(proj_mats), np.stack(intrinsics)
        world2cams, cam2worlds = np.stack(world2cams), np.stack(cam2worlds)
        return proj_mats, intrinsics, world2cams, cam2worlds


    def __len__(self):
        if self.split == 'train':
            return len(self.id_list) if self.max_len <= 0 else self.max_len
        return len(self.id_list) if self.max_len <= 0 else self.max_len


    def name(self):
        return 'NerfSynthFtDataset'


    def __del__(self):
        print("end loading")



    def __getitem__(self, id):

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


    # def normalize_rgb(self, data):
    #     # to unnormalize image for visualization
    #     # data C, H, W
    #     C, H, W = data.shape
    #     mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    #     std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    #     return (data - mean) / std


    # def __getitem__(self, id, crop=False, full_img=False):
    #
    #     item = {}
    #     vid = self.id_list[id]
    #     image_path = os.path.join(self.root_dir, "exported/color/{}.jpg".format(vid))
    #     # print("vid",vid)
    #     img = Image.open(image_path)
    #     img = img.resize(self.img_wh, Image.LANCZOS)
    #     img = self.transform(img)  # (4, h, w)
    #     c2w = np.loadtxt(os.path.join(self.root_dir, "exported/pose", "{}.txt".format(vid))).astype(np.float32)
    #     # w2c = np.linalg.inv(c2w)
    #     intrinsic = self.intrinsic
    #
    #     # print("gt_image", gt_image.shape)
    #     width, height = img.shape[2], img.shape[1]
    #     camrot = (c2w[0:3, 0:3])
    #     campos = c2w[0:3, 3]
    #     # print("camrot", camrot, campos)
    #
    #     item["intrinsic"] = intrinsic
    #     # item["intrinsic"] = sample['intrinsics'][0, ...]
    #     item["campos"] = torch.from_numpy(campos).float()
    #     item["c2w"] = torch.from_numpy(c2w).float()
    #     item["camrotc2w"] = torch.from_numpy(camrot).float() # @ FLIP_Z
    #
    #     dist = np.linalg.norm(campos)
    #
    #     middle = dist + 0.7
    #     item['middle'] = torch.FloatTensor([middle]).view(1, 1)
    #     item['far'] = torch.FloatTensor([self.near_far[1]]).view(1, 1)
    #     item['near'] = torch.FloatTensor([self.near_far[0]]).view(1, 1)
    #     item['h'] = height
    #     item['w'] = width
    #     item['id'] = id
    #     item['vid'] = vid
    #     # bounding box
    #     margin = self.args.edge_filter
    #     if full_img:
    #         item['images'] = img[None,...].clone()
    #     gt_image = np.transpose(img, (1, 2, 0))
    #     subsamplesize = self.args.random_sample_size
    #     if self.args.random_sample == "patch":
    #         indx = np.random.randint(margin, width - margin - subsamplesize + 1)
    #         indy = np.random.randint(margin, height - margin - subsamplesize + 1)
    #         px, py = np.meshgrid(
    #             np.arange(indx, indx + subsamplesize).astype(np.float32),
    #             np.arange(indy, indy + subsamplesize).astype(np.float32))
    #     elif self.args.random_sample == "random":
    #         px = np.random.randint(margin,
    #                                width-margin,
    #                                size=(subsamplesize,
    #                                      subsamplesize)).astype(np.float32)
    #         py = np.random.randint(margin,
    #                                height-margin,
    #                                size=(subsamplesize,
    #                                      subsamplesize)).astype(np.float32)
    #     elif self.args.random_sample == "random2":
    #         px = np.random.uniform(margin,
    #                                width - margin - 1e-5,
    #                                size=(subsamplesize,
    #                                      subsamplesize)).astype(np.float32)
    #         py = np.random.uniform(margin,
    #                                height - margin - 1e-5,
    #                                size=(subsamplesize,
    #                                      subsamplesize)).astype(np.float32)
    #     elif self.args.random_sample == "proportional_random":
    #         raise Exception("no gt_mask, no proportional_random !!!")
    #     else:
    #         px, py = np.meshgrid(
    #             np.arange(margin, width - margin).astype(np.float32),
    #             np.arange(margin, height- margin).astype(np.float32))
    #     pixelcoords = np.stack((px, py), axis=-1).astype(np.float32)  # H x W x 2
    #     # raydir = get_cv_raydir(pixelcoords, self.height, self.width, focal, camrot)
    #     item["pixel_idx"] = pixelcoords
    #     # print("pixelcoords", pixelcoords.reshape(-1,2)[:10,:])
    #     raydir = get_dtu_raydir(pixelcoords, item["intrinsic"], camrot, self.args.dir_norm > 0)
    #     raydir = np.reshape(raydir, (-1, 3))
    #     item['raydir'] = torch.from_numpy(raydir).float()
    #     gt_image = gt_image[py.astype(np.int32), px.astype(np.int32)]
    #     # gt_mask = gt_mask[py.astype(np.int32), px.astype(np.int32), :]
    #     gt_image = np.reshape(gt_image, (-1, 3))
    #     item['gt_image'] = gt_image
    #
    #     if self.bg_color:
    #         if self.bg_color == 'random':
    #             val = np.random.rand()
    #             if val > 0.5:
    #                 item['bg_color'] = torch.FloatTensor([1, 1, 1])
    #             else:
    #                 item['bg_color'] = torch.FloatTensor([0, 0, 0])
    #         else:
    #             item['bg_color'] = torch.FloatTensor(self.bg_color)
    #
    #     return item
    #
    #
    #
    # def get_item(self, idx, crop=False, full_img=False):
    #     item = self.__getitem__(idx, crop=crop, full_img=full_img)
    #
    #     for key, value in item.items():
    #         if not isinstance(value, str):
    #             if not torch.is_tensor(value):
    #                 value = torch.as_tensor(value)
    #             item[key] = value.unsqueeze(0)
    #     return item


    #
    # def get_dummyrot_item(self, idx, crop=False):
    #
    #     item = {}
    #     width, height = self.width, self.height
    #
    #     transform_matrix = self.render_poses[idx]
    #     camrot = (transform_matrix[0:3, 0:3])
    #     campos = transform_matrix[0:3, 3]
    #     focal = self.focal
    #
    #     item["focal"] = focal
    #     item["campos"] = torch.from_numpy(campos).float()
    #     item["camrotc2w"] = torch.from_numpy(camrot).float()
    #     item['lightpos'] = item["campos"]
    #
    #     dist = np.linalg.norm(campos)
    #
    #     # near far
    #     if self.args.near_plane is not None:
    #         near = self.args.near_plane
    #     else:
    #         near = max(dist - 1.5, 0.02)
    #     if self.args.far_plane is not None:
    #         far = self.args.far_plane  # near +
    #     else:
    #         far = dist + 0.7
    #     middle = dist + 0.7
    #     item['middle'] = torch.FloatTensor([middle]).view(1, 1)
    #     item['far'] = torch.FloatTensor([far]).view(1, 1)
    #     item['near'] = torch.FloatTensor([near]).view(1, 1)
    #     item['h'] = self.height
    #     item['w'] = self.width
    #
    #
    #     subsamplesize = self.args.random_sample_size
    #     if self.args.random_sample == "patch":
    #         indx = np.random.randint(0, width - subsamplesize + 1)
    #         indy = np.random.randint(0, height - subsamplesize + 1)
    #         px, py = np.meshgrid(
    #             np.arange(indx, indx + subsamplesize).astype(np.float32),
    #             np.arange(indy, indy + subsamplesize).astype(np.float32))
    #     elif self.args.random_sample == "random":
    #         px = np.random.randint(0,
    #                                width,
    #                                size=(subsamplesize,
    #                                      subsamplesize)).astype(np.float32)
    #         py = np.random.randint(0,
    #                                height,
    #                                size=(subsamplesize,
    #                                      subsamplesize)).astype(np.float32)
    #     elif self.args.random_sample == "random2":
    #         px = np.random.uniform(0,
    #                                width - 1e-5,
    #                                size=(subsamplesize,
    #                                      subsamplesize)).astype(np.float32)
    #         py = np.random.uniform(0,
    #                                height - 1e-5,
    #                                size=(subsamplesize,
    #                                      subsamplesize)).astype(np.float32)
    #     elif self.args.random_sample == "proportional_random":
    #         px, py = self.proportional_select(gt_mask)
    #     else:
    #         px, py = np.meshgrid(
    #             np.arange(width).astype(np.float32),
    #             np.arange(height).astype(np.float32))
    #
    #     pixelcoords = np.stack((px, py), axis=-1).astype(np.float32)  # H x W x 2
    #     # raydir = get_cv_raydir(pixelcoords, self.height, self.width, focal, camrot)
    #     raydir = get_blender_raydir(pixelcoords, self.height, self.width, focal, camrot, self.args.dir_norm > 0)
    #     item["pixel_idx"] = pixelcoords
    #     raydir = np.reshape(raydir, (-1, 3))
    #     item['raydir'] = torch.from_numpy(raydir).float()
    #
    #     if self.bg_color:
    #         if self.bg_color == 'random':
    #             val = np.random.rand()
    #             if val > 0.5:
    #                 item['bg_color'] = torch.FloatTensor([1, 1, 1])
    #             else:
    #                 item['bg_color'] = torch.FloatTensor([0, 0, 0])
    #         else:
    #             item['bg_color'] = torch.FloatTensor(self.bg_color)
    #
    #     for key, value in item.items():
    #         if not torch.is_tensor(value):
    #             value = torch.as_tensor(value)
    #         item[key] = value.unsqueeze(0)
    #
    #     return item



    # def get_init_item(self, idx, crop=False):
    #     sample = {}
    #     init_view_num = self.args.init_view_num
    #     view_ids = self.view_id_list[idx]
    #     if self.split == 'train':
    #         view_ids = view_ids[:init_view_num]
    #
    #     affine_mat, affine_mat_inv = [], []
    #     mvs_images, imgs, depths_h, alphas = [], [], [], []
    #     proj_mats, intrinsics, w2cs, c2ws, near_fars = [], [], [], [], []  # record proj mats between views
    #     for i in view_ids:
    #         vid = self.view_id_dict[i]
    #         # mvs_images += [self.normalize_rgb(self.blackimgs[vid])]
    #         # mvs_images += [self.whiteimgs[vid]]
    #         mvs_images += [self.blackimgs[vid]]
    #         imgs += [self.whiteimgs[vid]]
    #         proj_mat_ls, near_far = self.proj_mats[vid]
    #         intrinsics.append(self.intrinsics[vid])
    #         w2cs.append(self.world2cams[vid])
    #         c2ws.append(self.cam2worlds[vid])
    #
    #         affine_mat.append(proj_mat_ls)
    #         affine_mat_inv.append(np.linalg.inv(proj_mat_ls))
    #         depths_h.append(self.depths[vid])
    #         alphas.append(self.alphas[vid])
    #         near_fars.append(near_far)
    #
    #     for i in range(len(affine_mat)):
    #         view_proj_mats = []
    #         ref_proj_inv = affine_mat_inv[i]
    #         for j in range(len(affine_mat)):
    #             if i == j:  # reference view
    #                 view_proj_mats += [np.eye(4)]
    #             else:
    #                 view_proj_mats += [affine_mat[j] @ ref_proj_inv]
    #         # view_proj_mats: 4, 4, 4
    #         view_proj_mats = np.stack(view_proj_mats)
    #         proj_mats.append(view_proj_mats[:, :3])
    #     # (4, 4, 3, 4)
    #     proj_mats = np.stack(proj_mats)
    #     imgs = np.stack(imgs).astype(np.float32)
    #     mvs_images = np.stack(mvs_images).astype(np.float32)
    #
    #     depths_h = np.stack(depths_h)
    #     alphas = np.stack(alphas)
    #     affine_mat, affine_mat_inv = np.stack(affine_mat), np.stack(affine_mat_inv)
    #     intrinsics, w2cs, c2ws, near_fars = np.stack(intrinsics), np.stack(w2cs), np.stack(c2ws), np.stack(near_fars)
    #
    #     sample['images'] = imgs  # (V, 3, H, W)
    #     sample['mvs_images'] = mvs_images  # (V, 3, H, W)
    #     sample['depths_h'] = depths_h.astype(np.float32)  # (V, H, W)
    #     sample['alphas'] = alphas.astype(np.float32)  # (V, H, W)
    #     sample['w2cs'] = w2cs.astype(np.float32)  # (V, 4, 4)
    #     sample['c2ws'] = c2ws.astype(np.float32)  # (V, 4, 4)
    #     sample['near_fars'] = near_fars.astype(np.float32)
    #     sample['proj_mats'] = proj_mats.astype(np.float32)
    #     sample['intrinsics'] = intrinsics.astype(np.float32)  # (V, 3, 3)
    #     sample['view_ids'] = np.array(view_ids)
    #     sample['affine_mat'] = affine_mat
    #     sample['affine_mat_inv'] = affine_mat_inv
    #     for key, value in sample.items():
    #         if not isinstance(value, str):
    #             if not torch.is_tensor(value):
    #                 value = torch.as_tensor(value)
    #                 sample[key] = value.unsqueeze(0)
    #
    #     return sample



    # def get_campos_ray(self):
    #     centerpixel=np.asarray(self.img_wh).astype(np.float32)[None,:] // 2
    #     camposes=[]
    #     centerdirs=[]
    #     for id in self.id_list:
    #         c2w = np.loadtxt(os.path.join(self.root_dir, "exported/pose", "{}.txt".format(id))).astype(np.float32)  #@ self.blender2opencv
    #         campos = c2w[:3, 3]
    #         camrot = c2w[:3,:3]
    #         raydir = get_dtu_raydir(centerpixel, self.intrinsic, camrot, True)
    #         camposes.append(campos)
    #         centerdirs.append(raydir)
    #     camposes=np.stack(camposes, axis=0) # 2091, 3
    #     centerdirs=np.concatenate(centerdirs, axis=0) # 2091, 3
    #     # print("camposes", camposes.shape, centerdirs.shape)
    #     return torch.as_tensor(camposes, device="cuda", dtype=torch.float32), torch.as_tensor(centerdirs, device="cuda", dtype=torch.float32)



    # @staticmethod
    # def modify_commandline_argsions(parser, is_train):
    #     # ['random', 'random2', 'patch'], default: no random sample
    #     parser.add_argument('--random_sample',
    #                         type=str,
    #                         default='none',
    #                         help='random sample pixels')
    #     parser.add_argument('--random_sample_size',
    #                         type=int,
    #                         default=1024,
    #                         help='number of random samples')
    #     parser.add_argument('--init_view_num',
    #                         type=int,
    #                         default=3,
    #                         help='number of random samples')
    #     parser.add_argument('--edge_filter',
    #                         type=int,
    #                         default=3,
    #                         help='number of random samples')
    #     parser.add_argument('--shape_id', type=int, default=0, help='shape id')
    #     parser.add_argument('--trgt_id', type=int, default=0, help='shape id')
    #     parser.add_argument('--num_nn',
    #                         type=int,
    #                         default=1,
    #                         help='number of nearest views in a batch')
    #     parser.add_argument(
    #         '--near_plane',
    #         type=float,
    #         default=0.5,
    #         help=
    #         'Near clipping plane, by default it is computed according to the distance of the camera '
    #     )
    #     parser.add_argument(
    #         '--far_plane',
    #         type=float,
    #         default=5.0,
    #         help=
    #         'Far clipping plane, by default it is computed according to the distance of the camera '
    #     )
    #
    #     parser.add_argument(
    #         '--bg_color',
    #         type=str,
    #         default="white",
    #         help=
    #         'background color, white|black(None)|random|rgb (float, float, float)'
    #     )
    #
    #     parser.add_argument(
    #         '--scan',
    #         type=str,
    #         default="scan1",
    #         help=''
    #     )
    #
    #     parser.add_argument('--inverse_gamma_image',
    #                         type=int,
    #                         default=-1,
    #                         help='de-gamma correct the input image')
    #     parser.add_argument('--pin_data_in_memory',
    #                         type=int,
    #                         default=-1,
    #                         help='load whole data in memory')
    #     parser.add_argument('--normview',
    #                         type=int,
    #                         default=0,
    #                         help='load whole data in memory')
    #     parser.add_argument(
    #         '--id_range',
    #         type=int,
    #         nargs=3,
    #         default=(0, 385, 1),
    #         help=
    #         'the range of data ids selected in the original dataset. The default is range(0, 385). If the ids cannot be generated by range, use --id_list to specify any ids.'
    #     )
    #     parser.add_argument(
    #         '--id_list',
    #         type=int,
    #         nargs='+',
    #         default=None,
    #         help=
    #         'the list of data ids selected in the original dataset. The default is range(0, 385).'
    #     )
    #     parser.add_argument(
    #         '--split',
    #         type=str,
    #         default="train",
    #         help=
    #         'train, val, test'
    #     )
    #     parser.add_argument("--half_res", action='store_true',
    #                         help='load blender synthetic data at 400x400 instead of 800x800')
    #     parser.add_argument("--testskip", type=int, default=8,
    #                         help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')
    #     parser.add_argument('--dir_norm',
    #                         type=int,
    #                         default=0,
    #                         help='normalize the ray_dir to unit length or not, default not')
    #
    #     parser.add_argument('--train_load_num',
    #                         type=int,
    #                         default=0,
    #                         help='normalize the ray_dir to unit length or not, default not')
    #     parser.add_argument(
    #         '--img_wh',
    #         type=int,
    #         nargs=2,
    #         default=(640, 480),
    #         help='resize target of the image'
    #     )
    #     return parser

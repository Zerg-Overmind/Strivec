import torch,cv2
from torch.utils.data import Dataset
import json
from tqdm import tqdm
import os
from PIL import Image
from torchvision import transforms as T
from .data_utils import *
import sys
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../models/")
sys.path.append(models_dir)
from apparatus import draw_ray, draw_box
from .ray_utils import *
from plyfile import PlyData

class IndoorDataset(Dataset):
    def __init__(self, datadir, split='train', downsample=1.0, is_stack=False, N_vis=-1, rnd_ray=False, args=None):

        self.N_vis = N_vis
        self.root_dir = datadir
        self.split = split
        self.is_stack = is_stack
        self.downsample = downsample
        self.define_transforms()

        self.scene_bbox = torch.tensor([[-51.2, -51.2, -51.2], [25.8, 25.8, 25.8]])  # [-51.2, -51.2, -1.5], [25.8, 25.8, 25.8]
        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.opengl = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, -1], [1, 1, 1, 1]])
        self.read_meta()
        #self.define_proj_mat()

        self.white_bg = False
        self.near_far = [0, 50.0]

        self.unbounded_inward = False
        self.unbounded_inner_r = 0.0
        self.flip_y = False
        self.flip_x = False
        self.inverse_y = False
        self.ndc = False
        self.near_clip = None
        self.irregular_shape = False
 

        self.center = torch.mean(self.scene_bbox, axis=0).float().view(1, 1, 3)
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)
        self.downsample=downsample

    def read_depth(self, filename):
        depth = np.array(read_pfm(filename)[0], dtype=np.float32)  # not (800, 800), no depth map in enes
        return depth
    
    def read_meta(self):
        self.meta = []
        img_idx = []
        img_w = []
        img_h = []
        R_matrix = []
        t_matrix = [] 
        focal_length = []

        
       
        with open(os.path.join(self.root_dir, "scene_metadata_jpg.txt"), 'r') as f:
            imgs = f.readlines()
        meta_raw = imgs[4:326]
        
        with open(os.path.join(self.root_dir, "cameras/bundle.out"), 'r') as f1:
            trans_mtrx = f1.readlines()
        transform_matrix_raw = trans_mtrx[2:1612]

        # list of train data or test data
        with open(os.path.join(self.root_dir, f"{self.split}_list.txt"), 'r') as f_lst:
            lst = f_lst.readlines()

        ## plot scene box
        #logfolder = '/home/gqk/cloud_tensoRF/log/indoor_scnenes'
        #center = (self.scene_bbox[0]+self.scene_bbox[1])/2
        #box_range = (self.scene_bbox[1]-self.scene_bbox[0])/2
        #draw_box(center, box_range, logfolder, 0, rot_m=None) 
  
        for meta_num in range(len(lst)):
            img_idx.append(meta_raw[int(lst[meta_num])].split()[0])
            img_w.append(np.array(meta_raw[int(lst[meta_num])].split()[1]))
            img_h.append(np.array(meta_raw[int(lst[meta_num])].split()[2]))
            
            R_matrix.append([np.array(transform_matrix_raw[int(lst[meta_num])*5+1].split()), np.array(transform_matrix_raw[int(lst[meta_num])*5+2].split()), np.array(transform_matrix_raw[int(lst[meta_num])*5+3].split()), np.array([0,0,0])])
            tt = transform_matrix_raw[int(lst[meta_num])*5+4].split()
            tt.append(1)
            t_matrix.append([np.array(tt)])
            #R_matrix.append([np.array(transform_matrix_raw[meta_num*5+1].split()), np.array(transform_matrix_raw[meta_num*5+2].split()), np.array(transform_matrix_raw[meta_num*5+3].split()), np.array(transform_matrix_raw[meta_num*5+4].split())])
                      
            focal_length.append(np.array(transform_matrix_raw[int(lst[meta_num])*5].split())[0])
      
       
        transform_matrix = np.concatenate((np.array(R_matrix, dtype=np.float32), np.array(t_matrix, dtype=np.float32).transpose(0,2,1)), axis=2)
        #matrix_fill = np.array([[0,0,0,1]])
        #transform_matrix = np.concatenate((np.array(R_matrix, dtype=np.float32).transpose(0,2,1), matrix_fill.repeat([len(meta_raw)], axis=0)[:,None,:]), axis=1)
       
        focal = np.array(focal_length, dtype=np.float32)
        img_h = np.array(img_h, dtype=np.int)+1
        img_w = np.array(img_w, dtype=np.int)+1
    
        #w, h = int(self.meta['w']/self.downsample), int(self.meta['h']/self.downsample)

        # self.img_wh = [w,h]
        # self.fx = self.near * (self.width-1)/2. 
        # self.fy = self.near * (self.height-1)/2. 
        # self.fy = self.fx 
        # self.cx = (self.width-1)/2. 
        # self.cy = (self.height-1)/2.

        # self.focal_x = 0.5 * w / np.tan(0.5 * self.meta['camera_angle_x'])  # original focal length
        # self.focal_y = 0.5 * h / np.tan(0.5 * self.meta['camera_angle_y'])  # original focal length
        # self.cx, self.cy = self.meta['cx'],self.meta['cy']


        # # ray directions for all pixels, same for all images (same H, W, focal)
        # self.directions = get_ray_directions(h, w, [self.focal_x,self.focal_y], center=[self.cx, self.cy])  # (h, w, 3)
        # self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True)
        # self.intrinsics = torch.tensor([[self.focal_x,0,self.cx],[0,self.focal_y,self.cy],[0,0,1]]).float()

        self.image_paths = []
        self.poses = []
        self.all_rays = []

        self.all_rays_vis_edge = []
        self.all_rays_vis_vert = [] 

        self.all_rgbs = []
        self.all_rgbs_stack = []
        self.all_masks = []
        self.all_depth = []
        self.proj_mat = []
        self.img_wh = []
        self.raw_poses = []
        self.intrinsics = []
        self.camera_dir = []
        #img_eval_interval = 1 if self.N_vis < 0 else len(self.meta['frames']) // self.N_vis
        img_eval_interval = 1 if self.N_vis < 0 else len(lst) // self.N_vis
       
        idxs = list(range(0, len(lst), img_eval_interval))
        for i in tqdm(idxs, desc=f'Loading data {self.split} ({len(lst)})'):#img_list:#
            pose = np.linalg.inv(np.array(transform_matrix[i])) 
            #pose = -np.array(transform_matrix[i][:3,:3]).T*np.array(transform_matrix[i][:3,3]).T # * self.opengl  #@ self.blender2opencv
            c2w = torch.FloatTensor(pose)
            self.poses += [c2w]
            self.raw_poses += [torch.FloatTensor(np.array(transform_matrix[i]))]
            image_path = os.path.join(self.root_dir, f"{self.split}/{img_idx[i]}")
            self.image_paths += [image_path]
            img = Image.open(image_path)
            #if self.downsample!=1.0:
            #    img = img.resize([img_w[i], img_h[i]], Image.LANCZOS)
            img = self.transform(img)  # (4, h, w)
            self.img_wh += [[img_w[i], img_h[i]]]
            ####### when we have RGBA images
            #img = img.view(-1, img_w[i]*img_h[i]).permute(1, 0)  # (h*w, 4) RGBA
            #if img.shape[-1]==4:
            #    img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB
            #######
           
            self.all_rgbs_stack += [img.permute(1, 2, 0)]
            img = img.view(-1, img_w[i]*img_h[i]).permute(1, 0)
            self.all_rgbs += [img]
            
 
            cx = (img_w[i]-1)/2. 
            cy = (img_h[i]-1)/2.
            self.directions, _ = get_ray_directions(img_h[i], img_w[i], [focal[i], focal[i]])

            self.directions =self.directions / torch.norm(self.directions, dim=-1, keepdim=True)
            intrinsics = torch.tensor([[focal[i], 0, cx],[0, focal[i], cy], [0, 0, 1]]).float()
            
            if i == 0: 
                self.proj_mat = intrinsics.unsqueeze(0) @ torch.inverse(c2w)[:3]
            else:
                self.proj_mat = torch.cat((self.proj_mat, intrinsics.unsqueeze(0) @ torch.inverse(c2w)[:3]), dim=2)

            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
            self.camera_dir += [torch.cat([rays_o[(img_h[i]//2)*(img_w[i]//2),:], rays_o[(img_h[i]//2)*(img_w[i]//2),:]+40*rays_d[(img_h[i]//2)*(img_w[i]//2),:]], 0)]
            self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)
            #self.all_rays_vis_edge += [torch.stack([rays_o+rays_d*0.01, rays_o+rays_d*20], 2)]
            #self.all_rays_vis_vert += [torch.cat([rays_o+rays_d*0.01, rays_o+rays_d*20], 0)]
            self.intrinsics += [intrinsics] 

       
        self.intrinsics = torch.stack(self.intrinsics, 0)
        self.proj_mat = torch.tensor(self.proj_mat)
        self.raw_poses = torch.stack(self.raw_poses)
        self.poses = torch.stack(self.poses)
        self.camera_dir = torch.stack(self.camera_dir)

        #self.all_rays_vis_edge = torch.cat(self.all_rays_vis_edge, 0)
        #self.all_rays_vis_vert = torch.cat(self.all_rays_vis_vert, 0)
        

        if not self.is_stack:  
            self.all_rays = torch.cat(self.all_rays, 0)  # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0)  # (len(self.meta['frames])*h*w, 3)
            #self.all_depth = torch.cat(self.all_depth, 0)  # (len(self.meta['frames])*h*w, 3)
        #else:
        #    self.all_rays = torch.stack(self.all_rays, 0)  # (len(self.meta['frames]),h*w, 3)
        #    self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3)  # (len(self.meta['frames]),h,w,3)

            # self.all_masks = torch.stack(self.all_masks, 0).reshape(-1,*self.img_wh[::-1])  # (len(self.meta['frames]),h,w,3)
        
        #np.savetxt(f"/home/gqk/cloud_tensoRF/log/indoor_scnenes/camera_posit.txt", self.camera_dir[:,:3].numpy(), delimiter=";")

        #draw_ray(self.camera_dir.reshape([self.camera_dir.shape[0]*2, 3]).numpy(), self.camera_dir.numpy(), 0.01, 30)
        #exit()
       

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
                      'rgbs': img}
        return sample



class IndoorMVSDataset(Dataset):

    def __init__(self, datadir, split='train', downsample=1.0, is_stack=False, N_vis=-1):
        self.data_dir = datadir
        self.split = split

        #self.img_wh = (int(800 * downsample), int(800 * downsample))
        self.downsample = downsample

        self.scale_factor = 1.0 / 1.0

        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.opengl = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        #self.height, self.width = int(self.img_wh[1]), int(self.img_wh[0])

        self.white_bg = True

        self.define_transforms()

        with open(os.path.join(self.root_dir, "scene_metadata_jpg.txt"), 'r') as f:
            imgs = f.readlines()
        meta_raw = imgs[4:326]
        self.id_list = [i for i in range(len(meta_raw))]

        with open(os.path.join(self.root_dir, "cameras/bundle.out"), 'r') as f1:
            trans_mtrx = f1.readlines()
        self.transform_matrix_raw = trans_mtrx[2:1612]
        self.meta = []
        img_idx = []
        img_w = []
        img_h = []
        R_matrix = []
        t_matrix = [] 
        focal_length = []

        # list of train data or test data
        with open(os.path.join(self.root_dir, f"{self.split}_list.txt"), 'r') as f_lst:
            lst = f_lst.readlines()

        for meta_num in range(len(meta_raw)):
            img_idx.append(meta_raw[int(lst[meta_num])].split()[0])
            img_w.append(np.array(meta_raw[int(lst[meta_num])].split()[1]))
            img_h.append(np.array(meta_raw[int(lst[meta_num])].split()[2]))
          
            R_matrix.append([np.array(self.transform_matrix_raw[int(lst[meta_num])*5+1].split()), np.array(self.transform_matrix_raw[int(lst[meta_num])*5+2].split()), np.array(self.transform_matrix_raw[int(lst[meta_num])*5+3].split()), np.array([0,0,0])])
            tt = self.transform_matrix_raw[int(lst[meta_num])*5+4].split()
            tt.append(1)
            t_matrix.append([np.array(tt)])

            focal_length.append(np.array(self.transform_matrix_raw[int(lst[meta_num])*5].split())[0])
        
       
        self.transform_matrix = np.concatenate((np.array(R_matrix, dtype=np.float32), np.array(t_matrix, dtype=np.float32).transpose(0,2,1)), axis=2)
        self.focal = np.array(focal_length, dtype=np.float32)
        self.img_h = np.array(img_h, dtype=np.int)+1
        self.img_w = np.array(img_w, dtype=np.int)+1



        self.proj_mats, self.intrinsics, self.world2cams, self.cam2worlds = self.build_proj_mats()
        self.build_init_metas()
        #self.read_meta()
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
        #focal = 0.5 * 800 / np.tan(0.5 * self.meta['camera_angle_x'])  # original focal length
        #focal *= self.img_wh[0] / 800  # modify focal length to match size self.img_wh
        #self.focal = focal
        self.near_far = np.array([0.1, 45.0])

 
        for vid in list:
            c2w = np.array(self.transform_matrix[vid]) #@ self.blender2opencv
            w2c = np.linalg.inv(c2w)
            cam2worlds.append(c2w)
            world2cams.append(w2c)

            intrinsic = np.array([[self.focal[vid], 0, self.img_w[vid] / 2], [0, self.focal[vid], self.img_h[vid] / 2], [0, 0, 1]])
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
            img = img.resize(self.img_wh, Image.LANCZOS)
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

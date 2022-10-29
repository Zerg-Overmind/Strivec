from .models import *
from .mvs_utils import *
from . import filter_utils
from collections import OrderedDict
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from .mvsnet.mvsnet import MVSNet
feature_str_lst=['appr_feature_str0', 'appr_feature_str1', 'appr_feature_str2', 'appr_feature_str3']



def create_mvs(args):
    """Instantiate mvs NeRF's MLP model.
    """
    net_2d = FeatureNet(intermediate=True).to(device)
    EncodingNet = MVSNet(refine=False).to(device)
    EncodingNet.eval()
    render_kwargs_train = {
        'network_featmvs': EncodingNet,
        'network_2d': net_2d,
    }

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False

    return render_kwargs_train, render_kwargs_test


class MvsPointsModel(nn.Module):

    def __init__(self, args):
        super(MvsPointsModel, self).__init__()
        self.args = args
        self.args.feat_dim = 8+3*4
        self.idx = 0

        # Create nerf model
        self.render_kwargs_train, self.render_kwargs_test = create_mvs(args)
        # filter_keys(self.render_kwargs_train)

        # Create mvs model
        self.ReconNet = self.render_kwargs_train['network_featmvs']
        self.load_pretrained_d_est(self.ReconNet, args.pretrained_mvs_ckpt)
        self.render_kwargs_train.pop('network_featmvs')
        self.render_kwargs_train.pop('network_2d')
        self.render_kwargs_train['NDC_local'] = False
        # self.cnt = 0


    def load_pretrained_d_est(self, model, pretrained_mvs_ckpt):
        print("loading model {}".format(pretrained_mvs_ckpt))
        state_dict = torch.load(pretrained_mvs_ckpt, map_location=lambda storage, loc: storage)
        new_state_dict = OrderedDict()
        for k, v in state_dict['model'].items():
            name = k[7:]  # remove module.
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument("--mvs_lr", type=float, default=5e-4,
                            help='learning rate')
        parser.add_argument('--pad', type=int, default=24)
        parser.add_argument('--depth_grid', type=int, default=128)
        parser.add_argument('--prob_thresh', type=float, default=0.8)
        parser.add_argument('--dprob_thresh', type=float, default=0.8)
        parser.add_argument('--num_neighbor', type=int, default=1)
        parser.add_argument('--depth_vid', type=str, default="0", help="0123")
        parser.add_argument('--ref_vid', type=int, default=0, help="0, 1, 2, or 3")
        parser.add_argument('--num_each_depth', type=int, default=1)
        parser.add_argument('--depth_conf_thresh', type=float, default=None)
        parser.add_argument('--depth_occ', type=int, default=0)
        parser.add_argument('--manual_depth_view', type=int, default=0, help="-1 for learning probability, 0 for gt, 1 for pretrained MVSNet")
        parser.add_argument('--pre_d_est', type=str, default=None, help="loading pretrained depth estimator")
        parser.add_argument('--manual_std_depth', type=float, default=0)
        parser.add_argument('--far_plane_shift', type=float, default=None)
        parser.add_argument('--appr_feature_str1',
            type=str,
            nargs='+',
            # default=["imgfeat_0_0123", "vol"],
            default=["imgfeat_0_0", "vol"],
            help=
            "which feature_map")
        parser.add_argument('--appr_feature_str2',
            type=str,
            nargs='+',
            # default=["imgfeat_0_0123", "vol"],
            default=["imgfeat_0_0", "vol"],
            help=
            "which feature_map")
        parser.add_argument('--appr_feature_str3',
            type=str,
            nargs='+',
            # default=["imgfeat_0_0123", "vol"],
            default=["imgfeat_0_0", "vol"],
            help=
            "which feature_map")
        parser.add_argument('--vox_res', type=int, default=0, help='vox_resolution if > 0')


    def decode_batch(self, batch, idx=list(torch.arange(4))):
        data_mvs = sub_selete_data(batch, device, idx, filtKey=[])
        pose_ref = {'w2cs': data_mvs['w2cs'].squeeze(), 'intrinsics': data_mvs['intrinsics'].squeeze(),
                    'c2ws': data_mvs['c2ws'].squeeze(),'near_fars':data_mvs['near_fars'].squeeze()}
        return data_mvs, pose_ref


    def normalize_rgb(self, data, shape=(1,1,3,1,1)):
        # to unnormalize image for visualization
        # data N V C H W
        device = data.device
        mean = torch.tensor([0.485, 0.456, 0.406]).view(*shape).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(*shape).to(device)
        return (data - mean) / std


    def sample_func(self, volume_prob, args, ref_intrinsic, near_far, cam_expected_depth=None, ndc_std_depth=None):
        # volume_prob # ([1, 1, 128, 176, 208])
        if cam_expected_depth is None:
            B, C, D, H, W = volume_prob.shape
            v = 1.0 / D
            ndc_depths = torch.linspace(0.5 * v, 1.0 - 0.5 * v, steps=D, device=volume_prob.device)[None, None, :, None, None].expand(1, 1, -1, H, W)
            # B, C, H, W
            ndc_expected_depth = torch.sum(volume_prob * ndc_depths, dim=2)  # ([1, 1, 1, 176, 208])
            ndc_std_depth = torch.sqrt(torch.sum(volume_prob * torch.square(ndc_depths-ndc_expected_depth), dim=2)) #([1, 1, 176, 208])
            mask = self.prob_filter(args.dprob_thresh, args.num_neighbor, volume_prob, ndc_expected_depth, ndc_std_depth)
        else:
            # [1, 1, 512, 640]
            mask = torch.logical_and(cam_expected_depth >= near_far[0], cam_expected_depth <= near_far[1])
            ndc_expected_depth = (cam_expected_depth - near_far[0]) / (near_far[1] - near_far[0])
        sampled_depth = self.sample_by_gau(ndc_expected_depth, ndc_std_depth, args) #([1, 1, 5, 512, 640])
        ndc_xyz, cam_xyz = self.depth2point(sampled_depth, ref_intrinsic, near_far) # 1, 1, 512, 640, 3

        return ndc_xyz, cam_xyz, ndc_expected_depth.shape[-2:], mask


    def sample_by_gau(self, ndc_expected_depth, ndc_std_depth, args):

        B, C, H, W = ndc_expected_depth.shape
        N = 1
        # [1, 5, 1, 176, 208]
        sampled_depth = ndc_std_depth[:,None,...] * torch.normal(mean=torch.zeros((B, N, C, H, W), device="cuda"), std=torch.ones((B, N, C, H, W), device=ndc_expected_depth.device)) + ndc_expected_depth[:,None,...]
        return torch.clamp(sampled_depth, min=0.0, max=1.0)


    def depth2point(self, sampled_depth, ref_intrinsic, near_far):
        B, N, C, H, W = sampled_depth.shape
        valid_z = sampled_depth
        valid_x = torch.arange(W, dtype=torch.float32, device=sampled_depth.device) / (W - 1)
        valid_y = torch.arange(H, dtype=torch.float32, device=sampled_depth.device) / (H - 1)
        valid_y, valid_x = torch.meshgrid(valid_y, valid_x)
        # B,N,H,W
        valid_x = valid_x[None, None, None, ...].expand(B, N, C, -1, -1)
        valid_y = valid_y[None, None, None, ...].expand(B, N, C, -1, -1)
        ndc_xyz = torch.stack([valid_x, valid_y, valid_z], dim=-1).view(B, N, C, H, W, 3)  # 1, 1, 5, 512, 640, 3
        cam_xyz = ndc_2_cam(ndc_xyz, near_far, ref_intrinsic, W, H) # 1, 1, 5, 512, 640, 3
        return ndc_xyz, cam_xyz


    def prob_filter(self, thresh, num_neighbor, volume_prob, ndc_expected_depth, ndc_std_depth):
        B, C, D, H, W = volume_prob.shape
        ceil_idx = torch.ceil(ndc_expected_depth)
        lower_idx = ceil_idx - num_neighbor // 2 + 1 # B, C, 1, H, W
        # upper_idx = ceil_idx + num_neighbor // 2
        shifts = torch.arange(0, num_neighbor, device=volume_prob.device, dtype=torch.int64)[None, :, None, None]
        idx = torch.clamp(lower_idx.to(torch.int64) + shifts, min=0, max=D-1) # B, num_neighbor, H, W
        select_probs = torch.gather(torch.squeeze(volume_prob, dim=1), 1, idx) # B, num_neighbor, H, W
        sumprobs = torch.sum(select_probs, dim=1, keepdim=True) #([1, 1, 176, 208])
        mask = sumprobs > thresh
        return mask


    def gen_points(self, batch):
        if 'scan' in batch.keys():
            batch.pop('scan')
        data_mvs, pose_ref = self.decode_batch(batch)
        imgs, proj_mats = data_mvs['images'], data_mvs['proj_mats']
        near_fars, depths_h = data_mvs['near_fars'], data_mvs['depths_h'] if 'depths_h' in data_mvs else None
        # print("depths_h", batch["near_fars"], depths_h.shape, depths_h[0,0,:,:])
        # volume_feature:(1, 8, D, 176, 208)       img_feat:(B, V, C, h, w)
        cam_expected_depth = None
        ndc_std_depth = None
        # volume_feature: 1, 8, 128, 176, 208;
        # img_feat: 1, 3, 32, 128, 160;
        # depth_values: 1, 128
        photometric_confidence_lst=[]
        cam_xyz_lst = []
        xyz_color_lst = []
        nearfar_mask_lst = []
        volume_prob = None
        # w2c_ref = batch["w2cs"][:, self.args.ref_vid, ...].transpose(1, 2)
        depth_vid = [0]
        init_view_num=3
        manual_std_depth=0.0
        near_far_depth = batch["near_fars_depth"][0]
        depth_interval, depth_min = (near_far_depth[1] - near_far_depth[0]) / 192., near_far_depth[0]
        depth_values = (depth_min + torch.arange(0, 192, device="cuda", dtype=torch.float32) * depth_interval)[None, :]
        dimgs = batch["mvs_images"] if "mvs_images" in batch else imgs
        bmvs_2d_features=None
        # print("dimgs",dimgs.shape) # [1, 3, 3, 800, 800]
        bimgs = dimgs[:, :init_view_num].expand(len(depth_vid), -1, -1, -1, -1)
        bvid = torch.as_tensor(depth_vid, dtype=torch.long, device="cuda")
        bproj_mats = proj_mats[0, bvid, ...]
        bdepth_values = depth_values.expand(len(depth_vid), -1)

        with torch.no_grad():
            # 1, 128, 160;  1, 128, 160; prob_volume: 1, 192, 128, 160
            depths_h, photometric_confidence, _, _ = self.ReconNet(bimgs, bproj_mats, bdepth_values, features=bmvs_2d_features)
            depths_h, photometric_confidence = depths_h[:,None,...], photometric_confidence[:,None,...]

        bcam_expected_depth = torch.nn.functional.interpolate(depths_h, size=list(dimgs.shape)[-2:], mode='nearest')

        photometric_confidence = torch.nn.functional.interpolate(photometric_confidence, size=list(dimgs.shape)[-2:], mode='nearest')  # 1, 1, H, W
        photometric_confidence_lst = torch.unbind(photometric_confidence[:,None,...], dim=0)
        bndc_std_depth = torch.ones_like(bcam_expected_depth) * manual_std_depth
        # print(bimgs.shape, batch["intrinsics"].shape) # torch.Size([1, 3, 3, 800, 800]) torch.Size([1, 3, 3, 3])
        for i in range(len(depth_vid)):
            vid = depth_vid[i]
            cam_expected_depth, ndc_std_depth = bcam_expected_depth[i:i+1], bndc_std_depth[i:i+1]
            ndc_xyz, cam_xyz, HDWD, nearfar_mask = self.sample_func(volume_prob, self.args, batch["intrinsics"][:, vid,...], near_fars[0, vid], cam_expected_depth=cam_expected_depth, ndc_std_depth=ndc_std_depth)
            if cam_xyz.shape[1] > 0:
                # cam_xyz torch.Size([1, 1, 1, 800, 800, 3])
                # print("imgs", imgs.shape) # ([1, 3, 3, 800, 800])
                xyz_color_lst.append(imgs[0, 0, ...])
                cam_xyz_lst.append(cam_xyz)
                nearfar_mask_lst.append(nearfar_mask)
        return cam_xyz_lst, xyz_color_lst, photometric_confidence_lst, nearfar_mask_lst, HDWD, data_mvs, [batch["intrinsics"][:,int(vid),...] for vid in depth_vid], [batch["w2cs"][:,int(vid),...] for vid in depth_vid]



    def forward(self, batch):
        # 3 , 3, 3, 2, 4, dict, 3, 3

        cam_xyz_lst, _, photometric_confidence_lst, nearfar_mask_lst, HDWD, data_mvs, intrinsics_lst, extrinsics_lst  = self.gen_points(batch)
        # #################### FILTER by Masks ##################
        gpu_filter = True
        if self.args.manual_depth_view != 0:
            # cuda filter
            if gpu_filter:
                cam_xyz_lst, _, photometric_confidence_lst = filter_utils.filter_by_masks_gpu(cam_xyz_lst, intrinsics_lst, extrinsics_lst, photometric_confidence_lst, nearfar_mask_lst, self.args)
            else:
                cam_xyz_lst, _, photometric_confidence_lst = filter_utils.filter_by_masks([cam_xyz.cpu().numpy() for cam_xyz in cam_xyz_lst], [intrinsics.cpu().numpy() for intrinsics in intrinsics_lst], [extrinsics.cpu().numpy() for extrinsics in extrinsics_lst], [confidence.cpu().numpy() for confidence in photometric_confidence_lst], [nearfar_mask.cpu().numpy() for nearfar_mask in nearfar_mask_lst], self.args)
                cam_xyz_lst = [torch.as_tensor(cam_xyz, device="cuda", dtype=torch.float32) for cam_xyz in cam_xyz_lst]
                photometric_confidence_lst = [torch.as_tensor(confidence, device="cuda", dtype=torch.float32) for confidence in photometric_confidence_lst]
        else:
            B, N, C, H, W, _ = cam_xyz_lst[0].shape
            cam_xyz_lst = [cam_xyz.view(C, H, W, 3) for cam_xyz in cam_xyz_lst]
            cam_xyz_lst = [cam_xyz[nearfar_mask[0,...], :] for cam_xyz, nearfar_mask in zip(cam_xyz_lst, nearfar_mask_lst)]
            # print("after filterd", cam_xyz_lst[0].shape)
            photometric_confidence_lst = [torch.ones_like(cam_xyz[...,0]) for cam_xyz in cam_xyz_lst]
        print("get_image_features")
        img_feats = self.get_image_features(batch['images'])

        points_features_lst = [self.query_embedding(HDWD, torch.as_tensor(cam_xyz_lst[i][None, ...], device="cuda", dtype=torch.float32), photometric_confidence_lst[i][None, ..., None], img_feats, data_mvs['c2ws'], data_mvs['w2cs'], batch["intrinsics"], int(self.args.depth_vid[i]), pointdir_w=False) for i in range(len(cam_xyz_lst))]


        # #################### start query embedding ##################
        xyz_ref_lst = [(torch.cat([xyz_cam, torch.ones_like(xyz_cam[..., 0:1])], dim=-1) @ torch.linalg.inv(
            cam_extrinsics[0]).transpose(0, 1) @ batch["w2cs"][0, self.args.ref_vid, ...].transpose(0, 1))[..., :3] for
                       xyz_cam, cam_extrinsics in zip(cam_xyz_lst, extrinsics_lst)]
        ref_xyz = torch.cat(xyz_ref_lst, dim=0)
        points_embedding = torch.cat([points_features[0] for points_features in points_features_lst], dim=1)
        points_colors = torch.cat([points_features[1] for points_features in points_features_lst], dim=1) if points_features_lst[0][1] is not None else None
        points_ref_dirs = torch.cat([points_features[2] for points_features in points_features_lst], dim=1) if points_features_lst[0][2] is not None else None
        points_conf = torch.cat([points_features[3] for points_features in points_features_lst], dim=1) if points_features_lst[0][3] is not None else None

        return ref_xyz, points_embedding, points_colors, points_ref_dirs, points_conf


    def save_points(self, xyz, dir, total_steps):
        if xyz.ndim < 3:
            xyz = xyz[None, ...]
        os.makedirs(dir, exist_ok=True)
        for i in range(xyz.shape[0]):
            if isinstance(total_steps, str):
                filename = 'step-{}-{}.txt'.format(total_steps, i)
            else:
                filename = 'step-{:04d}-{}.txt'.format(total_steps, i)
            filepath = os.path.join(dir, filename)
            np.savetxt(filepath, xyz[i, ...].reshape(-1, xyz.shape[-1]), delimiter=";")

    def save_image(self, img_array, filepath):
        assert len(img_array.shape) == 2 or (len(img_array.shape) == 3
                                             and img_array.shape[2] in [3, 4])

        if img_array.dtype != np.uint8:
            img_array = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        Image.fromarray(img_array).save(filepath)
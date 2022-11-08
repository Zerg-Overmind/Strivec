import configargparse

def config_parser(cmd=None):
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./log',
                        help='where to store ckpts and logs')
    parser.add_argument("--add_timestamp", type=int, default=0,
                        help='add timestamp to dir')
    parser.add_argument("--use_geo", type=int, default=1,
                        help='1 for geo, 0 for not using geo, -1 for use dvgo density estimation ')
    parser.add_argument("--skip_zero_grad", type=int, default=0,
                        help='use masked adam for skip zero')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                        help='input data directory')
    parser.add_argument("--pointfile", type=str, default='./log/den_prefix',
                        help='input data directory')
    parser.add_argument("--pretrained_mvs_ckpt", type=str, default="/home/xharlie/dev/cloud_tensoRF/log/MVSNet/model_000014.ckpt",
                        help='checkpoints of the pretrained_mvs network')
    parser.add_argument("--progress_refresh_rate", type=int, default=10,
                        help='how many iterations to show psnrs or iters')

    parser.add_argument('--with_depth', action='store_true')
    parser.add_argument('--downsample_train', type=float, default=1.0, help="downsample image size")
    parser.add_argument('--downsample_test', type=float, default=1.0, help="downsample image size")
    parser.add_argument('--model_name', type=str, default='PointTensorCP',
                        choices=['PointTensorCP', 'PointTensorCP_hier', 'PointTensorCP_adapt', 'PointTensorCPB' ,
                                 'PointTensorCPD', 'PointTensorVMSplit'])
    parser.add_argument('--gpu_ids',
                        type=str,
                        default='0',
                        help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    # mvs options
    parser.add_argument('--mvs_model', type=str, default='mvs_points',
                        choices=['mvs_points'], help="which mvs model to use for intial geometry reconstruction")
    parser.add_argument("--depth_conf_thresh", type=float, default=0.8,
                        help='thresholds for depth merge')
    parser.add_argument("--geo_cnsst_num", type=int, default=0,
                        help='number views required for passing depth merge threshold')
    parser.add_argument("--inall_img", type=int, default=1,
                        help='force points in all image during alpha masking')
    parser.add_argument("--default_conf", type=float, default=0.15,
                        help='default confidence for point confident')
    parser.add_argument(
        '--ranges',
        type=float,
        nargs='+',
        default=(-100.0, -100.0, -100.0, 100.0, 100.0, 100.0),
        help='spatial ranges of initial geometry, we filter out initial points out side of ranges'
    )
    parser.add_argument("--vox_res", type=int, default=0,
                        help='resolution of voxlization to filter initial points')
    parser.add_argument(
        '--fps_num',
        type=int,
        nargs='+',
        default=None,
        help='filter initial points to fps_num points by fps sampling'
    )
    parser.add_argument("--shp_rand", type=float, default=0, help="randomply sample shading points along each ray")

    # local pointTensor
    parser.add_argument("--tensoRF_shape", type=str, default="cube", choices=['cube', 'sphere'], help="the shape of each tensoRF")

    parser.add_argument(
        '--vox_range',
        type=float,
        nargs='+',
        default=None,
        help='to determine the center of tensoRF, we voxelize the space by vox_range and locate voxel having inital points'
    )
    parser.add_argument(
        '--local_range',
        type=float,
        nargs='+',
        default=(0.1, 0.1, 0.1),
        help='local tensoRF half range'
    )

    parser.add_argument(
        '--dilation_ratio',
        type=float,
        nargs='+',
        default=(1.0, 1.0),
        help='local tensoRF half range'
    )
    parser.add_argument(
        '--local_unit',
        type=float,
        nargs='+',
        default=None,
        help='local unit for each stage'
    )
    parser.add_argument(
        '--local_dims_init',
        type=int,
        nargs='+',
        default=(10, 10, 10),
        help='initial local dimension in each tensoRF'
    )

    parser.add_argument(
        '--local_dims_final',
        type=int,
        nargs='+',
        default=(20, 20, 20),
        help='final local dimension in each tensoRF'
    )
    parser.add_argument(
        '--rot_init',
        type=float,
        nargs='+',
        default=None,
        help='initial local dimension in each tensoRF during rotation'
    )
    parser.add_argument(
        '--rotgrad',
        type=int,
        default=0,
        help='if use gradient to rotate'
    )

    parser.add_argument("--local_dims_trend", type=int, default=None, action="append",
                        help="target tensoRF voxel dims for each upsampling step")

    parser.add_argument(
        '--adapt_local_range',
        type=float,
        nargs='+',
        default=(0.1, 0.1, 0.1),
        help='adaptive local point tensor half range'
    )
    parser.add_argument(
        '--adapt_dims_init',
        type=int,
        nargs='+',
        default=(10, 10, 10),
        help='adaptive tensoRF initial local dimension in each tensoRF'
    )

    parser.add_argument(
        '--adapt_dims_final',
        type=int,
        nargs='+',
        default=(20, 20, 20),
        help='adaptive tensoRF final local dimension in each tensoRF'
    )

    parser.add_argument(
        '--adapt_dims_trend',
        type=int,
        nargs='+',
        default=(20, 20, 20),
        help='target adaptive tensoRF dims for each upsampling step'
    )

    parser.add_argument(
        '--vox_center',
        type=int,
        nargs='+',
        default=(1),
        help='put tensoRF at the center of occupied voxel or centroid of points in the occupied voxel'
    )
    parser.add_argument("--rot_step", type=int, default=None, action="append")
    parser.add_argument("--unit_lvl", type=int, default=0, help='which lvl we take grid unit')
    parser.add_argument("--filterall", type=int, default=0, help='if only compute shading when all lvl tensoRF covers or any lvl tensoRF covers')

    parser.add_argument("--KNN", type=int, default=1, help="if use KNN for sampling")
    parser.add_argument("--rot_KNN", type=int, default=None, help="if use KNN for sampling during rotation optimization")

    parser.add_argument("--margin", type=int, default=0, help="exclude number of pixel on edge for scannet during train, due to cam distortion")
    parser.add_argument("--test_margin", type=int, default=0, help="number of pixel on edge for scannet during test, due to cam distortion")
    parser.add_argument("--radiance_add", type=int, default=0, help="1, add radiance feature; 0, cat radiance feature")
    parser.add_argument("--rad_lvl_norm", type=int, default=0, help="1, normalize radiance by num of valid lvls")
    parser.add_argument("--den_lvl_norm", type=int, default=0, help="1, normalize density by  num of valid lvls")
    parser.add_argument(
        '--max_tensoRF',
        type=int,
        nargs='+',
        default=[4],
        help='max queried nearby tensoRFs for each shading point'
    )
    parser.add_argument(
        '--rot_max_tensoRF',
        type=int,
        nargs='+',
        default=None,
        help='max queried nearby tensoRFs for each shading point during rotation optimization'
    )
    parser.add_argument(
        '--K_tensoRF',
        type=int,
        nargs='+',
        default=None,
        help='among max_tensoRF tensoRF, resample a subset of K_tensoRF tensoRFs'
    )
    parser.add_argument("--rot_K_tensoRF", type=int, default=None, help='during rotation optimization, among rot_max_tensoRF tensoRF, resample a subset of rot K_tensoRF tensoRFs')

    parser.add_argument("--intrp_mthd", type=str, default="linear", choices=['linear', 'avg', 'quadric'], help="how to blend features from each queried tensoRFs")

    # loader options
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--n_iters", type=int, default=30000, help="total training iters")

    parser.add_argument('--dataset_name', type=str, default='blender',
                        choices=['blender', 'llff', 'nsvf', 'dtu','tankstemple', 'TanksAndTempleBG', 'own_data', 'scannet'])
    parser.add_argument('--align_center', type=int, default=1)
    # parser.add_argument('--cal_mthd', type=str, default='drct', choices=['splat', 'drct'])

    # training options
    # learning rate
    parser.add_argument("--lr_geo_init", type=float, default=0.03,
                        help='learning rate of tensorf geo parameter')
    parser.add_argument("--lr_init", type=float, default=0.02,
                        help='inital learning rate of xyz tensoRF features')
    parser.add_argument("--lr_basis", type=float, default=1e-3,
                        help='inital learning rate of network')
    parser.add_argument("--lr_decay_iters", type=int, default=-1,
                        help = 'number of iterations the lr will decay to the target ratio; -1 will set it to n_iters')
    parser.add_argument("--lr_decay_target_ratio", type=float, default=0.1,
                        help='the target decay ratio; after decay_iters inital lr decays to lr*ratio')
    parser.add_argument("--lr_upsample_reset", type=int, default=1,
                        help='reset lr to inital after upsampling')

    # loss
    parser.add_argument("--L1_weight_inital", type=float, default=0.0,
                        help='before shrink scene box, l1 loss weight on tensoRF features')
    parser.add_argument("--L1_weight_rest", type=float, default=0,
                        help='after shrink scene box, l1 loss weight on tensoRF features ')
    parser.add_argument("--Ortho_weight", type=float, default=0.0,
                        help='tensoRF orthorgonal loss weight')
    parser.add_argument("--TV_weight_density", type=float, default=0.0,
                        help='TV density loss weight')
    parser.add_argument("--TV_weight_app", type=float, default=0.0,
                        help='TV appearance loss weight')
    parser.add_argument("--weight_rgbper", type=float, default=0.0,
                        help='rbg per shading points before ray marching, loss weight')
    
    # model
    # volume options
    parser.add_argument("--n_lamb_sigma", type=int, action="append",
                        help='num. density components in each tensoRF')
    parser.add_argument("--n_lamb_sh", type=int, action="append",
                        help='num. radiance color components in each tensoRF')
    parser.add_argument("--data_dim_color", type=int, nargs='+', default=(27),
                        help='num. feature components in each tensoRF')

    parser.add_argument("--alpha_mask_thre", type=float, default=0.0001,
                        help='threshold for creating alpha mask volume')
    parser.add_argument("--distance_scale", type=float, default=25,
                        help='scaling sampling distance for computation in Raw2Alpha')
    parser.add_argument("--density_shift", type=float, default=-10,
                        help='shift density in softplus; making density = 0  when feature == 0')
                        
    # network decoder
    parser.add_argument("--shadingMode", type=str, default="MLP_PE",
                        help='which shading mode to use')
    parser.add_argument("--pos_pe", type=int, default=6,
                        help='number of pe for pos')
    parser.add_argument("--view_pe", type=int, default=6,
                        help='number of pe for view')
    parser.add_argument("--fea_pe", type=int, default=6,
                        help='number of pe for features')
    parser.add_argument("--featureC", type=int, default=128,
                        help='hidden feature channel in MLP')
    


    parser.add_argument("--ckpt", type=str, default=None,
                        help='specific weights npy file to reload')
    parser.add_argument("--info_ckpt", type=str, default=None,
                        help='specific tensorf info pickle file to reload')
    parser.add_argument("--render_only", type=int, default=0)
    parser.add_argument("--render_test", type=int, default=0)
    parser.add_argument("--render_train", type=int, default=0)
    parser.add_argument("--render_path", type=int, default=0)
    parser.add_argument("--export_mesh", type=int, default=0)

    # rendering options
    parser.add_argument('--lindisp', default=False, action="store_true",
                        help='use disparity depth sampling')
    parser.add_argument("--fea2denseAct", type=str, default='softplus', help="activation function in feature2density")
    parser.add_argument('--ray_type', type=int, default=0, help="0 for torch linear, 1 for torch ndc, -1 for cuda linear")
    parser.add_argument('--nSamples', type=int, default=1e6,
                        help='sample point each ray, pass 1e6 if automatic adjust')
    parser.add_argument('--step_ratio',type=float,default=0.5, help="shading point sampling interval ration w.r.t. global voxel edge length")


    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')

    parser.add_argument("--upsamp_list", type=int, action="append", default=None, help="list of iterations to  upsample voxel dims")
    parser.add_argument("--top_rays", type=int, action="append", default=None, help="add new adaptive tensoRF to num. top_rays which has highest rendering loss")
    parser.add_argument("--adapt_list", type=int, action="append" , help="list of iterations to add new adaptive tensoRF at regions with high rendering loss")
    parser.add_argument("--upsamp_reset_list", type=int, default=None, action="append", help="list of iterations to reset lr after upsample the tensoRF dims")
    parser.add_argument("--shrink_list", type=int, default=None, action="append", help="list of iterations to shrink the scene box")

    parser.add_argument("--filter_ray_list", type=int, default=None, action="append", help="list of iterations to filter the ray that has no cross to object")
    parser.add_argument("--update_AlphaMask_list", type=int, action="append", help="list of iterations to update alpha mask to skip computation of shading")
    parser.add_argument("--rmv_unused_list", type=int, action="append", default=None, help="list of iterations to remove tensorfs which r always behind others")
    parser.add_argument("--rmv_unused_ord_thresh", type=int, action="append", default=None, help="ordinal threshold of removing tensorf which r always behind others")
    parser.add_argument('--cluster_method', type=str, action="append",  default=None, help="clustering method, in None, use voxel to cluster")
    parser.add_argument('--boxing_method', type=str, action="append",  default=None, help="bbox method, if None, usc pca")
    parser.add_argument('--cluster_num', type=int, action="append", default=None)
    parser.add_argument('--idx_view', type=int, default=0)
    # logging/saving options
    parser.add_argument("--N_vis", type=int, default=5,
                        help='N images to vis')
    parser.add_argument("--vis_every", type=int, default=10000,
                        help='frequency of visualize the image')

    parser.add_argument("--world_bound_scale", type=float, default=1.0,
                        help='scale up the bbox of scene')

    ########################## args for dvgo initialization ##########################

    parser.add_argument("--pre_num_voxels", type=int, default=1024000, help='N num voxel in dvgo initialization')
    parser.add_argument("--pre_batch_size", type=int, default=8192, help='batch size in dvgo initialization')
    parser.add_argument("--pervoxel_lr", type=int, default=1, help='view-count-based lr')
    parser.add_argument("--pre_maskout_near_cam_vox", type=int, default=1, help='maskout grid points that between cameras and their near planes')
    parser.add_argument("--pre_N_iters", type=int, default=5000, help='in pre den, number of optimization steps')
    parser.add_argument("--pre_alpha_init", type=float, default=1e-6, help='set the alpha values everywhere at the begin of training')
    parser.add_argument("--pre_lrate_decay", type=int, default=20, help='lr decay by 0.1 after every lrate_decay*1000 steps')
    parser.add_argument("--pre_lrate_density", type=float, default=1e-1, help='lr of density voxel grid')
    parser.add_argument("--pre_lrate_k0", type=float, default=1e-1, help='lr of color/feature voxel grid')
    parser.add_argument("--decay_after_scale", type=float, default=1.0, help='decay act_shift after scaling')
    parser.add_argument("--pre_weight_entropy_last", type=float, default=0.01, help='decay act_shift after scaling')
    parser.add_argument("--pre_weight_rgbper", type=float, default=0.1, help='weight of per-point rgb loss')
    parser.add_argument("--pre_maskout_lt_nviews", type=int, default=0, help='N images to vis')
    parser.add_argument(
        '--pre_pg_scale',
        type=int,
        nargs='+',
        default=[],
        help='steps for progressive scaling'
    )
    if cmd is not None:
        return parser.parse_args(cmd)
    else:
        return parser.parse_args()


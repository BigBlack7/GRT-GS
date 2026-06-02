#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#


from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            default_value = value
            t = type(default_value)
            value = default_value if not fill_none else None 
            if shorthand:
                if t == bool:
                    if default_value is True:
                        group.add_argument("--" + key, ("-" + key[0:1]), dest=key, action="store_true")
                        group.add_argument("--no_" + key, dest=key, action="store_false")
                        group.set_defaults(**{key: value})
                    else:
                        group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                elif t == list: # #
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, nargs="+")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    if default_value is True:
                        group.add_argument("--" + key, dest=key, action="store_true")
                        group.add_argument("--no_" + key, dest=key, action="store_false")
                        group.set_defaults(**{key: value})
                    else:
                        group.add_argument("--" + key, default=value, action="store_true")
                elif t == list: # #
                    group.add_argument("--" + key, default=value, nargs="+")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        # Rendering Settings
        self.sh_degree = 3
        self._resolution = -1
        self._white_background = False
        self.render_items = ['RGB', 'Alpha', 'Normal', 'Depth', 'Edge', 'Curvature']
        self.use_ncif = True
        
        # Paths
        self._source_path = ""
        self._model_path = ""
        self._images = "images"

        # Device Settings
        self.data_device = "cuda"
        self.eval = False
        self.lod = 0
        self.llffhold = 8
        self.undistorted = False
        self.val_hold = 0
        self.val_offset = 0
        self.val_max = 8

        # EnvLight Settings
        self.envmap_max_res = 128
        self.envmap_max_roughness = 0.5
        self.envmap_min_roughness = 0.08
        self.relight = False

        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        group = super().extract(args)
        group.source_path = os.path.abspath(group.source_path)
        return group


class PipelineParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        # Processing Settings
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.use_asg = False

        # Debugging
        self.depth_ratio = 0.0
        self.debug = False

        super().__init__(parser, "Pipeline Parameters", sentinel)


class OptimizationParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        # Learning Rate Settings
        self.iterations = 50_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.features_lr = 0.0075 
        self.indirect_lr = 0.0075 
        self.asg_lr = 0.0075 
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001


        self.ori_color_lr = 0.0075 
        self.refl_strength_lr =  0.005 
        self.roughness_lr =  0.005 
        self.metalness_lr = 0.01
        self.normal_lr = 0.006

        self.envmap_cubemap_lr = 0.01
        self.ncif_lr = 0.002
        self.probe_lr = 0.002
        self.prt_lr = 0.001
        self.prt_occlusion_lr = 0.001
        
        # Densification Settings
        self.percent_dense = 0.01

        # Regularization Parameters
        self.lambda_dssim = 0.2
        self.lambda_dist = 0.0
        self.lambda_normal_render_depth = 0.05
        self.lambda_normal_smooth = 0.0
        self.lambda_depth_smooth = 0.0
        self.lambda_ncif_smooth = 0.005
        self.lambda_ncif_magnitude = 0.001
        self.lambda_env_tv = 0.0001
        self.lambda_env_energy = 0.0001
        self.lambda_probe_smooth = 0.003
        self.lambda_probe_energy = 0.0001
        self.lambda_probe_magnitude = 0.001
        self.lambda_prt_smooth = 0.001
        self.lambda_prt_energy = 0.0001
        self.lambda_prt_magnitude = 0.001
        self.lambda_prt_occlusion = 0.0001


        # initial values
        self.init_roughness_value = 0.1
        self.init_refl_value = 0.01
        self.init_refl_value_vol = 0.01
        self.init_metalness_value = 0.5
        self.rough_msk_thr = 0.01
        self.refl_msk_thr = 0.02
        self.refl_msk_thr_vol = 0.02

        self.enlarge_scale = 1.5


        # Opacity and Densify Settings
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 25000 

        # # Extra settings
        self.densify_grad_threshold = 0.0002
        self.prune_opacity_threshold = 0.05



        # Stage Settings
        self.initial = 0
        self.init_until_iter = 0 
        self.volume_render_until_iter = 18000 
        self.normal_smooth_from_iter = 0
        self.normal_smooth_until_iter = 18000
        self.ncif_from_iter = 3000
        self.ncif_tau = 0.15
        self.ncif_ramp_iters = 5000
        self.ncif_diffuse_mu = 1.0
        self.ncif_diffuse_nu = 1.0
        self.use_probe_gi = False
        self.probe_from_iter = 5000
        self.probe_tau = 0.12
        self.probe_ramp_iters = 5000
        self.probe_grid_res = 8
        self.probe_sh_degree = 2
        self.probe_diffuse_mu = 1.0
        self.probe_diffuse_nu = 1.0
        self.use_prt_gs = False
        self.prt_from_iter = 5000
        self.prt_tau = 0.12
        self.prt_ramp_iters = 5000
        self.prt_sh_degree = 2
        self.prt_diffuse_mu = 1.0
        self.prt_diffuse_nu = 1.0
        self.prt_occlusion_init = 0.75
        self.use_r2if = True
        self.r2if_specular_alpha = 1.0
        self.r2if_specular_beta = 1.0
        self.r2if_min_specular_gate = 0.02
        self.r2if_gate_render = False
        self.use_oaf = False
        self.oaf_from_iter = 18000
        self.oaf_tau = 0.25
        self.oaf_ramp_iters = 4000
        self.oaf_power = 1.0
        self.oaf_max_blend = 0.35
        self.use_pcc = True
        self.pcc_keep_ratio = 0.75
        self.use_adaptive_pcc = False
        self.pcc_min_keep = 0.45
        self.pcc_max_keep = 0.85
        self.pcc_opacity_weight = 0.7
        self.pcc_specular_weight = 0.3
        self.pcc_confidence_gamma = 1.0
        self.use_cgi = True
        self.cgi_ramp_iters = 5000
                
        self.indirect = 0
        self.indirect_from_iter =  20000 

        self.feature_rest_from_iter = 5_000
        self.normal_prop_until_iter = 25_000 

        self.normal_prop_interval = 1000
        self.opac_lr0_interval = 200
        self.densification_interval_when_prop = 500



        self.normal_loss_start = 0
        self.dist_loss_start = 3000

        # Environmental Scoping
        self.use_env_scope = False
        self.env_scope_center = [0., 0., 0.]
        self.env_scope_radius = 0.0
        
        # SRGB Transformation
        self.srgb = False

        # mesh
        self.voxel_size = -1.0
        self.depth_trunc = -1.0
        self.sdf_trunc = -1.0
        self.mesh_res = 512
        self.num_cluster = 1


        super().__init__(parser, "Optimization Parameters", sentinel)

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)

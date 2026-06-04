import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import torch.nn.functional as F
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from cubemapencoder import CubemapEncoder
from scene.light import EnvLight
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud, init_predefined_omega
from utils.general_utils import strip_symmetric, build_scaling_rotation, safe_normalize, flip_align_view
from utils.refl_utils import sample_camera_rays, get_env_rayd1, get_env_rayd2
import raytracing


def get_env_direction1(H, W):
    gy, gx = torch.meshgrid(torch.linspace(0.0 + 1.0 / H, 1.0 - 1.0 / H, H, device='cuda'), 
                            torch.linspace(-1.0 + 1.0 / W, 1.0 - 1.0 / W, W, device='cuda'),
                            indexing='ij')
    sintheta, costheta = torch.sin(gy*np.pi), torch.cos(gy*np.pi)
    sinphi, cosphi = torch.sin(gx*np.pi), torch.cos(gx*np.pi)
    env_directions = torch.stack((
        sintheta*sinphi, 
        costheta, 
        -sintheta*cosphi
        ), dim=-1)
    return env_directions


def get_env_direction2(H, W):
    gx, gy = torch.meshgrid(
        torch.linspace(-torch.pi, torch.pi, W, device='cuda'),
        torch.linspace(0, torch.pi, H, device='cuda'),
        indexing='xy'
    )
    env_directions = torch.stack((
        torch.sin(gy)*torch.cos(gx), 
        torch.sin(gy)*torch.sin(gx), 
        torch.cos(gy)
    ), dim=-1)
    return env_directions


class GaussianModel:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
            RS = build_scaling_rotation(torch.cat([scaling * scaling_modifier, torch.ones_like(scaling)], dim=-1), rotation).permute(0,2,1)
            trans = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device="cuda")
            trans[:,:3,:3] = RS
            trans[:, 3,:3] = center
            trans[:, 3, 3] = 1
            return trans
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.refl_activation = torch.sigmoid
        self.inverse_refl_activation = inverse_sigmoid

        self.metalness_ativation = torch.sigmoid
        self.inverse_metalness_activation = inverse_sigmoid

        self.roughness_activation = torch.sigmoid
        self.inverse_roughness_activation = inverse_sigmoid

        self.color_activation = torch.sigmoid
        self.inverse_color_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize
        self.asg_param = init_predefined_omega(4, 8)


    def __init__(self, sh_degree : int, use_ncif: bool = False):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self.use_ncif = use_ncif
        self._xyz = torch.empty(0)
        self._refl_strength = torch.empty(0) 
        self._ori_color = torch.empty(0) 
        self._diffuse_color = torch.empty(0) 
        self._metalness = torch.empty(0) 
        self._roughness = torch.empty(0) 
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._indirect_dc = torch.empty(0)
        self._indirect_rest = torch.empty(0)
        self._indirect_asg = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)

        self._normal1 = torch.empty(0)
        self._normal2 = torch.empty(0)
        self._ncif_dir = torch.empty(0)
        self._probe_grid = torch.empty(0)
        self._probe_aabb_min = torch.empty(0)
        self._probe_aabb_max = torch.empty(0)
        self.probe_grid_res = 0
        self.probe_sh_degree = 0
        self._prt_lighting = torch.empty(0)
        self._prt_occlusion = torch.empty(0)
        self.prt_sh_degree = 0
        self._grt_transfer = torch.empty(0)
        self.grt_sh_degree = 0

        self.optimizer = None
        self.free_radius = 0    
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.init_refl_value = 0.01
        self.init_roughness_value = 0.1 #[0,1]
        self.init_metalness_value = 0.5 #[0,1]
        self.init_ori_color = 0  
        self.enlarge_scale = 1.5
        self.refl_msk_thr = 0.02
        self.rough_msk_thr = 0.1

        self.env_map = None
        self.env_map_2 = None
        self.env_H, self.env_W = 256, 512
        self.env_directions1 = get_env_direction1(self.env_H, self.env_W)
        self.env_directions2 = get_env_direction2(self.env_H, self.env_W)
        self.ray_tracer = None
        self.grt_transfer_bvh_initialized = False
        self.grt_transfer_bvh_iteration = -1
        self.setup_functions()

    def capture(self):
        return (
            "physnorm_grt_v2",
            self.active_sh_degree,
            self._xyz,
            self._refl_strength, 
            self._metalness, 
            self._roughness, 
            self._ori_color, 
            self._diffuse_color, 
            self._features_dc,
            self._features_rest,
            self._indirect_dc,
            self._indirect_rest,
            self._indirect_asg,
            self._scaling,
            self._rotation,
            self._opacity,
            self._normal1,  
            self._normal2,  
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self._ncif_dir,
            self._probe_grid,
            self._probe_aabb_min,
            self._probe_aabb_max,
            self.probe_grid_res,
            self.probe_sh_degree,
            self._prt_lighting,
            self._prt_occlusion,
            self.prt_sh_degree,
            self._grt_transfer,
            self.grt_sh_degree,
            self.grt_transfer_bvh_initialized,
            self.grt_transfer_bvh_iteration,
        )
    
    def restore(self, model_args, training_args):
        if len(model_args) > 0 and model_args[0] in {"physnorm_grt_v1", "physnorm_grt_v2", "physnorm_prt_v1"}:
            (_version,
            self.active_sh_degree,
            self._xyz,
            self._refl_strength,
            self._metalness,
            self._roughness,
            self._ori_color,
            self._diffuse_color,
            self._features_dc,
            self._features_rest,
            self._indirect_dc,
            self._indirect_rest,
            self._indirect_asg,
            self._scaling,
            self._rotation,
            self._opacity,
            self._normal1,
            self._normal2,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
            self._ncif_dir,
            self._probe_grid,
            self._probe_aabb_min,
            self._probe_aabb_max,
            self.probe_grid_res,
            self.probe_sh_degree,
            self._prt_lighting,
            self._prt_occlusion,
            self.prt_sh_degree,
            *grt_state) = model_args
            if len(grt_state) >= 2:
                self._grt_transfer = grt_state[0]
                self.grt_sh_degree = grt_state[1]
                self.grt_transfer_bvh_initialized = bool(grt_state[2]) if len(grt_state) >= 3 else False
                self.grt_transfer_bvh_iteration = int(grt_state[3]) if len(grt_state) >= 4 else -1
            else:
                self._grt_transfer = torch.empty((self._xyz.shape[0], 0), device=self._xyz.device)
                self.grt_sh_degree = 0
                self.grt_transfer_bvh_initialized = False
                self.grt_transfer_bvh_iteration = -1
        elif len(model_args) > 0 and model_args[0] == "physnorm_probe_v1":
            (_version,
            self.active_sh_degree,
            self._xyz,
            self._refl_strength,
            self._metalness,
            self._roughness,
            self._ori_color,
            self._diffuse_color,
            self._features_dc,
            self._features_rest,
            self._indirect_dc,
            self._indirect_rest,
            self._indirect_asg,
            self._scaling,
            self._rotation,
            self._opacity,
            self._normal1,
            self._normal2,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
            self._ncif_dir,
            self._probe_grid,
            self._probe_aabb_min,
            self._probe_aabb_max,
            self.probe_grid_res,
            self.probe_sh_degree) = model_args
        elif len(model_args) > 0 and model_args[0] == "physnorm_no_metal_v1":
            (_version,
            self.active_sh_degree,
            self._xyz,
            self._refl_strength,
            self._roughness,
            self._ori_color,
            self._diffuse_color,
            self._features_dc,
            self._features_rest,
            self._indirect_dc,
            self._indirect_rest,
            self._indirect_asg,
            self._scaling,
            self._rotation,
            self._opacity,
            self._normal1,
            self._normal2,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
            self._ncif_dir) = model_args
            metalness = torch.ones_like(self._refl_strength, device="cuda") * self.init_metalness_value
            self._metalness = nn.Parameter(self.inverse_metalness_activation(metalness).requires_grad_(True))
        elif len(model_args) >= 26:
            (self.active_sh_degree,
            self._xyz,
            self._refl_strength,
            self._metalness,
            self._roughness,
            self._ori_color,
            self._diffuse_color,
            self._features_dc,
            self._features_rest,
            self._indirect_dc,
            self._indirect_rest,
            self._indirect_asg,
            self._scaling,
            self._rotation,
            self._opacity,
            self._normal1,
            self._normal2,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
            _legacy_unused_feat,
            self._ncif_dir,
            *_legacy_states) = model_args
        elif len(model_args) == 23:
            (self.active_sh_degree,
            self._xyz,
            self._refl_strength,
            self._metalness,
            self._roughness,
            self._ori_color,
            self._diffuse_color,
            self._features_dc,
            self._features_rest,
            self._indirect_dc,
            self._indirect_rest,
            self._indirect_asg,
            self._scaling,
            self._rotation,
            self._opacity,
            self._normal1,
            self._normal2,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
            self._ncif_dir,
            *_legacy_states) = model_args
        elif len(model_args) >= 25:
            (self.active_sh_degree,
            self._xyz,
            self._refl_strength,
            self._metalness,
            self._roughness,
            self._ori_color,
            self._diffuse_color,
            self._features_dc,
            self._features_rest,
            self._indirect_dc,
            self._indirect_rest,
            self._indirect_asg,
            self._scaling,
            self._rotation,
            self._opacity,
            self._normal1,
            self._normal2,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
            _legacy_unused_feat,
            *_legacy_states) = model_args
            self._ncif_dir = torch.zeros((self._xyz.shape[0], 3), device="cuda")
        else:
            (self.active_sh_degree,
            self._xyz,
            self._refl_strength,
            self._metalness,
            self._roughness,
            self._ori_color,
            self._diffuse_color,
            self._features_dc,
            self._features_rest,
            self._indirect_dc,
            self._indirect_rest,
            self._indirect_asg,
            self._scaling,
            self._rotation,
            self._opacity,
            self._normal1,
            self._normal2,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale) = model_args
            self._ncif_dir = torch.zeros((self._xyz.shape[0], 3), device="cuda")
        self._indirect_asg = nn.Parameter(torch.zeros(self._rotation.shape[0], 32, 5, device='cuda').requires_grad_(True))
        if not isinstance(self._ncif_dir, nn.Parameter):
            self._ncif_dir = nn.Parameter(self._ncif_dir.to("cuda").requires_grad_(True))
        if self._probe_grid.numel() > 0 and not isinstance(self._probe_grid, nn.Parameter):
            self._probe_grid = nn.Parameter(self._probe_grid.to("cuda").requires_grad_(True))
            self._probe_aabb_min = self._probe_aabb_min.to("cuda")
            self._probe_aabb_max = self._probe_aabb_max.to("cuda")
        if self._prt_lighting.numel() > 0 and not isinstance(self._prt_lighting, nn.Parameter):
            self._prt_lighting = nn.Parameter(self._prt_lighting.to("cuda").requires_grad_(True))
        if self._prt_occlusion.numel() > 0 and not isinstance(self._prt_occlusion, nn.Parameter):
            self._prt_occlusion = nn.Parameter(self._prt_occlusion.to("cuda").requires_grad_(True))
        if self._grt_transfer.numel() == 0 and self._grt_transfer.dim() <= 1:
            self._grt_transfer = torch.empty((self._xyz.shape[0], 0), device="cuda")
            self.grt_sh_degree = 0
        elif self._grt_transfer.numel() > 0 and not isinstance(self._grt_transfer, nn.Parameter):
            self._grt_transfer = nn.Parameter(self._grt_transfer.to("cuda").requires_grad_(True))
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        # self.optimizer.load_state_dict(opt_dict)

    def set_opacity_lr(self, lr):   
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "opacity":
                param_group['lr'] = lr

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling) 
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property   
    def get_refl(self): 
        return self.refl_activation(self._refl_strength)

    @property
    def get_metalness(self):
        return self.metalness_ativation(self._metalness)

    @property
    def get_rough(self): 
        return self.roughness_activation(self._roughness)

    @property
    def get_ori_color(self): 
        return self.color_activation(self._ori_color)
    
    @property
    def get_diffuse_color(self): 
        return self.color_activation(self._diffuse_color)

    def get_ncif_dir(self):
        if not self.use_ncif or self._ncif_dir.numel() == 0:
            return None
        return self._ncif_dir

    def _probe_num_coeffs(self, degree):
        return (int(degree) + 1) ** 2

    def _init_probe_field(self, training_args):
        if not (getattr(training_args, "use_probe_gi", False) or getattr(training_args, "use_grt", False)):
            return
        if self._probe_grid.numel() > 0:
            if not isinstance(self._probe_grid, nn.Parameter):
                self._probe_grid = nn.Parameter(self._probe_grid.to("cuda").requires_grad_(True))
            return

        res = max(2, int(getattr(training_args, "probe_grid_res", 8)))
        degree_name = "grt_sh_degree" if getattr(training_args, "use_grt", False) else "probe_sh_degree"
        degree = max(0, min(2, int(getattr(training_args, degree_name, 2))))
        coeffs = self._probe_num_coeffs(degree)
        with torch.no_grad():
            xyz = self.get_xyz.detach()
            aabb_min = xyz.amin(dim=0)
            aabb_max = xyz.amax(dim=0)
            extent = (aabb_max - aabb_min).clamp_min(1e-4)
            padding = extent.max() * 0.05
            self._probe_aabb_min = (aabb_min - padding).to("cuda")
            self._probe_aabb_max = (aabb_max + padding).to("cuda")

        grid = torch.zeros((res, res, res, coeffs, 3), dtype=torch.float, device="cuda")
        self._probe_grid = nn.Parameter(grid.requires_grad_(True))
        self.probe_grid_res = res
        self.probe_sh_degree = degree

    def has_probe_field(self):
        return self._probe_grid.numel() > 0 and self._probe_aabb_min.numel() == 3 and self._probe_aabb_max.numel() == 3

    def _probe_sh_basis(self, normals, degree):
        normals = F.normalize(normals, dim=-1, eps=1e-6)
        x, y, z = normals.unbind(dim=-1)
        basis = [torch.ones_like(x) * 0.28209479177387814]
        if degree >= 1:
            basis.extend([
                -0.4886025119029199 * y,
                0.4886025119029199 * z,
                -0.4886025119029199 * x,
            ])
        if degree >= 2:
            basis.extend([
                1.0925484305920792 * x * y,
                -1.0925484305920792 * y * z,
                0.31539156525252005 * (3.0 * z * z - 1.0),
                -1.0925484305920792 * x * z,
                0.5462742152960396 * (x * x - y * y),
            ])
        return torch.stack(basis, dim=-1)

    def _trilinear_probe_coeffs(self, xyz):
        res = int(self.probe_grid_res)
        grid = self._probe_grid
        denom = (self._probe_aabb_max - self._probe_aabb_min).clamp_min(1e-6)
        coord = ((xyz - self._probe_aabb_min) / denom).clamp(0.0, 1.0) * (res - 1)
        idx0 = torch.floor(coord).long().clamp(0, res - 1)
        idx1 = (idx0 + 1).clamp(0, res - 1)
        frac = (coord - idx0.float()).view(-1, 3, 1, 1)

        x0, y0, z0 = idx0[:, 0], idx0[:, 1], idx0[:, 2]
        x1, y1, z1 = idx1[:, 0], idx1[:, 1], idx1[:, 2]
        wx, wy, wz = frac[:, 0], frac[:, 1], frac[:, 2]

        c000 = grid[x0, y0, z0]
        c100 = grid[x1, y0, z0]
        c010 = grid[x0, y1, z0]
        c110 = grid[x1, y1, z0]
        c001 = grid[x0, y0, z1]
        c101 = grid[x1, y0, z1]
        c011 = grid[x0, y1, z1]
        c111 = grid[x1, y1, z1]

        c00 = c000 * (1.0 - wx) + c100 * wx
        c10 = c010 * (1.0 - wx) + c110 * wx
        c01 = c001 * (1.0 - wx) + c101 * wx
        c11 = c011 * (1.0 - wx) + c111 * wx
        c0 = c00 * (1.0 - wy) + c10 * wy
        c1 = c01 * (1.0 - wy) + c11 * wy
        return c0 * (1.0 - wz) + c1 * wz

    def get_probe_irradiance(self, xyz, normals, opt=None):
        if not self.has_probe_field():
            return None
        degree = int(self.probe_sh_degree)
        coeffs = self._trilinear_probe_coeffs(xyz)
        basis = self._probe_sh_basis(normals, degree)
        return (coeffs * basis[..., None]).sum(dim=1)

    def _grt_num_coeffs(self, degree):
        return (int(degree) + 1) ** 2

    def _init_grt_transfer(self, training_args):
        if not getattr(training_args, "use_grt", False):
            return
        if self._grt_transfer.numel() > 0:
            if not isinstance(self._grt_transfer, nn.Parameter):
                self._grt_transfer = nn.Parameter(self._grt_transfer.to("cuda").requires_grad_(True))
            return

        degree = max(0, min(2, int(getattr(training_args, "grt_sh_degree", 2))))
        coeffs = self._grt_num_coeffs(degree)
        transfer = torch.zeros((self.get_xyz.shape[0], coeffs), dtype=torch.float, device="cuda")
        transfer[:, 0] = float(getattr(training_args, "grt_transfer_init", 1.0))
        self._grt_transfer = nn.Parameter(transfer.requires_grad_(True))
        self.grt_sh_degree = degree

    def _grt_geometry_normals(self):
        splat2world = self.get_covariance(1.0)
        return safe_normalize(splat2world[:, 2, :3])

    def _grt_local_hemisphere_directions(self, n_rays, device):
        idx = torch.arange(n_rays, dtype=torch.float32, device=device)
        z = (idx + 0.5) / float(n_rays)
        r = torch.sqrt((1.0 - z * z).clamp_min(0.0))
        phi = idx * (np.pi * (3.0 - np.sqrt(5.0)))
        return torch.stack((r * torch.cos(phi), r * torch.sin(phi), z), dim=-1)

    def _grt_world_hemisphere_directions(self, normals, local_dirs):
        up_z = torch.zeros_like(normals)
        up_z[:, 2] = 1.0
        up_y = torch.zeros_like(normals)
        up_y[:, 1] = 1.0
        up = torch.where(normals[:, 2:3].abs() < 0.9, up_z, up_y)
        tangent = safe_normalize(torch.cross(up, normals, dim=-1))
        bitangent = safe_normalize(torch.cross(normals, tangent, dim=-1))
        dirs = (
            local_dirs[None, :, 0:1] * tangent[:, None, :]
            + local_dirs[None, :, 1:2] * bitangent[:, None, :]
            + local_dirs[None, :, 2:3] * normals[:, None, :]
        )
        return safe_normalize(dirs)

    def _assign_grt_transfer(self, transfer):
        transfer = transfer.to(self.get_xyz.device).float()
        old_param = self._grt_transfer if isinstance(self._grt_transfer, nn.Parameter) else None
        if isinstance(self._grt_transfer, nn.Parameter) and self._grt_transfer.shape == transfer.shape:
            self._grt_transfer.data.copy_(transfer)
        else:
            self._grt_transfer = nn.Parameter(transfer.requires_grad_(True))
            if self.optimizer is not None:
                for group in self.optimizer.param_groups:
                    if group.get("name") == "grt_transfer":
                        group["params"] = [self._grt_transfer]
                        break
        if isinstance(self._grt_transfer, nn.Parameter) and self._grt_transfer.grad is not None:
            self._grt_transfer.grad.zero_()
        if old_param is not None and old_param is not self._grt_transfer and old_param.grad is not None:
            old_param.grad = None

    @torch.no_grad()
    def initialize_grt_transfer_from_bvh(self, training_args, iteration=None):
        if not getattr(training_args, "use_grt", False):
            return None
        if not getattr(training_args, "use_grt_visibility_init", True):
            return None
        if self.ray_tracer is None or self.get_xyz.numel() == 0:
            return None

        iteration = int(iteration) if iteration is not None else -1
        grt_from = int(getattr(training_args, "grt_from_iter", 0))
        if iteration >= 0 and iteration < grt_from:
            return None

        refresh_interval = int(getattr(training_args, "grt_transfer_refresh_interval", 0))
        if self.grt_transfer_bvh_initialized:
            if refresh_interval <= 0:
                return None
            if iteration >= 0 and (iteration - self.grt_transfer_bvh_iteration) < refresh_interval:
                return None

        if self._grt_transfer.numel() == 0 or not isinstance(self._grt_transfer, nn.Parameter):
            self._init_grt_transfer(training_args)
        if self._grt_transfer.numel() == 0:
            return None

        degree = max(0, min(2, int(getattr(training_args, "grt_sh_degree", self.grt_sh_degree))))
        coeffs = self._grt_num_coeffs(degree)
        n_rays = max(coeffs + 4, int(getattr(training_args, "grt_visibility_rays", 64)))
        chunk = max(1, int(getattr(training_args, "grt_visibility_chunk", 4096)))
        eps = max(1e-5, float(getattr(training_args, "grt_visibility_eps", 0.02)))
        blend = min(1.0, max(0.0, float(getattr(training_args, "grt_transfer_blend", 1.0))))
        clamp_value = float(getattr(training_args, "grt_transfer_clamp", 4.0))

        xyz = self.get_xyz.detach()
        normals = self._grt_geometry_normals().detach()
        extent = (xyz.amax(dim=0) - xyz.amin(dim=0)).norm().item()
        max_distance = float(getattr(training_args, "grt_visibility_max_distance", 0.0))
        if max_distance <= 0.0:
            max_distance = max(extent * 1.5, 1.0)

        local_dirs = self._grt_local_hemisphere_directions(n_rays, xyz.device)
        fitted = torch.empty((xyz.shape[0], coeffs), dtype=torch.float32, device=xyz.device)
        visibility_means = []

        for start in range(0, xyz.shape[0], chunk):
            end = min(start + chunk, xyz.shape[0])
            chunk_xyz = xyz[start:end]
            chunk_normals = normals[start:end]
            dirs = self._grt_world_hemisphere_directions(chunk_normals, local_dirs)
            origins = (chunk_xyz[:, None, :] + chunk_normals[:, None, :] * eps).expand_as(dirs).contiguous()

            _, _, depth = self.ray_tracer.trace(
                origins.reshape(-1, 3),
                dirs.reshape(-1, 3),
            )
            depth = depth.reshape(end - start, n_rays)
            hit = torch.isfinite(depth) & (depth > eps) & (depth < max_distance)
            visible = (~hit).float()
            visibility_means.append(visible.mean())

            basis = self._probe_sh_basis(dirs.reshape(-1, 3), degree).reshape(end - start, n_rays, coeffs)
            rhs = visible.unsqueeze(-1)
            try:
                solution = torch.linalg.lstsq(basis, rhs).solution.squeeze(-1)
            except RuntimeError:
                solution = torch.matmul(torch.linalg.pinv(basis), rhs).squeeze(-1)
            solution = torch.nan_to_num(solution, nan=0.0, posinf=0.0, neginf=0.0)
            if clamp_value > 0.0:
                solution = solution.clamp(-clamp_value, clamp_value)
            fitted[start:end] = solution

        current = self._grt_transfer.detach()
        if current.shape[0] == fitted.shape[0] and current.shape[1] >= coeffs and blend < 1.0:
            target = current[:, :coeffs] * (1.0 - blend) + fitted * blend
        else:
            target = fitted
        if current.shape[0] == fitted.shape[0] and current.shape[1] > coeffs:
            target = torch.cat([target, current[:, coeffs:]], dim=1)

        self._assign_grt_transfer(target)
        self.grt_sh_degree = degree
        self.grt_transfer_bvh_initialized = True
        self.grt_transfer_bvh_iteration = iteration

        mean_visibility = torch.stack(visibility_means).mean().item() if visibility_means else 0.0
        return {
            "iteration": iteration,
            "gaussians": int(xyz.shape[0]),
            "rays": int(n_rays),
            "coeffs": int(coeffs),
            "mean_visibility": float(mean_visibility),
        }

    def has_grt_field(self):
        return self.has_probe_field() and self._grt_transfer.numel() > 0

    @property
    def get_grt_transfer(self):
        return self._grt_transfer

    def get_grt_indirect(self, xyz, normals, reflection, opt=None):
        if not self.has_grt_field():
            return None, None, None, None
        degree = min(int(self.grt_sh_degree), int(self.probe_sh_degree))
        coeff_count = self._grt_num_coeffs(degree)
        coeffs = self._trilinear_probe_coeffs(xyz)[:, :coeff_count]
        transfer = self.get_grt_transfer[:, :coeff_count]
        probe_dot = (coeffs * transfer[..., None]).sum(dim=1)

        refl_basis = self._probe_sh_basis(reflection, degree)
        probe_dir = (coeffs * refl_basis[..., None]).sum(dim=1)
        transfer_vis = torch.sigmoid((transfer * refl_basis).sum(dim=1, keepdim=True))
        directional = probe_dir * transfer_vis

        mode = str(getattr(opt, "grt_mode", "dot")).lower() if opt is not None else "dot"
        if mode == "directional":
            indirect = directional
        elif mode == "hybrid":
            indirect = 0.5 * probe_dot + 0.5 * directional
        else:
            indirect = probe_dot
        return indirect.clamp_min(0.0), probe_dir.clamp_min(0.0), transfer_vis, probe_dot

    def probe_smoothness_loss(self):
        if not self.has_probe_field():
            return torch.zeros((), device="cuda")
        grid = self._probe_grid
        loss = (grid[1:, :, :, :, :] - grid[:-1, :, :, :, :]).abs().mean()
        loss = loss + (grid[:, 1:, :, :, :] - grid[:, :-1, :, :, :]).abs().mean()
        loss = loss + (grid[:, :, 1:, :, :] - grid[:, :, :-1, :, :]).abs().mean()
        return loss / 3.0

    def probe_energy_loss(self):
        if not self.has_probe_field():
            return torch.zeros((), device="cuda")
        return self._probe_grid.pow(2).mean()

    def grt_transfer_loss(self):
        if self._grt_transfer.numel() == 0:
            return torch.zeros((), device="cuda")
        if self._grt_transfer.shape[1] <= 1:
            return self._grt_transfer.pow(2).mean()
        return self._grt_transfer[:, 1:].pow(2).mean()

    def _prt_num_coeffs(self, degree):
        return (int(degree) + 1) ** 2

    def _init_prt_field(self, training_args):
        if not getattr(training_args, "use_prt_gs", False):
            return
        if self._prt_occlusion.numel() == 0 and self.get_xyz.numel() > 0:
            init_occ = float(getattr(training_args, "prt_occlusion_init", 0.75))
            init_occ = max(1e-4, min(1.0 - 1e-4, init_occ))
            occ = torch.ones((self.get_xyz.shape[0], 1), device="cuda") * init_occ
            self._prt_occlusion = nn.Parameter(self.inverse_opacity_activation(occ).requires_grad_(True))
        elif self._prt_occlusion.numel() > 0 and not isinstance(self._prt_occlusion, nn.Parameter):
            self._prt_occlusion = nn.Parameter(self._prt_occlusion.to("cuda").requires_grad_(True))

        if self._prt_lighting.numel() > 0:
            if not isinstance(self._prt_lighting, nn.Parameter):
                self._prt_lighting = nn.Parameter(self._prt_lighting.to("cuda").requires_grad_(True))
            return

        degree = max(0, min(2, int(getattr(training_args, "prt_sh_degree", 2))))
        coeffs = self._prt_num_coeffs(degree)
        lighting = torch.zeros((coeffs, 3), dtype=torch.float, device="cuda")
        self._prt_lighting = nn.Parameter(lighting.requires_grad_(True))
        self.prt_sh_degree = degree

    def has_prt_field(self):
        return self._prt_lighting.numel() > 0 and self._prt_occlusion.numel() > 0

    @property
    def get_prt_occlusion(self):
        if self._prt_occlusion.numel() == 0:
            return None
        return torch.sigmoid(self._prt_occlusion)

    def get_prt_irradiance(self, normals, opt=None):
        if not self.has_prt_field():
            return None
        degree = int(self.prt_sh_degree)
        basis = self._probe_sh_basis(normals, degree)
        lighting = (basis[..., None] * self._prt_lighting[None]).sum(dim=1)
        occlusion = self.get_prt_occlusion
        return lighting * occlusion

    def prt_energy_loss(self):
        if not self.has_prt_field():
            return torch.zeros((), device="cuda")
        return self._prt_lighting.pow(2).mean()

    def prt_occlusion_loss(self):
        if self._prt_occlusion.numel() == 0:
            return torch.zeros((), device="cuda")
        return (1.0 - self.get_prt_occlusion).abs().mean()
    

    def get_normal(self, scaling_modifier, dir_pp_normalized, return_delta=False): 
        splat2world = self.get_covariance(scaling_modifier)
        normals_raw = splat2world[:,2,:3] 
        normals_raw, positive = flip_align_view(normals_raw, dir_pp_normalized)

        if return_delta:
            delta_normal1 = self._normal1 
            delta_normal2 = self._normal2 
            delta_normal = torch.stack([delta_normal1, delta_normal2], dim=-1) 
            idx = torch.where(positive, 0, 1).long()[:,None,:].repeat(1, 3, 1) 
            delta_normal = torch.gather(delta_normal, index=idx, dim=-1).squeeze(-1) 
            normals = delta_normal + normals_raw
            normals = safe_normalize(normals) 
            return normals, delta_normal
        else:
            normals = safe_normalize(normals_raw)
            return normals

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_indirect(self):
        indirect_dc = self._indirect_dc
        indirect_rest = self._indirect_rest
        return torch.cat((indirect_dc, indirect_rest), dim=1)
    
    @property
    def get_asg(self):
        return self._indirect_asg
    
    def render_env_map(self, H=512):
        if H == self.env_H:
            directions1 = self.env_directions1
            directions2 = self.env_directions2
        else:
            W = H * 2
            directions1 = get_env_direction1(H, W)
            directions2 = get_env_direction2(H, W)
        return {'env1': self.env_map(directions1, mode="pure_env"), 'env2': self.env_map(directions2, mode="pure_env")}
    
    def render_env_map_2(self, H=512):
        if H == self.env_H:
            directions1 = self.env_directions1
            directions2 = self.env_directions2
        else:
            W = H * 2
            directions1 = get_env_direction1(H, W)
            directions2 = get_env_direction2(H, W)
        return {'env1': self.env_map_2(directions1, mode="pure_env"), 'env2': self.env_map_2(directions2, mode="pure_env")}

    @property   
    def get_envmap(self): 
        return self.env_map
    
    @property   
    def get_envmap_2(self): 
        return self.env_map_2
    
    @property   
    def get_refl_strength_to_total(self):
        refl = self.get_refl
        return (refl>0.1).sum() / refl.shape[0]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_xyz, self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, args):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        sh_features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        sh_features[:, :3, 0 ] = fused_color
        sh_features[:, 3:, 1:] = 0.0
        sh_indirect = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        asg_indirect = torch.zeros((fused_color.shape[0], 5, 32)).float().cuda()

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 2)
        rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        refl = self.inverse_refl_activation(torch.ones_like(opacities).cuda() * self.init_refl_value)
        refl_strength = refl.cuda()

        metalness = self.inverse_metalness_activation(torch.ones_like(opacities).cuda() * self.init_metalness_value)
        metalness = metalness.cuda()

        roughness = self.inverse_roughness_activation(torch.ones_like(opacities).cuda() * self.init_roughness_value)
        roughness = roughness.cuda()

        def initialize_ori_color(point_cloud, init_color= 0.5, noise_level=0.05):
            base_color = torch.full((point_cloud.shape[0], 3), init_color, dtype=torch.float, device="cuda")
            noise = (torch.rand(point_cloud.shape[0], 3, dtype=torch.float, device="cuda") - 0.5) * noise_level
            ori_color = base_color + noise
            ori_color = torch.clamp(ori_color, 0.0, 1.0)
            return ori_color
        
        ori_color = self.inverse_color_activation(initialize_ori_color(fused_point_cloud))
        diffuse_color = self.inverse_color_activation(initialize_ori_color(fused_point_cloud))  # Initialize diffuse_color similarly

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))

        self._refl_strength = nn.Parameter(refl_strength.requires_grad_(True))  
        self._ori_color = nn.Parameter(ori_color.requires_grad_(True)) 
        self._diffuse_color = nn.Parameter(diffuse_color.requires_grad_(True))  # Initialize _diffuse_color
        self._roughness = nn.Parameter(roughness.requires_grad_(True)) 
        self._metalness = nn.Parameter(metalness.requires_grad_(True)) 
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._features_dc = nn.Parameter(sh_features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(sh_features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._indirect_dc = nn.Parameter(sh_indirect[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._indirect_rest = nn.Parameter(sh_indirect[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._indirect_asg = nn.Parameter(asg_indirect.transpose(1, 2).contiguous().requires_grad_(True))
        
        normals1 = np.zeros_like(np.asarray(pcd.points, dtype=np.float32))
        normals2 = np.copy(normals1)
        self._normal1 = nn.Parameter(torch.from_numpy(normals1).to(self._xyz.device).requires_grad_(True))
        self._normal2 = nn.Parameter(torch.from_numpy(normals2).to(self._xyz.device).requires_grad_(True))
        self._ncif_dir = torch.empty((self._xyz.shape[0], 0), device="cuda")
        self._prt_occlusion = torch.empty((self._xyz.shape[0], 0), device="cuda")
        self._grt_transfer = torch.empty((self._xyz.shape[0], 0), device="cuda")

        self.env_map = EnvLight(path=None, device='cuda', max_res=args.envmap_max_res, min_roughness=args.envmap_min_roughness, max_roughness=args.envmap_max_roughness, trainable=True).cuda()
        self.env_map_2 = EnvLight(path=None, device='cuda', max_res=args.envmap_max_res, min_roughness=args.envmap_min_roughness, max_roughness=args.envmap_max_roughness, trainable=True).cuda()

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self._init_probe_field(training_args)
        self._init_grt_transfer(training_args)

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.features_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.features_lr / 20.0, "name": "f_rest"},
            
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': self.env_map.parameters(), 'lr': training_args.envmap_cubemap_lr, "name": "env"},     
            {'params': self.env_map_2.parameters(), 'lr': training_args.envmap_cubemap_lr, "name": "env2"}     
        ]

        self._normal1.requires_grad_(requires_grad=False)
        self._normal2.requires_grad_(requires_grad=False)
        l.extend([
            {'params': [self._refl_strength], 'lr': training_args.refl_strength_lr, "name": "refl_strength"},  
            {'params': [self._ori_color], 'lr': training_args.ori_color_lr, "name": "ori_color"},  
            {'params': [self._diffuse_color], 'lr': training_args.ori_color_lr, "name": "diffuse_color"},  
            {'params': [self._roughness], 'lr': training_args.roughness_lr, "name": "roughness"},  
            {'params': [self._metalness], 'lr': training_args.metalness_lr, "name": "metalness"},  
            {'params': [self._normal1], 'lr': training_args.normal_lr, "name": "normal1"},
            {'params': [self._normal2], 'lr': training_args.normal_lr, "name": "normal2"},
            {'params': [self._indirect_dc], 'lr': training_args.indirect_lr, "name": "ind_dc"},
            {'params': [self._indirect_rest], 'lr': training_args.indirect_lr / 20.0, "name": "ind_rest"},
            {'params': [self._indirect_asg], 'lr': training_args.asg_lr, "name": "ind_asg"},
        ])
        if self.has_probe_field():
            probe_lr = training_args.probe_lr if (getattr(training_args, "use_probe_gi", False) or getattr(training_args, "use_grt", False)) else 0.0
            l.append({'params': [self._probe_grid], 'lr': probe_lr, "name": "probe_grid"})
        if self._grt_transfer.numel() > 0:
            grt_lr = training_args.grt_transfer_lr if getattr(training_args, "use_grt", False) else 0.0
            l.append({'params': [self._grt_transfer], 'lr': grt_lr, "name": "grt_transfer"})

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz','nx2', 'ny2', 'nz2']
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        for i in range(self._indirect_dc.shape[1]*self._indirect_dc.shape[2]):
            l.append('ind_dc_{}'.format(i))
        for i in range(self._indirect_rest.shape[1]*self._indirect_rest.shape[2]):
            l.append('ind_rest_{}'.format(i))
        for i in range(self._indirect_asg.shape[1]*self._indirect_asg.shape[2]):
            l.append('ind_asg_{}'.format(i))
        l.append('opacity')
        l.append('refl_strength') 
        l.append('metalness') 
        l.append('roughness') 
        for i in range(self._ori_color.shape[1]):
            l.append('ori_color_{}'.format(i))
        for i in range(self._diffuse_color.shape[1]):  # Add diffuse_color attributes
            l.append('diffuse_color_{}'.format(i))
        for i in range(self._ncif_dir.shape[1]):
            l.append('ncif_dir_{}'.format(i))
        for i in range(self._prt_occlusion.shape[1]):
            l.append('prt_occlusion_{}'.format(i))
        for i in range(self._grt_transfer.shape[1]):
            l.append('grt_transfer_{}'.format(i))

        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        ind_dc = self._indirect_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        ind_rest = self._indirect_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        ind_asg = self._indirect_asg.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()

        refl_strength = self._refl_strength.detach().cpu().numpy()    
        metalness = self._metalness.detach().cpu().numpy()    
        roughness = self._roughness.detach().cpu().numpy()    
        ori_color = self._ori_color.detach().cpu().numpy()    
        diffuse_color = self._diffuse_color.detach().cpu().numpy()  
        ncif_dir = self._ncif_dir.detach().cpu().numpy()
        prt_occlusion = self._prt_occlusion.detach().cpu().numpy()
        grt_transfer = self._grt_transfer.detach().cpu().numpy() if self._grt_transfer.numel() > 0 else np.zeros((xyz.shape[0], 0))
        
        normals1 = self._normal1.detach().cpu().numpy()
        normals2 = self._normal2.detach().cpu().numpy() 

        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)

        attributes = np.concatenate((xyz, normals1, normals2, f_dc, f_rest, ind_dc, ind_rest, ind_asg, opacities, refl_strength, metalness, roughness, ori_color, diffuse_color, ncif_dir, prt_occlusion, grt_transfer, scale, rotation), axis=1)

        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        
        if self.env_map is not None:
            save_path = path.replace('.ply', '1.map')
            torch.save(self.env_map.state_dict(), save_path)

        if self.env_map_2 is not None:
            save_path = path.replace('.ply', '2.map')
            torch.save(self.env_map_2.state_dict(), save_path)

        if self.has_probe_field():
            save_path = path.replace('.ply', '.probe')
            torch.save(
                {
                    "probe_grid": self._probe_grid.detach().cpu(),
                    "probe_aabb_min": self._probe_aabb_min.detach().cpu(),
                    "probe_aabb_max": self._probe_aabb_max.detach().cpu(),
                    "probe_grid_res": int(self.probe_grid_res),
                    "probe_sh_degree": int(self.probe_sh_degree),
                },
                save_path,
            )

        if self.has_prt_field():
            save_path = path.replace('.ply', '.prt')
            torch.save(
                {
                    "prt_lighting": self._prt_lighting.detach().cpu(),
                    "prt_sh_degree": int(self.prt_sh_degree),
                },
                save_path,
            )

    def reset_opacity0(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def reset_opacity1(self, exclusive_msk = None):
        RESET_V = 0.9
        opacity_old = self.get_opacity
        o_msk = (opacity_old > RESET_V).flatten()
        if exclusive_msk is not None:
            o_msk = torch.logical_or(o_msk, exclusive_msk)
        opacities_new = torch.ones_like(opacity_old)*inverse_sigmoid(torch.tensor([RESET_V]).cuda())
        opacities_new[o_msk] = self._opacity[o_msk]
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        if "opacity" not in optimizable_tensors: return
        self._opacity = optimizable_tensors["opacity"]

    def reset_opacity1_strategy2(self):
        RESET_B = 1.5
        opacity_old = self.get_opacity
        opacities_new = inverse_sigmoid((opacity_old*RESET_B).clamp(0,0.99))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        if "opacity" not in optimizable_tensors: return
        self._opacity = optimizable_tensors["opacity"]


    def reset_refl(self, exclusive_msk = None):
        refl_new = inverse_sigmoid(torch.max(self.get_refl, torch.ones_like(self.get_refl)*self.init_refl_value))
        if exclusive_msk is not None:
            refl_new[exclusive_msk] = self._refl_strength[exclusive_msk]
        optimizable_tensors = self.replace_tensor_to_optimizer(refl_new, "refl_strength")
        if "refl_strength" not in optimizable_tensors: return
        self._refl_strength = optimizable_tensors["refl_strength"]


    def dist_rot(self): 
        REFL_MSK_THR = self.refl_msk_thr
        refl_msk = self.get_refl.flatten() > REFL_MSK_THR
        rot = self.get_rotation.clone()
        dist_rot = self.rotation_activation(rot + torch.randn_like(rot)*0.08)
        dist_rot[refl_msk] = rot[refl_msk]
        optimizable_tensors = self.replace_tensor_to_optimizer(dist_rot, "rotation")
        if "rotation" not in optimizable_tensors: return
        self._rotation = optimizable_tensors["rotation"]

    def dist_albedo(self, exclusive_msk = None):
        DIST_RANGE = 0.4
        refl_msk = self.get_refl.flatten() > self.refl_msk_thr
        if exclusive_msk is not None:
            refl_msk = torch.logical_or(refl_msk, exclusive_msk)
        dcc = self._ori_color.clone()
        dist_dcc = dcc + (torch.rand_like(dcc)*DIST_RANGE*2-DIST_RANGE) 
        dist_dcc[refl_msk] = dcc[refl_msk]
        optimizable_tensors = self.replace_tensor_to_optimizer(dist_dcc, "ori_color")
        if "ori_color" not in optimizable_tensors: return
        self._ori_color = optimizable_tensors["ori_color"]

    def dist_color(self, exclusive_msk = None):
        DIST_RANGE = 0.4
        refl_msk = self.get_refl.flatten() > self.refl_msk_thr
        if exclusive_msk is not None:
            refl_msk = torch.logical_or(refl_msk, exclusive_msk)
        dcc = self._features_dc.clone()
        dist_dcc = dcc + (torch.rand_like(dcc)*DIST_RANGE*2-DIST_RANGE) 
        dist_dcc[refl_msk] = dcc[refl_msk]
        optimizable_tensors = self.replace_tensor_to_optimizer(dist_dcc, "f_dc")
        if "f_dc" not in optimizable_tensors: return
        self._features_dc = optimizable_tensors["f_dc"]

    def enlarge_refl_scales(self, ret_raw=True, ENLARGE_SCALE=1.5, REFL_MSK_THR=0.02, ROUGH_MSK_THR=0.1, exclusive_msk=None):
        ENLARGE_SCALE = self.enlarge_scale
        REFL_MSK_THR = self.refl_msk_thr
        ROUGH_MSK_THR = self.rough_msk_thr

        refl_msk = self.get_refl.flatten() < REFL_MSK_THR
        rough_msk = self.get_rough.flatten() > ROUGH_MSK_THR
        combined_msk = torch.logical_or(refl_msk, rough_msk)
        if exclusive_msk is not None:
            combined_msk = torch.logical_or(combined_msk, exclusive_msk) 
        scales = self.get_scaling
        rmin_axis = (torch.ones_like(scales) * ENLARGE_SCALE)
        if ret_raw:
            scale_new = self.scaling_inverse_activation(scales * rmin_axis)
            scale_new[combined_msk] = self._scaling[combined_msk]
        else:
            scale_new = scales * rmin_axis
            scale_new[combined_msk] = scales[combined_msk]   
        return scale_new

    def reset_scale(self, exclusive_msk = None):
        scale_new = self.enlarge_refl_scales(ret_raw=True, exclusive_msk=exclusive_msk)
        optimizable_tensors = self.replace_tensor_to_optimizer(scale_new, "scaling")
        if "scaling" not in optimizable_tensors: return
        self._scaling = optimizable_tensors["scaling"]


    def reset_features(self, reset_value_dc=0.0, reset_value_rest=0.0):
        # 重置 features_dc
        features_dc_new = torch.full_like(self._features_dc, reset_value_dc, dtype=torch.float, device="cuda")
        # 重置 features_rest
        features_rest_new = torch.full_like(self._features_rest, reset_value_rest, dtype=torch.float, device="cuda")

        # 将新的features_dc和features_rest替换到优化器中
        optimizable_tensors = self.replace_tensor_to_optimizer(features_dc_new, "f_dc")
        optimizable_tensors.update(self.replace_tensor_to_optimizer(features_rest_new, "f_rest"))
        # 更新active_sh_degree
        self.active_sh_degree = 0

        # 更新类中的属性
        if "f_dc" in optimizable_tensors:
            self._features_dc = optimizable_tensors["f_dc"]
        if "f_rest" in optimizable_tensors:
            self._features_rest = optimizable_tensors["f_rest"]


    def reset_ori_color(self, reset_value=0.5, noise_level=0.05):
        base_color = torch.full_like(self._ori_color, reset_value, dtype=torch.float, device="cuda")
        noise = (torch.rand_like(base_color, dtype=torch.float, device="cuda") - 0.5) * noise_level
        ori_color_new = base_color + noise
        ori_color_new = torch.clamp(ori_color_new, 0.0, 1.0)
        
        # 将重置后的 ori_color 更新到优化器中
        optimizable_tensors = self.replace_tensor_to_optimizer(self.inverse_color_activation(ori_color_new), "ori_color")
        if "ori_color" in optimizable_tensors:
            self._ori_color = optimizable_tensors["ori_color"]

    def reset_diffuse_color(self, reset_value=0.5, noise_level=0.05):
        base_color = torch.full_like(self._diffuse_color, reset_value, dtype=torch.float, device="cuda")
        noise = (torch.rand_like(base_color, dtype=torch.float, device="cuda") - 0.5) * noise_level
        diffuse_color_new = torch.clamp(base_color + noise, 0.0, 1.0)

        optimizable_tensors = self.replace_tensor_to_optimizer(
            self.inverse_color_activation(diffuse_color_new), "diffuse_color")
        if "diffuse_color" in optimizable_tensors:
            self._diffuse_color = optimizable_tensors["diffuse_color"]

    def reset_refl_strength(self, reset_value=0.01):
        refl_strength_new = torch.full_like(self._refl_strength, reset_value, dtype=torch.float, device="cuda")
        optimizable_tensors = self.replace_tensor_to_optimizer(self.inverse_refl_activation(refl_strength_new), "refl_strength")
        if "refl_strength" in optimizable_tensors:
            self._refl_strength = optimizable_tensors["refl_strength"]

    def reset_metalness(self, reset_value=0.5):
        metalness_new = torch.full_like(self._metalness, reset_value, dtype=torch.float, device="cuda")
        optimizable_tensors = self.replace_tensor_to_optimizer(
            self.inverse_metalness_activation(metalness_new), "metalness")
        if "metalness" in optimizable_tensors:
            self._metalness = optimizable_tensors["metalness"]

    def reset_roughness(self, reset_value=0.1):
        roughness_new = torch.full_like(self._roughness, reset_value, dtype=torch.float, device="cuda")
        optimizable_tensors = self.replace_tensor_to_optimizer(self.inverse_roughness_activation(roughness_new), "roughness")
        if "roughness" in optimizable_tensors:
            self._roughness = optimizable_tensors["roughness"]

    def _blend01(self, current_value, reset_value, keep_ratio):
        target = torch.full_like(current_value, reset_value, dtype=torch.float, device="cuda")
        if torch.is_tensor(keep_ratio):
            keep_ratio = keep_ratio.to(device=current_value.device, dtype=current_value.dtype)
            while keep_ratio.dim() < current_value.dim():
                keep_ratio = keep_ratio.unsqueeze(-1)
        return torch.clamp(current_value * keep_ratio + target * (1.0 - keep_ratio), 1e-4, 1.0 - 1e-4)

    def soft_reset_ori_color(self, reset_value=0.5, keep_ratio=0.75):
        color_new = self._blend01(self.get_ori_color, reset_value, keep_ratio)
        optimizable_tensors = self.replace_tensor_to_optimizer(self.inverse_color_activation(color_new), "ori_color")
        if "ori_color" in optimizable_tensors:
            self._ori_color = optimizable_tensors["ori_color"]

    def soft_reset_diffuse_color(self, reset_value=0.5, keep_ratio=0.75):
        color_new = self._blend01(self.get_diffuse_color, reset_value, keep_ratio)
        optimizable_tensors = self.replace_tensor_to_optimizer(
            self.inverse_color_activation(color_new), "diffuse_color")
        if "diffuse_color" in optimizable_tensors:
            self._diffuse_color = optimizable_tensors["diffuse_color"]

    def soft_reset_refl_strength(self, reset_value=0.01, keep_ratio=0.75):
        refl_new = self._blend01(self.get_refl, reset_value, keep_ratio)
        optimizable_tensors = self.replace_tensor_to_optimizer(self.inverse_refl_activation(refl_new), "refl_strength")
        if "refl_strength" in optimizable_tensors:
            self._refl_strength = optimizable_tensors["refl_strength"]

    def soft_reset_metalness(self, reset_value=0.5, keep_ratio=0.75):
        metalness_new = self._blend01(self.get_metalness, reset_value, keep_ratio)
        optimizable_tensors = self.replace_tensor_to_optimizer(
            self.inverse_metalness_activation(metalness_new), "metalness")
        if "metalness" in optimizable_tensors:
            self._metalness = optimizable_tensors["metalness"]

    def soft_reset_roughness(self, reset_value=0.1, keep_ratio=0.75):
        roughness_new = self._blend01(self.get_rough, reset_value, keep_ratio)
        optimizable_tensors = self.replace_tensor_to_optimizer(self.inverse_roughness_activation(roughness_new), "roughness")
        if "roughness" in optimizable_tensors:
            self._roughness = optimizable_tensors["roughness"]


    def load_ply(self, path, relight=False, args=None):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        # # 
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        refl_strength = np.asarray(plydata.elements[0]["refl_strength"])[..., np.newaxis] # #

        ori_color = np.stack((np.asarray(plydata.elements[0]['ori_color_0']),
                              np.asarray(plydata.elements[0]['ori_color_1']),
                              np.asarray(plydata.elements[0]['ori_color_2'])),  axis=1)
        diffuse_color = np.stack((np.asarray(plydata.elements[0]['diffuse_color_0']),
                                np.asarray(plydata.elements[0]['diffuse_color_1']),
                                np.asarray(plydata.elements[0]['diffuse_color_2'])),  axis=1)
        ncif_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("ncif_dir_")]
        ncif_names = sorted(ncif_names, key=lambda x: int(x.split('_')[-1]))
        if ncif_names:
            ncif_dir = np.zeros((xyz.shape[0], len(ncif_names)))
            for idx, attr_name in enumerate(ncif_names):
                ncif_dir[:, idx] = np.asarray(plydata.elements[0][attr_name])
        else:
            ncif_dir = np.zeros((xyz.shape[0], 0))
        prt_occ_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("prt_occlusion_")]
        prt_occ_names = sorted(prt_occ_names, key=lambda x: int(x.split('_')[-1]))
        if prt_occ_names:
            prt_occlusion = np.zeros((xyz.shape[0], len(prt_occ_names)))
            for idx, attr_name in enumerate(prt_occ_names):
                prt_occlusion[:, idx] = np.asarray(plydata.elements[0][attr_name])
        else:
            prt_occlusion = np.zeros((xyz.shape[0], 0))
        grt_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("grt_transfer_")]
        grt_names = sorted(grt_names, key=lambda x: int(x.split('_')[-1]))
        if grt_names:
            grt_transfer = np.zeros((xyz.shape[0], len(grt_names)))
            for idx, attr_name in enumerate(grt_names):
                grt_transfer[:, idx] = np.asarray(plydata.elements[0][attr_name])
        else:
            grt_transfer = np.zeros((xyz.shape[0], 0))
        
        roughness = np.asarray(plydata.elements[0]["roughness"])[..., np.newaxis] # #
        metalness = np.asarray(plydata.elements[0]["metalness"])[..., np.newaxis] # #

        normal1 = np.stack((np.asarray(plydata.elements[0]["nx"]),
                        np.asarray(plydata.elements[0]["ny"]),
                        np.asarray(plydata.elements[0]["nz"])),  axis=1)
        normal2 = np.stack((np.asarray(plydata.elements[0]["nx2"]),
                        np.asarray(plydata.elements[0]["ny2"]),
                        np.asarray(plydata.elements[0]["nz2"])),  axis=1)


        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))
        self.active_sh_degree = self.max_sh_degree
        
        indirect_dc = np.zeros((xyz.shape[0], 3, 1))
        indirect_dc[:, 0, 0] = np.asarray(plydata.elements[0]["ind_dc_0"])
        indirect_dc[:, 1, 0] = np.asarray(plydata.elements[0]["ind_dc_1"])
        indirect_dc[:, 2, 0] = np.asarray(plydata.elements[0]["ind_dc_2"])

        extra_ind_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("ind_rest_")]
        extra_ind_names = sorted(extra_ind_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_ind_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        indirect_extra = np.zeros((xyz.shape[0], len(extra_ind_names)))
        for idx, attr_name in enumerate(extra_ind_names):
            indirect_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        indirect_extra = indirect_extra.reshape((indirect_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        extra_asg_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("ind_asg_")]
        extra_asg_names = sorted(extra_asg_names, key = lambda x: int(x.split('_')[-1]))
        indirect_asg = np.zeros((xyz.shape[0], len(extra_asg_names)))
        for idx, attr_name in enumerate(extra_asg_names):
            indirect_asg[:, idx] = np.asarray(plydata.elements[0][attr_name])
        indirect_asg = indirect_asg.reshape((indirect_asg.shape[0], 5, -1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # #
        if not relight:
            map_path1 = path.replace('.ply', '1.map')
            map_path2 = path.replace('.ply', '2.map')
            if os.path.exists(map_path1)  and os.path.exists(map_path2):
                # self.env_map = CubemapEncoder(output_dim=3, resolution=128).cuda()
                self.env_map = EnvLight(path=None, device='cuda',  max_res=args.envmap_max_res, min_roughness=args.envmap_min_roughness, max_roughness=args.envmap_max_roughness, trainable=True).cuda()
                self.env_map.load_state_dict(torch.load(map_path1))
                self.env_map.build_mips()
                self.env_map_2 = EnvLight(path=None, device='cuda',  max_res=args.envmap_max_res, min_roughness=args.envmap_min_roughness, max_roughness=args.envmap_max_roughness, trainable=True).cuda()
                self.env_map_2.load_state_dict(torch.load(map_path2))
                self.env_map_2.build_mips()
        else:
            map_path = path.replace('.ply', '.hdr')
            self.env_map = EnvLight(path=map_path, device='cuda', trainable=True).cuda()

        probe_path = path.replace('.ply', '.probe')
        if os.path.exists(probe_path):
            probe_state = torch.load(probe_path, map_location="cuda")
            self._probe_grid = nn.Parameter(probe_state["probe_grid"].to("cuda").float().requires_grad_(True))
            self._probe_aabb_min = probe_state["probe_aabb_min"].to("cuda").float()
            self._probe_aabb_max = probe_state["probe_aabb_max"].to("cuda").float()
            self.probe_grid_res = int(probe_state.get("probe_grid_res", self._probe_grid.shape[0]))
            self.probe_sh_degree = int(probe_state.get("probe_sh_degree", int(round(self._probe_grid.shape[-2] ** 0.5)) - 1))
        else:
            self._probe_grid = torch.empty(0, device="cuda")
            self._probe_aabb_min = torch.empty(0, device="cuda")
            self._probe_aabb_max = torch.empty(0, device="cuda")
            self.probe_grid_res = 0
            self.probe_sh_degree = 0

        prt_path = path.replace('.ply', '.prt')
        if os.path.exists(prt_path):
            prt_state = torch.load(prt_path, map_location="cuda")
            self._prt_lighting = nn.Parameter(prt_state["prt_lighting"].to("cuda").float().requires_grad_(True))
            self.prt_sh_degree = int(prt_state.get("prt_sh_degree", int(round(self._prt_lighting.shape[0] ** 0.5)) - 1))
        else:
            self._prt_lighting = torch.empty(0, device="cuda")
            self.prt_sh_degree = 0

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))

        self._refl_strength = nn.Parameter(torch.tensor(refl_strength, dtype=torch.float, device="cuda").requires_grad_(True))   # #
        self._metalness = nn.Parameter(torch.tensor(metalness, dtype=torch.float, device="cuda").requires_grad_(True))   # #
        self._roughness = nn.Parameter(torch.tensor(roughness, dtype=torch.float, device="cuda").requires_grad_(True))   # #
        self._ori_color = nn.Parameter(torch.tensor(ori_color, dtype=torch.float, device="cuda").requires_grad_(True))   # #
        self._diffuse_color = nn.Parameter(torch.tensor(diffuse_color, dtype=torch.float, device="cuda").requires_grad_(True))   # #
        self._ncif_dir = nn.Parameter(torch.tensor(ncif_dir, dtype=torch.float, device="cuda").requires_grad_(True))
        self._prt_occlusion = nn.Parameter(torch.tensor(prt_occlusion, dtype=torch.float, device="cuda").requires_grad_(True))
        self._grt_transfer = nn.Parameter(torch.tensor(grt_transfer, dtype=torch.float, device="cuda").requires_grad_(True)) if grt_transfer.shape[1] > 0 else torch.empty((xyz.shape[0], 0), device="cuda")
        self.grt_sh_degree = int(round(grt_transfer.shape[1] ** 0.5)) - 1 if grt_transfer.shape[1] > 0 else 0

        self._normal1 = nn.Parameter(torch.tensor(normal1, dtype=torch.float, device="cuda").requires_grad_(True))       # #
        self._normal2 = nn.Parameter(torch.tensor(normal2, dtype=torch.float, device="cuda").requires_grad_(True))       # #

        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        
        self._indirect_dc = nn.Parameter(torch.tensor(indirect_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._indirect_rest = nn.Parameter(torch.tensor(indirect_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._indirect_asg = nn.Parameter(torch.tensor(indirect_asg, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is None: continue
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in {"mlp", "env", "env2", "probe_grid", "prt_lighting"}:
                continue   # #
            stored_state = self.optimizer.state.get(group['params'][0], None)

            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]

        self._refl_strength = optimizable_tensors['refl_strength']    # #
        self._ori_color = optimizable_tensors['ori_color']    # #
        self._diffuse_color = optimizable_tensors['diffuse_color']    # #
        self._roughness = optimizable_tensors['roughness']    # #
        self._metalness = optimizable_tensors['metalness']    # #
        self._normal1 = optimizable_tensors["normal1"]        # #
        self._normal2 = optimizable_tensors["normal2"]        # #
        if "ncif_dir" in optimizable_tensors:
            self._ncif_dir = optimizable_tensors["ncif_dir"]
        else:
            self._ncif_dir = torch.empty((self._xyz.shape[0], 0), device="cuda")
        if "prt_occlusion" in optimizable_tensors:
            self._prt_occlusion = optimizable_tensors["prt_occlusion"]
        else:
            self._prt_occlusion = torch.empty((self._xyz.shape[0], 0), device="cuda")
        if "grt_transfer" in optimizable_tensors:
            self._grt_transfer = optimizable_tensors["grt_transfer"]
        else:
            self._grt_transfer = torch.empty((self._xyz.shape[0], 0), device="cuda")

        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._indirect_dc = optimizable_tensors["ind_dc"]
        self._indirect_rest = optimizable_tensors["ind_rest"]
        self._indirect_asg = optimizable_tensors["ind_asg"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in {"mlp", "env", "env2", "probe_grid", "prt_lighting"}:
                continue   # #
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_refl_strength, new_metalness, new_roughness, new_ori_color, new_diffuse_color, new_ncif_dir, new_prt_occlusion, new_grt_transfer, new_features_dc, new_features_rest, new_indirect_dc, new_indirect_asg, new_indirect_rest, new_opacities, new_scaling, new_rotation, new_normal1, new_normal2):
        d = {"xyz": new_xyz,
             
        "refl_strength": new_refl_strength,    # #
        "metalness": new_metalness,    # #
        "roughness": new_roughness,    # #
        "ori_color": new_ori_color,    # #
        "diffuse_color": new_diffuse_color,    # #
        "ncif_dir": new_ncif_dir,
        "prt_occlusion": new_prt_occlusion,
        "grt_transfer": new_grt_transfer,
        "normal1" : new_normal1,       # #
        "normal2" : new_normal2,       # #

        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        
        "ind_dc": new_indirect_dc,
        "ind_rest": new_indirect_rest,
        "ind_asg": new_indirect_asg,
        
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]

        self._refl_strength = optimizable_tensors['refl_strength']    # #
        self._metalness = optimizable_tensors['metalness']    # #
        self._roughness = optimizable_tensors['roughness']    # #
        self._ori_color = optimizable_tensors['ori_color']    # #
        self._diffuse_color = optimizable_tensors['diffuse_color']    # #
        if "ncif_dir" in optimizable_tensors:
            self._ncif_dir = optimizable_tensors["ncif_dir"]
        else:
            self._ncif_dir = torch.empty((self._xyz.shape[0], 0), device="cuda")
        if "prt_occlusion" in optimizable_tensors:
            self._prt_occlusion = optimizable_tensors["prt_occlusion"]
        else:
            self._prt_occlusion = torch.empty((self._xyz.shape[0], 0), device="cuda")
        if "grt_transfer" in optimizable_tensors:
            self._grt_transfer = optimizable_tensors["grt_transfer"]
        else:
            self._grt_transfer = torch.empty((self._xyz.shape[0], 0), device="cuda")
        self._normal1 = optimizable_tensors["normal1"]        # #
        self._normal2 = optimizable_tensors["normal2"]        # #

        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        
        self._indirect_dc = optimizable_tensors["ind_dc"]
        self._indirect_rest = optimizable_tensors["ind_rest"]
        self._indirect_asg = optimizable_tensors["ind_asg"]
        
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        stds = torch.cat([stds, 0 * torch.ones_like(stds[:,:1])], dim=-1)
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_refl_strength = self._refl_strength[selected_pts_mask].repeat(N,1)   # #
        new_ori_color = self._ori_color[selected_pts_mask].repeat(N,1)   # #
        new_diffuse_color = self._diffuse_color[selected_pts_mask].repeat(N,1)   # #
        new_ncif_dir = self._ncif_dir[selected_pts_mask].repeat(N,1)
        new_prt_occlusion = self._prt_occlusion[selected_pts_mask].repeat(N,1)
        new_grt_transfer = self._grt_transfer[selected_pts_mask].repeat(N,1)
        new_roughness = self._roughness[selected_pts_mask].repeat(N,1)   # #
        new_metalness = self._metalness[selected_pts_mask].repeat(N,1)   # #
        new_normal1 = self._normal1[selected_pts_mask].repeat(N,1)        # #
        new_normal2 = self._normal2[selected_pts_mask].repeat(N,1)       # #

        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        
        new_indirect_dc = self._indirect_dc[selected_pts_mask].repeat(N,1,1)
        new_indirect_rest = self._indirect_rest[selected_pts_mask].repeat(N,1,1)
        new_indirect_asg = self._indirect_asg[selected_pts_mask].repeat(N,1,1)
        
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_refl_strength, new_metalness, new_roughness, new_ori_color, new_diffuse_color, new_ncif_dir, new_prt_occlusion, new_grt_transfer, new_features_dc, new_features_rest, new_indirect_dc, new_indirect_asg, new_indirect_rest, new_opacity, new_scaling, new_rotation, new_normal1, new_normal2)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]

        new_refl_strength = self._refl_strength[selected_pts_mask]   # #
        new_metalness = self._metalness[selected_pts_mask]   # #
        new_roughness = self._roughness[selected_pts_mask]   # #
        new_ori_color = self._ori_color[selected_pts_mask]   # #
        new_diffuse_color = self._diffuse_color[selected_pts_mask]   # #
        new_ncif_dir = self._ncif_dir[selected_pts_mask]
        new_prt_occlusion = self._prt_occlusion[selected_pts_mask]
        new_grt_transfer = self._grt_transfer[selected_pts_mask]
        new_normal1 = self._normal1[selected_pts_mask]       # #
        new_normal2 = self._normal2[selected_pts_mask]       # #

        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        
        new_indirect_dc = self._indirect_dc[selected_pts_mask]
        new_indirect_rest = self._indirect_rest[selected_pts_mask]
        new_indirect_asg = self._indirect_asg[selected_pts_mask]
        
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_refl_strength, new_metalness, new_roughness, new_ori_color, new_diffuse_color, new_ncif_dir, new_prt_occlusion, new_grt_transfer, new_features_dc, new_features_rest, new_indirect_dc, new_indirect_asg, new_indirect_rest, new_opacities, new_scaling, new_rotation, new_normal1, new_normal2)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter], dim=-1, keepdim=True)  # #
        self.denom[update_filter] += 1

    # #
    def set_requires_grad(self, attrib_name, state: bool):
        getattr(self, f"_{attrib_name}").requires_grad = state
        
    def update_mesh(self, mesh):
        vertices = np.asarray(mesh.vertices).astype(np.float32)
        faces = np.asarray(mesh.triangles).astype(np.int32)
        self.ray_tracer = raytracing.RayTracer(vertices, faces)

    def load_mesh_from_ply(self, model_path, iteration):
        import open3d as o3d
        import os

        ply_path = os.path.join(model_path, f'test_{iteration:06d}.ply')
        mesh = o3d.io.read_triangle_mesh(ply_path)
        self.update_mesh(mesh)
        
    

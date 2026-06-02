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

import os
import html
import torch
import open3d as o3d
from random import randint
from utils.loss_utils import calculate_loss, l1_loss
from gaussian_renderer import render_surfel, render_initial, render_volume, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import numpy as np
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from datetime import datetime
from torchvision.utils import save_image, make_grid
import torch.nn.functional as F
from utils.image_utils import visualize_depth
from utils.graphics_utils import linear_to_srgb
from utils.mesh_utils import GaussianExtractor, post_process_mesh
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False






def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, model_path, debug_from=None):
    first_iter = 0
    tb_writer = prepare_output_and_logger()

    # Set up parameters 
    TOT_ITER = opt.iterations + 1
    TEST_INTERVAL = 1000
    MESH_EXTRACT_INTERVAL = 2000

    # For real scenes
    USE_ENV_SCOPE = opt.use_env_scope  # False
    if USE_ENV_SCOPE:
        center = [float(c) for c in opt.env_scope_center]
        ENV_CENTER = torch.tensor(center, device='cuda')
        ENV_RADIUS = opt.env_scope_radius
        REFL_MSK_LOSS_W = 0.4


    gaussians = GaussianModel(
        dataset.sh_degree,
        use_ncif=dataset.use_ncif,
    )
    set_gaussian_para(gaussians, opt, vol=(opt.volume_render_until_iter > opt.init_until_iter)) # #
    scene = Scene(dataset, gaussians)  # init all parameters(pos, scale, rot...) from pcds
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    gaussExtractor = GaussianExtractor(gaussians, render_initial, pipe, bg_color=bg_color) 

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0
    ema_normal_smooth_for_log = 0.0
    ema_depth_smooth_for_log = 0.0
    ema_psnr_for_log = 0.0
    psnr_test = 0

    progress_bar = tqdm(range(first_iter, TOT_ITER), desc="Training progress")
    first_iter += 1
    iteration = first_iter

    print(f'Propagation until: {opt.normal_prop_until_iter }')
    print(f'Densify until: {opt.densify_until_iter}')
    print(f'Total iterations: {TOT_ITER}')


    initial_stage = opt.initial
    if not initial_stage:
        opt.init_until_iter = 0


    # Training loop
    while iteration < TOT_ITER:
        iter_start.record()

        gaussians.update_learning_rate(iteration)


        # Increase SH levels every 1000 iterations
        if iteration > opt.feature_rest_from_iter and iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Control the init stage
        if iteration > opt.init_until_iter:
            initial_stage = False
        
        # Control the indirect stage
        if iteration == opt.indirect_from_iter + 1:
            opt.indirect = 1


        if iteration == (opt.volume_render_until_iter + 1) and opt.volume_render_until_iter > opt.init_until_iter:
            if opt.use_pcc:
                soft_reset_gaussian_para(gaussians, opt)
            else:
                reset_gaussian_para(gaussians, opt)

        # Initialize envmap
        if not initial_stage:
            if iteration <= opt.volume_render_until_iter:
                envmap2 = gaussians.get_envmap_2 
                envmap2.build_mips()
            else:
                envmap = gaussians.get_envmap 
                envmap.build_mips()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))


        # Set render
        opt.enable_ncif = dataset.use_ncif and iteration >= opt.ncif_from_iter
        opt.enable_probe_gi = getattr(opt, "use_probe_gi", False) and iteration >= getattr(opt, "probe_from_iter", 0)
        opt.enable_prt_gs = getattr(opt, "use_prt_gs", False) and iteration >= getattr(opt, "prt_from_iter", 0)
        opt.current_iteration = iteration
        render = select_render_method(iteration, opt, initial_stage)
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, srgb=opt.srgb, opt=opt)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        gt_image = viewpoint_cam.original_image.cuda()

        total_loss, tb_dict = calculate_loss(viewpoint_cam, gaussians, render_pkg, opt, iteration)
        dist_loss, normal_loss, loss, Ll1, normal_smooth_loss, depth_smooth_loss = tb_dict["loss_dist"], tb_dict["loss_normal_render_depth"], tb_dict["loss0"], tb_dict["loss_l1"], tb_dict["loss_normal_smooth"], tb_dict["loss_depth_smooth"] 

        def get_outside_msk():
            return None if not USE_ENV_SCOPE else torch.sum((gaussians.get_xyz - ENV_CENTER[None])**2, dim=-1) > ENV_RADIUS**2
        
        if USE_ENV_SCOPE and 'refl_strength_map' in render_pkg:
            refls = gaussians.get_refl
            refl_msk_loss = refls[get_outside_msk()].mean()
            total_loss += REFL_MSK_LOSS_W * refl_msk_loss
        
        total_loss.backward()

        iter_end.record()


        with torch.no_grad():
            
            if iteration % TEST_INTERVAL == 0 or iteration == first_iter + 1 or iteration == opt.volume_render_until_iter + 1:
                save_training_vis(viewpoint_cam, gaussians, background, render, pipe, opt, iteration, initial_stage)

            ema_loss_for_log = 0.4 * loss + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss + 0.6 * ema_normal_for_log
            ema_normal_smooth_for_log = 0.4 * normal_smooth_loss + 0.6 * ema_normal_smooth_for_log
            ema_depth_smooth_for_log = 0.4 * depth_smooth_loss + 0.6 * ema_depth_smooth_for_log
            ema_psnr_for_log = 0.4 * psnr(image, gt_image).mean().double().item() + 0.6 * ema_psnr_for_log
            if iteration % TEST_INTERVAL == 0:
                fast_split = "val_fast" if len(scene.getValCameras()) > 0 else "test_fast"
                psnr_test = evaluate_psnr(scene, render, {"pipe": pipe, "bg_color": background, "opt": opt}, split=fast_split.split("_")[0])
                append_eval_curve(model_path, iteration, fast_split, None, psnr_test)
            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "Distort": f"{ema_dist_for_log:.{5}f}",
                    "Normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}",
                    "PSNR-train": f"{ema_psnr_for_log:.{4}f}",
                    "PSNR-test": f"{psnr_test:.{4}f}"
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            if iteration == TOT_ITER:
                progress_bar.close()

            if tb_writer:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                            testing_iterations, scene, render, {"pipe": pipe, "bg_color": background, "opt":opt})

            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter and iteration != opt.volume_render_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                     radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration <= opt.init_until_iter:
                    opacity_reset_intval = 3000
                    densification_interval = 100
                elif iteration <= opt.normal_prop_until_iter :
                    opacity_reset_intval = 3000
                    densification_interval = opt.densification_interval_when_prop
                else:
                    opacity_reset_intval = 3000
                    densification_interval = 100

                if iteration > opt.densify_from_iter and iteration % densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.prune_opacity_threshold, scene.cameras_extent,
                                                size_threshold)

                HAS_RESET0 = False
                if iteration % opacity_reset_intval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    HAS_RESET0 = True
                    outside_msk = get_outside_msk()
                    gaussians.reset_opacity0()
                    gaussians.reset_refl(exclusive_msk=outside_msk)
                if opt.opac_lr0_interval > 0 and (
                        opt.init_until_iter < iteration <= opt.normal_prop_until_iter ) and iteration % opt.opac_lr0_interval == 0:
                    gaussians.set_opacity_lr(opt.opacity_lr)
                if (opt.init_until_iter < iteration <= opt.normal_prop_until_iter ) and iteration % opt.normal_prop_interval == 0:
                    if not HAS_RESET0:
                        outside_msk = get_outside_msk()
                        gaussians.reset_opacity1(exclusive_msk=outside_msk)
                        if iteration > opt.volume_render_until_iter and opt.volume_render_until_iter > opt.init_until_iter:
                            gaussians.dist_color(exclusive_msk=outside_msk)
                            # gaussians.dist_albedo(exclusive_msk=outside_msk)

                        gaussians.reset_scale(exclusive_msk=outside_msk)
                        if opt.opac_lr0_interval > 0 and iteration != opt.normal_prop_until_iter :
                            gaussians.set_opacity_lr(0.0)
                
            if (iteration >= opt.indirect_from_iter and iteration % MESH_EXTRACT_INTERVAL == 0) or iteration == (opt.indirect_from_iter):
                if not HAS_RESET0:
                    gaussExtractor.reconstruction(scene.getTrainCameras())
                    if 'ref_real' in dataset.source_path:
                        mesh = gaussExtractor.extract_mesh_unbounded(resolution=opt.mesh_res)
                    else:
                        depth_trunc = (gaussExtractor.radius * 2.0) if opt.depth_trunc < 0  else opt.depth_trunc
                        voxel_size = (depth_trunc / opt.mesh_res) if opt.voxel_size < 0 else opt.voxel_size
                        sdf_trunc = 5.0 * voxel_size if opt.sdf_trunc < 0 else opt.sdf_trunc
                        mesh = gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)
                    mesh = post_process_mesh(mesh, cluster_to_keep=opt.num_cluster)
                    ply_path = os.path.join(model_path,f'test_{iteration:06d}.ply')
                    o3d.io.write_triangle_mesh(ply_path, mesh)
                    gaussians.update_mesh(mesh)

            if iteration < TOT_ITER:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save((gaussians.capture(), iteration), scene.model_path + f"/chkpnt{iteration}.pth")

        iteration += 1








# ============================================================
# Utils for training


def select_render_method(iteration, opt, initial_stage):

    if initial_stage:
        render = render_initial
    elif iteration <= opt.volume_render_until_iter:
        render = render_volume
    else:   
        render = render_surfel

    return render


def set_gaussian_para(gaussians, opt, vol=False):
    gaussians.enlarge_scale = opt.enlarge_scale
    gaussians.rough_msk_thr = opt.rough_msk_thr 
    gaussians.init_roughness_value = opt.init_roughness_value
    gaussians.init_refl_value = opt.init_refl_value
    gaussians.init_metalness_value = opt.init_metalness_value
    gaussians.refl_msk_thr = opt.refl_msk_thr

def reset_gaussian_para(gaussians, opt):
    gaussians.reset_ori_color()
    gaussians.reset_refl_strength(opt.init_refl_value)
    gaussians.reset_roughness(opt.init_roughness_value)
    gaussians.refl_msk_thr = opt.refl_msk_thr
    gaussians.rough_msk_thr = opt.rough_msk_thr


def soft_reset_gaussian_para(gaussians, opt):
    keep_ratio = max(0.0, min(1.0, float(opt.pcc_keep_ratio)))
    if getattr(opt, "use_adaptive_pcc", False):
        with torch.no_grad():
            min_keep = max(0.0, min(1.0, float(getattr(opt, "pcc_min_keep", 0.45))))
            max_keep = max(min_keep, min(1.0, float(getattr(opt, "pcc_max_keep", 0.85))))
            opacity_w = max(0.0, float(getattr(opt, "pcc_opacity_weight", 0.7)))
            specular_w = max(0.0, float(getattr(opt, "pcc_specular_weight", 0.3)))
            gamma = max(1e-6, float(getattr(opt, "pcc_confidence_gamma", 1.0)))
            weight_sum = max(opacity_w + specular_w, 1e-6)

            opacity_conf = gaussians.get_opacity.detach().clamp(0.0, 1.0).pow(gamma)
            specular_conf = (
                gaussians.get_refl.detach().clamp(0.0, 1.0)
                * (1.0 - gaussians.get_rough.detach().clamp(0.0, 1.0))
            ).pow(gamma)
            confidence = (opacity_w * opacity_conf + specular_w * specular_conf) / weight_sum
            keep_ratio = min_keep + (max_keep - min_keep) * confidence.clamp(0.0, 1.0)
            print(
                "[PCC] adaptive keep "
                f"mean={keep_ratio.mean().item():.4f}, "
                f"min={keep_ratio.min().item():.4f}, "
                f"max={keep_ratio.max().item():.4f}"
            )
    gaussians.soft_reset_ori_color(keep_ratio=keep_ratio)
    gaussians.soft_reset_refl_strength(opt.init_refl_value, keep_ratio=keep_ratio)
    gaussians.soft_reset_roughness(opt.init_roughness_value, keep_ratio=keep_ratio)
    gaussians.refl_msk_thr = opt.refl_msk_thr
    gaussians.rough_msk_thr = opt.rough_msk_thr




def save_training_vis(viewpoint_cam, gaussians, background, render_fn, pipe, opt, iteration, initial_stage):
    with torch.no_grad():
        render_pkg = render_fn(viewpoint_cam, gaussians, pipe, background, srgb=opt.srgb, opt=opt)

        error_map = torch.abs(viewpoint_cam.original_image.cuda() - render_pkg["render"])

        if initial_stage:
            visualization_list = [
                viewpoint_cam.original_image.cuda(),
                render_pkg["render"], 
                render_pkg["rend_alpha"].repeat(3, 1, 1),
                visualize_depth(render_pkg["surf_depth"]),  
                render_pkg["rend_normal"] * 0.5 + 0.5, 
                render_pkg["surf_normal"] * 0.5 + 0.5, 
                error_map 
            ]

        elif iteration <= opt.volume_render_until_iter:
            visualization_list = [
                viewpoint_cam.original_image.cuda(),  
                render_pkg["render"], 
                render_pkg["base_color_map"], 
                render_pkg["diffuse_map"],      
                render_pkg["specular_map"],  
                render_pkg["refl_strength_map"].repeat(3, 1, 1),  
                render_pkg["roughness_map"].repeat(3, 1, 1),
                render_pkg["rend_alpha"].repeat(3, 1, 1),  
                visualize_depth(render_pkg["surf_depth"]), 
                render_pkg["rend_normal"] * 0.5 + 0.5,  
                render_pkg["surf_normal"] * 0.5 + 0.5, 
                error_map
            ]
            if opt.indirect:
                visualization_list += [
                    render_pkg["visibility"].repeat(3, 1, 1),
                    render_pkg["direct_light"],
                    render_pkg["indirect_light"],
                ]
            if "oaf_blend" in render_pkg:
                visualization_list.append(render_pkg["oaf_blend"].repeat(3, 1, 1))
            if "specular_reliability" in render_pkg:
                visualization_list.append(render_pkg["specular_reliability"].repeat(3, 1, 1))
            if "probe_map" in render_pkg:
                visualization_list.append(torch.tanh(render_pkg["probe_map"]) * 0.5 + 0.5)
            if "probe_responsibility" in render_pkg:
                visualization_list.append(render_pkg["probe_responsibility"].repeat(3, 1, 1))
            if "prt_map" in render_pkg:
                visualization_list.append(torch.tanh(render_pkg["prt_map"]) * 0.5 + 0.5)
            if "prt_responsibility" in render_pkg:
                visualization_list.append(render_pkg["prt_responsibility"].repeat(3, 1, 1))

        else:
            visualization_list = [
                viewpoint_cam.original_image.cuda(),  
                render_pkg["render"],  
                render_pkg["base_color_map"],  
                render_pkg["diffuse_map"],
                render_pkg["specular_map"],
                render_pkg["refl_strength_map"].repeat(3, 1, 1),  
                render_pkg["roughness_map"].repeat(3, 1, 1),
                render_pkg["rend_alpha"].repeat(3, 1, 1),  
                visualize_depth(render_pkg["surf_depth"]),  
                render_pkg["rend_normal"] * 0.5 + 0.5,  
                render_pkg["surf_normal"] * 0.5 + 0.5,  
                error_map, 
            ]
            if "oaf_blend" in render_pkg:
                visualization_list.append(render_pkg["oaf_blend"].repeat(3, 1, 1))
            if "specular_reliability" in render_pkg:
                visualization_list.append(render_pkg["specular_reliability"].repeat(3, 1, 1))
            if "probe_map" in render_pkg:
                visualization_list.append(torch.tanh(render_pkg["probe_map"]) * 0.5 + 0.5)
            if "probe_responsibility" in render_pkg:
                visualization_list.append(render_pkg["probe_responsibility"].repeat(3, 1, 1))
            if "prt_map" in render_pkg:
                visualization_list.append(torch.tanh(render_pkg["prt_map"]) * 0.5 + 0.5)
            if "prt_responsibility" in render_pkg:
                visualization_list.append(render_pkg["prt_responsibility"].repeat(3, 1, 1))
  

        grid = torch.stack(visualization_list, dim=0)
        grid = make_grid(grid, nrow=4)
        scale = grid.shape[-2] / 800
        grid = F.interpolate(grid[None], (int(grid.shape[-2] / scale), int(grid.shape[-1] / scale)))[0]
        save_image(grid, os.path.join(args.visualize_path, f"{iteration:06d}.png"))

        if not initial_stage:
            if opt.volume_render_until_iter > opt.init_until_iter and iteration <= opt.volume_render_until_iter:
                env_dict = gaussians.render_env_map_2() 
            else:
                env_dict = gaussians.render_env_map()

            grid = [
                env_dict["env1"].permute(2, 0, 1),
                env_dict["env2"].permute(2, 0, 1),
            ]
            grid = make_grid(grid, nrow=1, padding=10)
            save_image(grid, os.path.join(args.visualize_path, f"{iteration:06d}_env.png"))

      
NORM_CONDITION_OUTSIDE = False
def prepare_output_and_logger():    
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    args.visualize_path = os.path.join(args.model_path, "visualize")
    
    os.makedirs(args.visualize_path, exist_ok=True)
    print("Visualization folder: {}".format(args.visualize_path))
    
    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def append_eval_curve(model_path, iteration, split, l1_value, psnr_value):
    curve_path = os.path.join(model_path, "eval_curve.txt")
    write_header = not os.path.exists(curve_path)
    with open(curve_path, "a", encoding="utf-8") as f:
        if write_header:
            f.write(f"{'Iter':>8} {'Split':<12} {'L1':>14} {'PSNR':>14}\n")
            f.write("-" * 52 + "\n")
        if l1_value is None:
            l1_text = "N/A"
        else:
            l1_float = float(l1_value.detach().cpu().item()) if torch.is_tensor(l1_value) else float(l1_value)
            l1_text = f"{l1_float:.10f}"
        psnr_float = float(psnr_value.detach().cpu().item()) if torch.is_tensor(psnr_value) else float(psnr_value)
        f.write(f"{iteration:>8} {split:<12} {l1_text:>14} {psnr_float:>14.10f}\n")
    try:
        write_eval_curve_svg(curve_path, os.path.join(model_path, "eval_curve.svg"))
    except Exception:
        pass


def write_eval_curve_svg(curve_path, svg_path):
    rows = []
    with open(curve_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4 or not parts[0].isdigit():
                continue
            try:
                rows.append({
                    "iteration": int(parts[0]),
                    "split": parts[1],
                    "psnr": float(parts[3]),
                })
            except ValueError:
                continue
    if not rows:
        return

    colors = {
        "val_fast": "#7c3aed",
        "val": "#9333ea",
        "test_fast": "#2563eb",
        "test": "#16a34a",
        "train": "#d97706",
        "offline_eval": "#dc2626",
    }
    grouped = {}
    for row in rows:
        grouped.setdefault(row["split"], []).append(row)
    for values in grouped.values():
        values.sort(key=lambda item: item["iteration"])

    width, height = 920, 420
    left, right, top, bottom = 64, 24, 34, 58
    plot_w = width - left - right
    plot_h = height - top - bottom
    min_x = min(row["iteration"] for row in rows)
    max_x = max(row["iteration"] for row in rows)
    min_y = min(row["psnr"] for row in rows)
    max_y = max(row["psnr"] for row in rows)
    if max_x == min_x:
        max_x += 1
    if max_y == min_y:
        max_y += 1.0
    pad_y = max(0.25, (max_y - min_y) * 0.08)
    min_y -= pad_y
    max_y += pad_y

    def x_pos(value):
        return left + (value - min_x) / (max_x - min_x) * plot_w

    def y_pos(value):
        return top + (max_y - value) / (max_y - min_y) * plot_h

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="22" font-family="Arial" font-size="16" fill="#111827">Evaluation PSNR Curve</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#9ca3af"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#9ca3af"/>',
    ]

    for idx in range(5):
        y_val = min_y + (max_y - min_y) * idx / 4.0
        y = y_pos(y_val)
        svg.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e5e7eb"/>')
        svg.append(f'<text x="{left - 8}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial" font-size="11" fill="#4b5563">{y_val:.2f}</text>')

    for idx in range(5):
        x_val = int(round(min_x + (max_x - min_x) * idx / 4.0))
        x = x_pos(x_val)
        svg.append(f'<line x1="{x:.2f}" y1="{top + plot_h}" x2="{x:.2f}" y2="{top + plot_h + 5}" stroke="#9ca3af"/>')
        svg.append(f'<text x="{x:.2f}" y="{top + plot_h + 22}" text-anchor="middle" font-family="Arial" font-size="11" fill="#4b5563">{x_val}</text>')

    legend_x = left + 10
    for idx, split in enumerate(sorted(grouped.keys())):
        color = colors.get(split, "#6b7280")
        y = top + 18 + idx * 18
        svg.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 18}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        svg.append(f'<text x="{legend_x + 24}" y="{y + 4}" font-family="Arial" font-size="12" fill="#374151">{html.escape(split)}</text>')

    for split, values in grouped.items():
        color = colors.get(split, "#6b7280")
        points = " ".join(f'{x_pos(row["iteration"]):.2f},{y_pos(row["psnr"]):.2f}' for row in values)
        if len(values) == 1:
            row = values[0]
            svg.append(f'<circle cx="{x_pos(row["iteration"]):.2f}" cy="{y_pos(row["psnr"]):.2f}" r="3" fill="{color}"/>')
        else:
            svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.4"/>')
            for row in values:
                svg.append(f'<circle cx="{x_pos(row["iteration"]):.2f}" cy="{y_pos(row["psnr"]):.2f}" r="2.5" fill="{color}"/>')

    best = max((row for row in rows if row["split"] in {"test", "test_fast"}), key=lambda row: row["psnr"], default=max(rows, key=lambda row: row["psnr"]))
    final = max((row for row in rows if row["split"] in {"test", "test_fast"}), key=lambda row: row["iteration"], default=max(rows, key=lambda row: row["iteration"]))
    svg.append(f'<text x="{left}" y="{height - 16}" font-family="Arial" font-size="12" fill="#111827">best {best["iteration"]}: {best["psnr"]:.3f} | final {final["iteration"]}: {final["psnr"]:.3f} | drop {best["psnr"] - final["psnr"]:.3f}</text>')
    svg.append("</svg>")

    with open(svg_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))


@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderkwargs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1, iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss, iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = []
        if len(scene.getValCameras()) > 0:
            validation_configs.append({'name': 'val', 'cameras' : scene.getValCameras()})
        validation_configs.extend((
            {'name': 'test', 'cameras' : scene.getTestCameras()},
            {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]}
        ))

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(tqdm(config['cameras'])):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, **renderkwargs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                append_eval_curve(scene.model_path, iteration, config['name'], l1_test, psnr_test)
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

@torch.no_grad()
def evaluate_psnr(scene, renderFunc, renderkwargs, split="test"):
    psnr_test = 0.0
    torch.cuda.empty_cache()
    if split == "val":
        cameras = scene.getValCameras()
    elif split == "train":
        cameras = scene.getTrainCameras()
    else:
        cameras = scene.getTestCameras()
    if len(cameras):
        for viewpoint in cameras:
            render_pkg = renderFunc(viewpoint, scene.gaussians, **renderkwargs)
            image = torch.clamp(render_pkg["render"], 0.0, 1.0)
            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
            psnr_test += psnr(image, gt_image).mean().double()

        psnr_test /= len(cameras)
        
    torch.cuda.empty_cache()
    return psnr_test





# ============================================================================
# Main function


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[10000,20000,30000,40000,50000,60000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--gpu", type=str, default="-1")
    args = parser.parse_args(sys.argv[1:])
    if args.gpu != "-1":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.save_iterations.append(args.iterations)
    args.test_iterations = args.test_iterations + [i for i in range(10000, args.iterations+1, 5000)]
    args.test_iterations.append(args.volume_render_until_iter)

    
    if not args.model_path:
        # 获取当前时间并格式化为精确到分钟
        current_time = datetime.now().strftime('%m%d_%H%M')
        # 获取args.source_path的最后一个子目录名
        last_subdir = os.path.basename(os.path.normpath(args.source_path))

        
        # 生成带有时间戳和opt属性的简洁输出目录
        args.model_path = os.path.join(
            "./output/", f"{last_subdir}/",
            f"{last_subdir}-{current_time}"
        )

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.model_path)

    # All done
    print("\nTraining complete.")

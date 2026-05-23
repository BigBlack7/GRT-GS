import torch
import torch.nn.functional as F
from scene import Scene
import os, time, json
import numpy as np
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render_surfel
import torchvision
from utils.general_utils import safe_state
from utils.system_utils import searchForMaxIteration
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.image_utils import psnr
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
from torchvision.utils import save_image, make_grid

def render_set(model_path, views, gaussians, pipeline, background, save_ims, opt):
    if save_ims:
        # Create directories to save rendered images
        render_path = os.path.join(model_path, "test", "renders")
        color_path = os.path.join(render_path, 'rgb')
        normal_path = os.path.join(render_path, 'normal')
        makedirs(color_path, exist_ok=True)
        makedirs(normal_path, exist_ok=True)

    ssims = []
    psnrs = []
    lpipss = []
    normal_maes = []
    render_times = []

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        view.refl_mask = None  # When evaluating, reflection mask is disabled
        t1 = time.time()
        
        rendering = render_surfel(view, gaussians, pipeline, background, srgb=opt.srgb, opt=opt)
        render_time = time.time() - t1
        
        render_color = torch.clamp(rendering["render"], 0.0, 1.0)
        render_color = render_color[None]
        gt = torch.clamp(view.original_image, 0.0, 1.0)
        gt = gt[None, 0:3, :, :]

        ssims.append(ssim(render_color, gt).item())
        psnrs.append(psnr(render_color, gt).item())
        lpipss.append(lpips(render_color, gt, net_type='vgg').item())
        render_times.append(render_time)
        if getattr(view, "gt_normal", None) is not None and 'rend_normal' in rendering:
            pred_normal = F.normalize(rendering['rend_normal'], dim=0, eps=1e-6)
            gt_normal = F.normalize(view.gt_normal.to(pred_normal.device), dim=0, eps=1e-6)
            cos = (pred_normal * gt_normal).sum(dim=0).clamp(-1.0, 1.0)
            normal_err = torch.rad2deg(torch.acos(cos))
            alpha_mask = rendering.get('rend_alpha', torch.ones_like(normal_err[None]))[0] > 0.5
            normal_maes.append(normal_err[alpha_mask].mean().item() if alpha_mask.any() else normal_err.mean().item())

        if save_ims:
            # Save the rendered color image
            torchvision.utils.save_image(render_color, os.path.join(color_path, '{0:05d}.png'.format(idx)))
            # Save the normal map if available
            if 'rend_normal' in rendering:
                normal_map = rendering['rend_normal'] * 0.5 + 0.5
                torchvision.utils.save_image(normal_map, os.path.join(normal_path, '{0:05d}.png'.format(idx)))
            
    ssim_v = np.array(ssims).mean()
    psnr_v = np.array(psnrs).mean()
    lpip_v = np.array(lpipss).mean()
    fps = 1.0 / np.array(render_times).mean()
    metrics = {'psnr': float(psnr_v), 'ssim': float(ssim_v), 'lpips': float(lpip_v), 'fps': float(fps)}
    if normal_maes:
        metrics['normal_mae'] = float(np.array(normal_maes).mean())
    print(', '.join(f'{k}:{v}' for k, v in metrics.items()))
    dump_path = os.path.join(model_path, 'metric.txt')
    with open(dump_path, 'w') as f:
        f.write(', '.join(f'{k}:{v}' for k, v in metrics.items()))
    with open(os.path.join(model_path, 'results.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

def render_set_train(model_path, views, gaussians, pipeline, background, save_ims, opt):
    if save_ims:
        # Create directories to save rendered images
        render_path = os.path.join(model_path, "train", "renders")
        color_path = os.path.join(render_path, 'rgb')
        gt_path = os.path.join(render_path, 'gt')
        normal_path = os.path.join(render_path, 'normal')
        makedirs(color_path, exist_ok=True)
        makedirs(gt_path, exist_ok=True)
        makedirs(normal_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        view.refl_mask = None  # When evaluating, reflection mask is disabled
        rendering = render_surfel(view, gaussians, pipeline, background, srgb=opt.srgb, opt=opt)
 
        render_color = torch.clamp(rendering["render"], 0.0, 1.0)
        render_color = render_color[None]
        gt = torch.clamp(view.original_image, 0.0, 1.0)
        gt = gt[None, :3, :, :]

        if save_ims:
            # Save the rendered color image
            torchvision.utils.save_image(render_color, os.path.join(color_path, '{0:05d}.png'.format(idx)))
            torchvision.utils.save_image(gt, os.path.join(gt_path, '{0:05d}.png'.format(idx)))
            # Save the normal map if available
            if 'rend_normal' in rendering:
                normal_map = rendering['rend_normal'] * 0.5 + 0.5
                torchvision.utils.save_image(normal_map, os.path.join(normal_path, '{0:05d}.png'.format(idx)))
            

            

   
def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, save_ims: bool, op, indirect_override):
    with torch.no_grad():
        gaussians = GaussianModel(
            dataset.sh_degree,
            use_ncif=dataset.use_ncif,
        )
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        iteration = scene.loaded_iter
        if iteration is None:
            iteration = searchForMaxIteration(os.path.join(dataset.model_path, "point_cloud"))
        op.enable_ncif = dataset.use_ncif and iteration >= op.ncif_from_iter
        op.current_iteration = iteration
        mesh_path = os.path.join(dataset.model_path, f'test_{iteration:06d}.ply')
        use_indirect = (iteration >= op.indirect_from_iter and os.path.exists(mesh_path)) if indirect_override is None else indirect_override
        if use_indirect:
            op.indirect = 1
            gaussians.load_mesh_from_ply(dataset.model_path, iteration)
        else:
            op.indirect = 0
        print(
            "Loaded eval state: "
            f"iteration={iteration}, "
            f"enable_ncif={op.enable_ncif}, "
            f"indirect={op.indirect}, "
            f"mesh_exists={os.path.exists(mesh_path)}"
        )

        
        # render_set_train(dataset.model_path, scene.getTrainCameras(), gaussians, pipeline, background, save_ims, op)
        render_set(dataset.model_path, scene.getTestCameras(), gaussians, pipeline, background, save_ims, op)
        
        env_dict = gaussians.render_env_map()
        grid = [
            env_dict["env1"].permute(2, 0, 1),
        ]
        grid = make_grid(grid, nrow=1, padding=10)
        save_image(grid, os.path.join(dataset.model_path, "env1.png"))
        grid = [
            env_dict["env2"].permute(2, 0, 1),
        ]
        grid = make_grid(grid, nrow=1, padding=10)
        save_image(grid, os.path.join(dataset.model_path, "env2.png"))


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser, sentinel=True)
    pipeline = PipelineParams(parser, sentinel=True)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--gpu", type=str, default="-1")
    parser.add_argument("--force_indirect", dest="indirect_override", action="store_true", default=None)
    parser.add_argument("--no_indirect", dest="indirect_override", action="store_false")
    args = get_combined_args(parser)
    if not hasattr(args, "indirect_override"):
        args.indirect_override = None
    if args.gpu != "-1":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    dataset = model.extract(args)
    opt = op.extract(args)
    pipe = pipeline.extract(args)
    print(
        "Eval config: "
        f"iteration={args.iteration}, "
        f"use_ncif={dataset.use_ncif}, "
        f"ncif_from_iter={opt.ncif_from_iter}, "
        f"srgb={opt.srgb}, "
        f"indirect_override={args.indirect_override}"
    )
    render_sets(dataset, args.iteration, pipe, args.save_images, opt, args.indirect_override)

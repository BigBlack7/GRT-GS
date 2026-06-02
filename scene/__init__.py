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
import random
import json
import numpy as np
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.val_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args.lod, args.llffhold)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, undistorted=args.undistorted)
        else:
            raise AssertionError(
                "Could not recognize scene type! "
                f"source_path={args.source_path}, "
                "expected either a COLMAP 'sparse' directory or a Blender 'transforms_train.json' file."
            )

        train_cam_infos = list(scene_info.train_cameras)
        val_cam_infos = []
        test_cam_infos = list(scene_info.test_cameras)
        val_hold = int(getattr(args, "val_hold", 0) or 0)
        val_max = int(getattr(args, "val_max", 8) or 0)
        val_offset = int(getattr(args, "val_offset", 0) or 0)
        if val_hold > 0 and len(train_cam_infos) > 2:
            val_indices = [
                idx
                for idx in range(len(train_cam_infos))
                if (idx - val_offset) % val_hold == 0
            ]
            if val_max > 0:
                val_indices = val_indices[:val_max]
            if 0 < len(val_indices) < len(train_cam_infos):
                val_index_set = set(val_indices)
                val_cam_infos = [cam for idx, cam in enumerate(train_cam_infos) if idx in val_index_set]
                train_cam_infos = [cam for idx, cam in enumerate(train_cam_infos) if idx not in val_index_set]
                print(
                    "Validation split: "
                    f"held {len(val_cam_infos)} / {len(scene_info.train_cameras)} train cameras "
                    f"(val_hold={val_hold}, val_offset={val_offset}, val_max={val_max})"
                )

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if test_cam_infos:
                camlist.extend(test_cam_infos)
            if val_cam_infos:
                camlist.extend(val_cam_infos)
            if train_cam_infos:
                camlist.extend(train_cam_infos)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)
            split_info = {
                "train": [cam.image_name for cam in train_cam_infos],
                "val": [cam.image_name for cam in val_cam_infos],
                "test": [cam.image_name for cam in test_cam_infos],
                "val_hold": val_hold,
                "val_offset": val_offset,
                "val_max": val_max,
            }
            with open(os.path.join(self.model_path, "camera_split.json"), 'w') as file:
                json.dump(split_info, file, indent=2)

        if shuffle:
            random.shuffle(train_cam_infos)  # Multi-res consistent random shuffling
            random.shuffle(val_cam_infos)  # Multi-res consistent random shuffling
            random.shuffle(test_cam_infos)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(train_cam_infos, resolution_scale, args)
            print("Loading Validation Cameras")
            self.val_cameras[resolution_scale] = cameraList_from_camInfos(val_cam_infos, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(test_cam_infos, resolution_scale, args)

        if self.loaded_iter:
            if args.relight:
                self.gaussians.load_ply(os.path.join(self.model_path,
                                                            "point_cloud",
                                                            "iteration_" + str(self.loaded_iter),
                                                            "point_cloud.ply"), relight=True, args=args)
            else:
                self.gaussians.load_ply(os.path.join(self.model_path,
                                                            "point_cloud",
                                                            "iteration_" + str(self.loaded_iter),
                                                            "point_cloud.ply"), args=args)        
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, args)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getValCameras(self, scale=1.0):
        return self.val_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

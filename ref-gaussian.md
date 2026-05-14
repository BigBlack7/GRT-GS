# Ref-Gaussian: Reflective Gaussian Splatting
### [[Project]](https://fudan-zvg.github.io/ref-gaussian) [[Paper]](https://arxiv.org/abs/2412.19282)

> [**Reflective Gaussian Splatting**](https://arxiv.org/abs/2412.19282),            
> [Yuxuan Yao](https://yaoyuxuanyyds.github.io/), Zixuan Zeng, [Chun Gu](https://sulvxiangxin.github.io/), [Xiatian Zhu](https://surrey-uplab.github.io/), [Li Zhang](https://lzrobots.github.io)  
> **ICLR 2025**

**Official implementation of "Reflective Gaussian Splatting".** 

## ⚙️ Get started
### Installation
```bash
# clone the repo
git clone https://github.com/fudan-zvg/ref-gaussian.git --recursive
cd ref-gaussian
# create conda environment
conda create -n ref-gaussian python=3.8
conda activate ref-gaussian

# install pytorch (e.g. cuda 11.7)
pip install torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0
# install submodules
pip install submodules/cubemapencoder
pip install submodules/diff-surfel-rasterization
pip install submodules/simple-knn
pip install submodules/raytracing

# install other denpendencies
pip install -r requirements.txt
```


### Dateset
We mainly test our method on [Shiny Blender Synthetic](https://storage.googleapis.com/gresearch/refraw360/ref.zip), [Shiny Blender Real](https://storage.googleapis.com/gresearch/refraw360/ref_real.zip), [Glossy Synthetic](https://liuyuan-pal.github.io/NeRO/) and [NeRF Synthetic dataset](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1). Please run the script `nero2blender.py` to convert the format of the Glossy Synthetic dataset.


###  Running
We provide the script to test our code on each scene of datasets. Just run:
```
sh train.sh
```
You may need to modify the path in `train.sh`

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for train.py</span></summary>

  #### --iteration
  The number of total iteration for training.

  #### --lambda_normal_smooth 
  The strength of normal smooth loss.
  
</details>
<br>

### Evaluation
```
python eval.py --white_background --save_images --model_path output/NAME_OF_THE_SCENE --eval_indirect
```
You will get PSNR/SSIM/LPIPS/FPS results.



## Acknowledgement

This work is built on many amazing research works:

- [3DGS-DR](https://github.com/gapszju/3DGS-DR)
- [2DGS](https://github.com/hbb1/2d-gaussian-splatting)
- [Raytracing](https://github.com/ashawkey/raytracing)

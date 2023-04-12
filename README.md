# Edge-Aware NeRF
We make an attempt to make use of edge information. Here's the flow of execution.

# Our Code in Pytorch
Our work is based on this [open source work](https://github.com/sjtuytc/LargeScaleNeRFPytorch). Thank you very much for sharing ! ! !

## 1. Introduction

**I was writing a report about our progress now but feel free to use our code without citing us. Below is our abstract of report:**

*In this research, we investigate the novel challenge of enhancing the rendering quality of intricate scenes. Considering the issue of edge blurring arising from current image rendering techniques, we aim to augment the fidelity of Neural Radiance Fields (NeRF) rendering by leveraging available edge detection outcomes. To address this challenge, we scrutinize the distribution of edge information within color images. By integrating edge features into the NeRF network's learning process, we specifically assign weights to the outcomes of edge detection and incorporate them into the rendering loss of the NeRF network. To assess the practicality of our approach, we conducted experiments employing five prevalent edge detection methodologies on six distinct data sets, and subsequently visualized the results to analyze the influence of edge information.*


## 3. Installation
<details>
<summary>Expand / collapse installation steps.</summary>

1. Create conda environment.
   ```bash
   conda create -n edge-nerf python=3.9
   conda activate edge-nerf
   ```
2. Install pytorch, and other libs. Make sure your Pytorch version is compatible with your CUDA.
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia

3. Install grid-based operators to avoid running them every time, cuda lib required. (Check via "nvcc -V" to ensure that you have a latest cuda.)
   ```bash
   apt-get install g++ build-essential  # ensure you have g++ and other build essentials, sudo access required.
   cd FourierGrid/cuda
   python setup.py install
   cd ../../
   ```

## 4. Edge-Aware NeRF on the public datasets

Click the following sub-section titles to expand / collapse steps.

<details>
<summary> 4.1 Download processed data.</summary>

(1) [Unbounded Tanks & Temples](https://www.tanksandtemples.org/). Download data from [here](https://drive.google.com/file/d/11KRfN91W1AxAW6lOFs4EeYDbeoQZCi87/view). Then unzip the data.

```bash
cd data
gdown --id 11KRfN91W1AxAW6lOFs4EeYDbeoQZCi87
unzip tanks_and_temples.zip
cd ../
```
	
(2) The [Mip-NeRF-360](https://jonbarron.info/mipnerf360/) dataset.

```bash
cd data
wget http://storage.googleapis.com/gresearch/refraw360/360_v2.zip
mkdir 360_v2
unzip 360_v2.zip -d 360_v2
cd ../
```

The downloaded data would look like this:

   ```
   data
      |
      |——————360_v2                                    // the root folder for the Mip-NeRF-360 benchmark
      |        └——————bicycle                          // one scene under the Mip-NeRF-360 benchmark
      |        |         └——————images                 // rgb images
      |        |         └——————images_2               // rgb images downscaled by 2
      |        |         └——————sparse                 // camera poses
      |        ...
      |——————tanks_and_temples                         // the root folder for Tanks&Temples
      |        └——————tat_intermediate_M60             // one scene under Tanks&Temples
      |                  └——————camera_path            // render split camera poses, intrinsics and extrinsics
      |                  └——————test                   // test split
      |                  └——————train                  // train split
      |                  └——————validation             // validation split
      |——————-----
   ```
</details>

<details>
<summary> 4.2 Train models and see the results!</summary>

You only need to run "python run_FourierGrid.py" to finish the train-test-render cycle. Explanations of some arguments: 
```bash
--program: the program to run, normally --program train will be all you need.
--config: the config pointing to the scene file, e.g., --config FourierGrid/configs/tankstemple_unbounded/truck_single.py.
--num_per_block: number of blocks used in edge NeRFs, normally this is set to -1, unless specially needed.
--render_train: render the trained model on the train split.
--render_train: render the trained model on the test split.
--render_train: render the trained model on the render split.
--exp_id: add some experimental ids to identify different experiments. E.g., --exp_id 5.
--eval_ssim / eval_lpips_vgg: report SSIM / LPIPS(VGG) scores.
```

While we list major of the commands in scripts/train_FourierGrid.sh, we list some of commands below for better reproducibility.

```bash
# Unbounded tanks and temples
for edgetype in canny sobel laplacian roberts prewitt
do
    CUDA_VISIBLE_DEVICES=1 python -W ignore run_FourierGrid.py --program train --config FourierGrid/configs/tankstemple_unbounded//m60_single.py --num_per_block -1 --render_train --render_test --render_video --exp_id 6 --edgeLoss --edgeType ${edgetype} --edgeTrainType EdgeWeight --edgeRenderWeight 0.15
done

# 360 degree dataset
CUDA_VISIBLE_DEVICES=1 python -W ignore run_FourierGrid.py --program train --config FourierGrid/configs/nerf_unbounded/room_single.py --num_per_block -1 --eval_ssim --eval_lpips_vgg --render_train --render_test --render_video --exp_id 9
for edgetype in canny sobel laplacian roberts prewitt
do
    CUDA_VISIBLE_DEVICES=1 python -W ignore run_FourierGrid.py --program train --config FourierGrid/configs/nerf_unbounded/room_single.py --num_per_block -1 --eval_ssim --eval_lpips_vgg --render_train --render_test --render_video --exp_id 9 --edgeLoss --edgeType ${edgetype} --edgeTrainType EdgeWeight --edgeRenderWeight 0.15
done

CUDA_VISIBLE_DEVICES=1 python -W ignore run_FourierGrid.py --program train --config FourierGrid/configs/nerf_unbounded/stump_single.py --num_per_block -1 --eval_ssim --eval_lpips_vgg --render_train --render_test --render_video --exp_id 10
for edgetype in canny sobel laplacian roberts prewitt
do
    CUDA_VISIBLE_DEVICES=1 python -W ignore run_FourierGrid.py --program train --config FourierGrid/configs/nerf_unbounded/stump_single.py --num_per_block -1 --eval_ssim --eval_lpips_vgg --render_train --render_test --render_video --exp_id 10 --edgeLoss --edgeType ${edgetype} --edgeTrainType EdgeWeight --edgeRenderWeight 0.15
done

CUDA_VISIBLE_DEVICES=1 python -W ignore run_FourierGrid.py --program train --config FourierGrid/configs/nerf_unbounded/bonsai_single.py --num_per_block -1 --eval_ssim --eval_lpips_vgg --render_test --render_video --exp_id 3
for edgetype in canny sobel laplacian roberts prewitt
do
    CUDA_VISIBLE_DEVICES=1 python -W ignore run_FourierGrid.py --program train --config FourierGrid/configs/nerf_unbounded/bonsai_single.py --num_per_block -1 --eval_ssim --eval_lpips_vgg --render_test --render_video --exp_id 3 --edgeLoss --edgeType ${edgetype} --edgeTrainType EdgeWeight --edgeRenderWeight 0.15
done

CUDA_VISIBLE_DEVICES=1 python -W ignore run_FourierGrid.py --program train --config FourierGrid/configs/nerf_unbounded/kitchen_single.py --num_per_block -1 --eval_ssim --eval_lpips_vgg --render_train --render_test --render_video --exp_id 2
for edgetype in canny sobel laplacian roberts prewitt
do
    CUDA_VISIBLE_DEVICES=1 python -W ignore run_FourierGrid.py --program train --config FourierGrid/configs/nerf_unbounded/kitchen_single.py --num_per_block -1 --eval_ssim --eval_lpips_vgg --render_train --render_test --render_video --exp_id 2 --edgeLoss --edgeType ${edgetype} --edgeTrainType EdgeWeight --edgeRenderWeight 0.15
done

CUDA_VISIBLE_DEVICES=1 python -W ignore run_FourierGrid.py --program train --config FourierGrid/configs/nerf_unbounded/counter_single.py --num_per_block -1 --eval_ssim --eval_lpips_vgg --render_train --render_test --render_video --exp_id 2
for edgetype in canny sobel laplacian roberts prewitt
do
    CUDA_VISIBLE_DEVICES=1 python -W ignore run_FourierGrid.py --program train --config FourierGrid/configs/nerf_unbounded/counter_single.py --num_per_block -1 --eval_ssim --eval_lpips_vgg --render_train --render_test --render_video --exp_id 2 --edgeLoss --edgeType ${edgetype} --edgeTrainType EdgeWeight --edgeRenderWeight 0.15
done
```

## 5. Configuration of edge detection information

You can configure edge detection information in the corresponding config file !

5.1 Configuration of edge detection information

```
edgeType = 'canny' # canny; sobel; laplacian; roberts; prewitt;
edgeLoss = False
edgeTrainType = 'EdgeWeight' # EdgeWeight; FitEdge;
edgeRenderWeight = 0.15
```

5.2 Command line modification mode
After configuring edge detection information in the configuration file, if you want to modify the configuration parameters when running the training command, modify the command line parameters of run_FourierGrid.py in the following ways
```
--edgeLoss --edgeType ${edgetype} --edgeTrainType EdgeWeight --edgeRenderWeight 0.15
edgetype = canny/sobel/roberts/laplacian/prewitt
```

## 6. Visualization experiment

Run the following command to save the visual result to the corresponding log file directory:

6.1 Plot Edge Detection
```
python tools/plot_detection.py
```

6.2 Visualized rendering loss heatmap

```
python tools/catimage.py
```

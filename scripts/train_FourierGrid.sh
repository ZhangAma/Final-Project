# Unbounded tanks and temples
# CUDA_VISIBLE_DEVICES=1 python -W ignore run_FourierGrid.py --program train --config FourierGrid/configs/tankstemple_unbounded/playground_single.py --num_per_block -1 --render_train --render_test --render_video --exp_id 57
# CUDA_VISIBLE_DEVICES=0 python -W ignore run_FourierGrid.py --program train --config FourierGrid/configs/tankstemple_unbounded/train_single.py --num_per_block -1 --render_train --render_test --render_video --exp_id 12
# CUDA_VISIBLE_DEVICES=0 python -W ignore run_FourierGrid.py --program train --config FourierGrid/configs/tankstemple_unbounded/truck_single.py --num_per_block -1 --render_train --render_test --render_video --exp_id 4
# CUDA_VISIBLE_DEVICES=1 python -W ignore run_FourierGrid.py --program train --config FourierGrid/configs/tankstemple_unbounded/m60_single.py --num_per_block -1 --render_train --render_test --render_video --exp_id 6
# for edgetype in canny sobel laplacian roberts prewitt
# do
#     CUDA_VISIBLE_DEVICES=1 python -W ignore run_FourierGrid.py --program train --config FourierGrid/configs/tankstemple_unbounded//m60_single.py --num_per_block -1 --render_train --render_test --render_video --exp_id 6 --edgeLoss --edgeType ${edgetype} --edgeTrainType EdgeWeight --edgeRenderWeight 0.15
# done

# 360 degree dataset
# CUDA_VISIBLE_DEVICES=1 python -W ignore run_FourierGrid.py --program train --config FourierGrid/configs/nerf_unbounded/room_single.py --num_per_block -1 --eval_ssim --eval_lpips_vgg --render_train --render_test --render_video --exp_id 9
# for edgetype in sobel laplacian roberts prewitt
# do
#     CUDA_VISIBLE_DEVICES=1 python -W ignore run_FourierGrid.py --program train --config FourierGrid/configs/nerf_unbounded/room_single.py --num_per_block -1 --eval_ssim --eval_lpips_vgg --render_train --render_test --render_video --exp_id 9 --edgeLoss --edgeType ${edgetype} --edgeTrainType EdgeWeight --edgeRenderWeight 0.15
# done

CUDA_VISIBLE_DEVICES=1 python -W ignore run_FourierGrid.py --program train --config FourierGrid/configs/nerf_unbounded/stump_single.py --num_per_block -1 --eval_ssim --eval_lpips_vgg --render_train --render_test --render_video --exp_id 10
for edgetype in canny sobel laplacian roberts prewitt
do
    CUDA_VISIBLE_DEVICES=1 python -W ignore run_FourierGrid.py --program train --config FourierGrid/configs/nerf_unbounded/stump_single.py --num_per_block -1 --eval_ssim --eval_lpips_vgg --render_train --render_test --render_video --exp_id 10 --edgeLoss --edgeType ${edgetype} --edgeTrainType EdgeWeight --edgeRenderWeight 0.15
done

# CUDA_VISIBLE_DEVICES=1 python -W ignore run_FourierGrid.py --program train --config FourierGrid/configs/nerf_unbounded/bonsai_single.py --num_per_block -1 --eval_ssim --eval_lpips_vgg --render_test --render_video --exp_id 3
# for edgetype in canny sobel laplacian roberts prewitt
# do
#     CUDA_VISIBLE_DEVICES=1 python -W ignore run_FourierGrid.py --program train --config FourierGrid/configs/nerf_unbounded/bonsai_single.py --num_per_block -1 --eval_ssim --eval_lpips_vgg --render_test --render_video --exp_id 3 --edgeLoss --edgeType ${edgetype} --edgeTrainType EdgeWeight --edgeRenderWeight 0.15
# done

# CUDA_VISIBLE_DEVICES=1 python -W ignore run_FourierGrid.py --program train --config FourierGrid/configs/nerf_unbounded/kitchen_single.py --num_per_block -1 --eval_ssim --eval_lpips_vgg --render_train --render_test --render_video --exp_id 2
# for edgetype in canny sobel laplacian roberts prewitt
# do
#     CUDA_VISIBLE_DEVICES=1 python -W ignore run_FourierGrid.py --program train --config FourierGrid/configs/nerf_unbounded/kitchen_single.py --num_per_block -1 --eval_ssim --eval_lpips_vgg --render_train --render_test --render_video --exp_id 2 --edgeLoss --edgeType ${edgetype} --edgeTrainType EdgeWeight --edgeRenderWeight 0.15
# done

# CUDA_VISIBLE_DEVICES=1 python -W ignore run_FourierGrid.py --program train --config FourierGrid/configs/nerf_unbounded/counter_single.py --num_per_block -1 --eval_ssim --eval_lpips_vgg --render_train --render_test --render_video --exp_id 2
# for edgetype in canny sobel laplacian roberts prewitt
# do
#     CUDA_VISIBLE_DEVICES=1 python -W ignore run_FourierGrid.py --program train --config FourierGrid/configs/nerf_unbounded/counter_single.py --num_per_block -1 --eval_ssim --eval_lpips_vgg --render_train --render_test --render_video --exp_id 2 --edgeLoss --edgeType ${edgetype} --edgeTrainType EdgeWeight --edgeRenderWeight 0.15
# done

# CUDA_VISIBLE_DEVICES=0 python -W ignore run_FourierGrid.py --program train --config FourierGrid/configs/nerf_unbounded/bicycle_single.py --num_per_block -1 --eval_ssim --eval_lpips_vgg --render_train --render_test --render_video --exp_id 11
# CUDA_VISIBLE_DEVICES=0 python -W ignore run_FourierGrid.py --program train --config FourierGrid/configs/nerf_unbounded/garden_single.py --num_per_block -1 --eval_ssim --eval_lpips_vgg --render_train --render_test --render_video --exp_id 2

# # # bounded tanks and temples
# CUDA_VISIBLE_DEVICES=0 python -W ignore run_FourierGrid.py --program train --config FourierGrid/configs/tankstemple/caterpillar_single.py --num_per_block -1 --render_train --render_test --render_video --exp_id 4
# CUDA_VISIBLE_DEVICES=0 python -W ignore run_FourierGrid.py --program train --config FourierGrid/configs/tankstemple/barn_single.py --num_per_block -1 --render_train --render_test --render_video --exp_id 1
# CUDA_VISIBLE_DEVICES=0 python -W ignore run_FourierGrid.py --program train --config FourierGrid/configs/tankstemple/family_single.py --num_per_block -1 --render_train --render_test --render_video --exp_id 1


# # original DVGOv2 training
# # python -W ignore run_FourierGrid.py --program train --config FourierGrid/configs/waymo/block_0_tt.py
# # -----------------------------------------------------------------------
# # building
# CUDA_VISIBLE_DEVICES=0 python -W ignore run_FourierGrid.py --program train --config FourierGrid/configs/mega/building_no_block.py --sample_num 300 --render_train --render_video --exp_id 50
# CUDA_VISIBLE_DEVICES=0 python -W ignore run_FourierGrid.py --program render --config FourierGrid/configs/mega/building.py --sample_num 100 --render_train --exp_id 12 --num_per_block 5  # for render
# # rubble
# CUDA_VISIBLE_DEVICES=0 python -W ignore run_FourierGrid.py --program train --config FourierGrid/configs/mega/rubble.py --sample_num 100 --render_train --num_per_block 5 --exp_id 4  # for train
# CUDA_VISIBLE_DEVICES=0 python -W ignore run_FourierGrid.py --program train --config FourierGrid/configs/mega/rubble.py --sample_num 10 --render_train --num_per_block 5 --exp_id 4  # for debug
# CUDA_VISIBLE_DEVICES=0 python -W ignore run_FourierGrid.py --program render --config FourierGrid/configs/mega/rubble.py --sample_num 100 --render_train --num_per_block 5 --exp_id 4  # for render
# # quad
# CUDA_VISIBLE_DEVICES=0 python -W ignore run_FourierGrid.py --program train --config FourierGrid/configs/mega/quad.py --sample_num 100 --render_train --num_per_block 5 --exp_id 1
# # lego
# CUDA_VISIBLE_DEVICES=0 python -W ignore run_FourierGrid.py --program train --config FourierGrid/configs/nerf/lego.py --render_test --eval_ssim --eval_lpips_vgg --exp_id 8

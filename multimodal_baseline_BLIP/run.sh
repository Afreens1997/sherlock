#!/bin/bash
#SBATCH --job-name=blurBLIP
#SBATCH --error=blurBLIP.err
#SBATCH --output=blurBLIP.out
#SBATCH --mem=60G
#SBATCH --gres=gpu:A6000:2
#SBATCH --time=1-00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=svarna@andrew.cmu.edu
torchrun --nproc_per_node 1 --master_port 25678 BLIPBlurred.py ../../../../data/tir/projects/tir4/users/svarna/Sherlock/data/subset_train.json \
       ../../../../data/tir/projects/tir4/users/svarna/Sherlock/data/subset_train.json \
       ../../../../data/tir/projects/tir7/user_data/vishwavs/mmml_2024/blurred_images \
       --workers_dataloader 16 \
       --lr .00001 \
       --batch_size 64 \
       --n_epochs 2 \

# torchrun --nproc_per_node 1 --master_port 25678 BLIPGlobal.py ../../../../data/tir/projects/tir4/users/svarna/Sherlock/data/subset_train.json \
#        ../../../../data/tir/projects/tir4/users/svarna/Sherlock/data/multimodal_baselines/inferences/globalCapBLIP_train.json \
#        --vg_dir ../../../../data/tir/projects/tir4/users/svarna/Sherlock/data/VG \
#        --vcr_dir ../../../../data/tir/projects/tir4/users/svarna/Sherlock/data/VCR

# python BLIPZeroShot.py

# torchrun --nproc_per_node 1 --master_port 25675 BLIPHeatmap.py

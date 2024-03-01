#!/bin/bash
#SBATCH --job-name=inference_no_bbox
#SBATCH --error=inference_no_bbox.err
#SBATCH --output=inference_no_bbox.out
#SBATCH --mem=128g
#SBATCH --gres=gpu:A6000:1
#SBATCH --time=1-00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=svarna@andrew.cmu.edu
# torchrun --nproc_per_node 1 --master_port 25678 finetuningBLIP.py ../../../../data/tir/projects/tir4/users/svarna/Sherlock/data/subset_train.json \
#        ../../../../data/tir/projects/tir4/users/svarna/Sherlock/data/subset_val.json \
#        --workers_dataloader 16 \
#        --lr .00001 \
#        --batch_size 64 \
#        --n_epochs 5 \
#        --vg_dir ../../../../data/tir/projects/tir4/users/svarna/Sherlock/data/VG \
#        --vcr_dir ../../../../data/tir/projects/tir4/users/svarna/Sherlock/data/VCR

torchrun --nproc_per_node 1 --master_port 25670 BLIPInference.py ../../../../data/tir/projects/tir4/users/svarna/Sherlock/data/subset_test.json \
       ../../../../data/tir/projects/tir4/users/svarna/Sherlock/data/multimodal_baselines/inferences/BLIP_no_bbox.json \
       --vg_dir ../../../../data/tir/projects/tir4/users/svarna/Sherlock/data/VG \
       --vcr_dir ../../../../data/tir/projects/tir4/users/svarna/Sherlock/data/VCR

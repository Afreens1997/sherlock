#!/bin/bash
#SBATCH --job-name=exp
#SBATCH --error=exp.err
#SBATCH --mem=128g
#SBATCH --gres=gpu:A6000:4
#SBATCH --time=1-00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=svarna@andrew.cmu.edu
torchrun --nproc_per_node 1 sherlock.py ../../../../data/tir/projects/tir4/users/svarna/Sherlock/data/subset_train.json ../../../../data/tir/projects/tir4/users/svarna/Sherlock/data/subset_val.json \
       --workers_dataloader 16 \
       --lr .00001 \
       --batch_size 64 \
       --warmup 1000 \
       --n_epochs 5 --hide_true_bbox 8 --widescreen_processing 1 \
       --output_dir clip_exp \
       --vg_dir ../../../../data/tir/projects/tir4/users/svarna/Sherlock/data/VG \
       --vcr_dir ../../../../data/tir/projects/tir4/users/svarna/Sherlock/data/VCR

#!/bin/bash
#SBATCH --job-name=Sherlock
#SBATCH --output=baseline_2.out
#SBATCH --error=baseline_2.err
#SBATCH --mem=128g
#SBATCH --gres=gpu:L40:4
#SBATCH --time=1-00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=svarna@andrew.cmu.edu
torchrun --nproc_per_node 1 sherlock.py ../../../../data/tir/projects/tir4/users/svarna/Sherlock/data/train.json ../../../../data/tir/projects/tir4/users/svarna/Sherlock/data/val.json --workers_dataloader 16 --clip_model RN50x64 \
       --lr .00001 \
       --batch_size 64 \
       --warmup 1000 \
       --n_epochs 5 --hide_true_bbox 8 --widescreen_processing 1 \
       --output_dir clip_multitask \
       --vg_dir ../../../../data/tir/projects/tir4/users/svarna/Sherlock/data/VG
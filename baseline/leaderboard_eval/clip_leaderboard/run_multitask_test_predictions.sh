#!/bin/bash
#SBATCH --job-name=test
#SBATCH --error=test.err
#SBATCH --mem=128g
#SBATCH --gres=gpu:A6000:4
#SBATCH --time=1-00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=svarna@andrew.cmu.edu

torchrun --nproc_per_node 1 predict_clip_leaderboard.py \
	../../../../../../data/tir/projects/tir4/users/svarna/Sherlock/data/subset_test.json \
	../../clip_exp/model=ViT-B16~batch=64~warmup=1000~lr=1e-05~valloss=1.4708~randomclueinfhighlightbbox~cluewithprefix~widescreen.pt \
	test_set_predictions/retrieval.npy \
	--vg_dir ../../../../../../data/tir/projects/tir4/users/svarna/Sherlock/data/VG \
       --vcr_dir ../../../../../../data/tir/projects/tir4/users/svarna/Sherlock/data/VCR \
	--clip_model ViT-B/16 \
	--hide_true_bbox 8 \
	--workers_dataloader 8 \
	--task retrieval

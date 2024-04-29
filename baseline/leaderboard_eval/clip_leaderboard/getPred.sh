#!/bin/bash
#SBATCH --job-name=getPred
#SBATCH --output=getPred.out
#SBATCH --error=getPred.err
#SBATCH --mem=64g
#SBATCH --gres=gpu:4
#SBATCH --time=1-00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=svarna@andrew.cmu.edu

for split in {0..22};
do
    torchrun --nproc_per_node 1 predict_clip_leaderboard.py \
	../../../../../../data/tir/projects/tir4/users/svarna/Sherlock/data/val_retrieval/val_retrieval_$split\_instances.json \
	../../clip_exp/model=ViT-B16~batch=64~warmup=1000~lr=1e-05~valloss=0.8324~clueasimage~widescreen.pt \
	test_set_predictions/retrieval_$split\.npy \
	--vg_dir ../../../../../../data/tir/projects/tir4/users/svarna/Sherlock/data/VG \
    --vcr_dir ../../../../../../data/tir/projects/tir4/users/svarna/Sherlock/data/VCR \
	--clip_model ViT-B/16 \
	--hide_true_bbox 2 \
	--workers_dataloader 8 \
	--task retrieval
done
#!/bin/bash
#SBATCH --job-name=score
#SBATCH --output=score.out
#SBATCH --error=score.err
#SBATCH --mem=128g
#SBATCH --gres=gpu:A6000:1
#SBATCH --time=1-00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=svarna@andrew.cmu.edu

for i in {0..22}; do echo $i; python score_retrieval.py clip_leaderboard/test_set_predictions/retrieval_$i\.npy ../../../../../data/tir/projects/tir4/users/svarna/Sherlock/data/val_retrieval/val_retrieval_$i\_answer_key.json --instance_ids ../../../../../data/tir/projects/tir4/users/svarna/Sherlock/data/val_retrieval/val_retrieval_$i\_instance_ids.json; done;
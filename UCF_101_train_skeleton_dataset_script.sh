#!/bin/bash
#SBATCH --time=0-08:00:00
#SBATCH --mem=2000M
#SBATCH --gres=gpu:2
#SBATCH --constraint=volta

CODE_DIR=$(pwd)
echo "CODE_DIR= $CODE_DIR"

srun python $CODE_DIR/mmskl.py configs/recognition/st_gcn/dataset_example/train.yaml --gpus 2 --batch_size 6 --resume_from ./work_dir/recognition/st_gcn/UCF101_skeleton_dataset_train/latest.pth

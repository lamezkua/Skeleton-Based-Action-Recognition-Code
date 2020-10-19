#!/bin/bash
#SBATCH --time=0-00:30:00
#SBATCH --mem=2000M
#SBATCH --gres=gpu:1
#SBATCH --constraint=volta

CODE_DIR=$(pwd)
echo "CODE_DIR= $CODE_DIR"

srun python $CODE_DIR/mmskl.py configs/recognition_demo/recognition_demo.yaml --gpus 1 --recognition_checkpoint ./work_dir/recognition/st_gcn/UCF101_skeleton_dataset_train/latest.pth

#!/bin/bash
#SBATCH --time=0-08:00:00
#SBATCH --mem=2000M
#SBATCH --gres=gpu:1
#SBATCH --constraint=volta

mkdir /tmp/$SLURM_JOB_ID

cp UCF_101_mp4_train.zip /tmp/$SLURM_JOB_ID
cp UCF_category_annotations_train.json /tmp/$SLURM_JOB_ID

CODE_DIR=$(pwd)
echo "CODE_DIR= $CODE_DIR"

cd /tmp/$SLURM_JOB_ID
pwd

unzip UCF_101_mp4_train.zip

cd /scratch/work/amezcul1/RPinMLDS/mmskeleton_orig
pwd
echo $PATH

srun python $CODE_DIR/mmskl.py configs/utils/build_dataset_example.yaml --video_dir /tmp/$SLURM_JOB_ID/UCF_101_mp4_train --category_annotation /tmp/$SLURM_JOB_ID/UCF_category_annotations_train.json --out_dir $CODE_DIR/skeleton_dataset_train

#!/bin/bash
#$ -cwd
#$ -j y
#$ -l h_rt=240:0:0     
#$ -l h_vmem=11G      
#$ -pe smp 8
#$ -l gpu=1
#$ -l gpu_type=ampere

module load miniconda
conda activate ExpCLIP

python train.py \
--workers 8 \
--epochs 5 \
--batch-size 512 \
--batch-size-test-image 512 \
--batch-size-test-video 64 \
--lr 1e-3 \
--weight-decay 1e-4 \
--momentum 0.9 \
--print-freq 10 \
--milestones 30 \
--seed 1 \
--job-id $JOB_ID \
--instruction "Please play the role of a facial action describer. Objectively describe the detailed facial actions of the person in the image." \
--load-model "CLIP_L14" \

python test.py \
--load-model "CLIP_L14" \
--job-id $JOB_ID \
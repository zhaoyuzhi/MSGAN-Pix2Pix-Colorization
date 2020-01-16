#!/bin/bash
#MV2_USE_CUDA=1 MV2_ENABLE_AFFINITY=0 MV2_SMP_USE_CMA=0
#SBATCH -p Pixel2
#SBATCH --gres=gpu:1
#SBATCH --nodelist=HK-IDC2-10-1-75-61
#SBATCH --job-name=pre_train
#SBATCH -o ./logs/%j.txt
srun --mpi=pmi2 sh run.sh

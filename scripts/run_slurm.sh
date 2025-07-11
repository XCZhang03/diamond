#!/bin/bash

#SBATCH --job-name=diamond
#SBATCH -p gpu_requeue
#SBATCH --mem=50G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=504985967@qq.com
#SBATCH -o status/myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e status/myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --open-mode=append          # Append to the output and error files
#SBATCH --nodes=1                   # number of nodes
#SBATCH --ntasks-per-node=1         # number of MP tasks
#SBATCH --cpus-per-task=16           # number of CPU cores per task
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1                # number of GPUs per node
#SBATCH -t 3-00:00                  # maximum execution time (HH:MM:SS)
#SBATCH --contiguous
#SBATCH --account=ydu_lab



######################
### Set enviroment ###
######################
source /n/holylabs/ydu_lab/Lab/zhangxiangcheng/miniconda3/etc/profile.d/conda.sh
conda activate diamond
work_dir=/n/holylabs/ydu_lab/Lab/zhangxiangcheng/code/diamond
cd $work_dir
######################

srun bash scripts/resume.sh 


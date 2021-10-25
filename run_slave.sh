#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task 4
#SBATCH -p qTRDGPU

#SBATCH --gres=gpu:gforce:1
#SBATCH -A PSYC0002
#SBATCH --oversubscribe
#SBATCH --exclude trendsagn001.rs.gsu.edu
master=$1
rank=$2
num_nodes=$3
lr=$4
batch_size=$5
epochs=$6
name=$7
distributed_mode=$8
model=$9
dataset="${10}"
num_folds="${11}"
k="${12}"
backend="${13}"

eval "$(conda shell.bash hook)"
conda activate catalyst
cd /data/users2/bbaker/projects/dist_autodiff
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/users2/bbaker/bin/miniconda3/lib
PYTHONPATH=. python distributed_auto_differentiation/experiment.py --rank $rank --num-nodes $num_nodes --backend $backend --num-nodes $num_nodes --lr $lr --batch-size $batch_size --epochs $epochs --name $name --distributed-mode $distributed_mode --num-folds $num_folds --model $model --dataset $dataset --k $k --backend $backend


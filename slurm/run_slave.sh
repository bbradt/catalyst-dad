#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task 4
#SBATCH -p qTRDGPU

#SBATCH --gres=gpu:gforce:1
#SBATCH -A PSYC0002
#SBATCH --oversubscribe
#SBATCH --exclude=trendsagn001.rs.gsu.edu
master=$1
rank=$2
num_nodes=$3
additional_args=`echo "${@:4}" | xargs`
echo additional_args $additional_args
eval "$(conda shell.bash hook)"
conda activate catalyst
cd /data/users2/bbaker/projects/dist_autodiff
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/users2/bbaker/bin/miniconda3/lib
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python distributed_auto_differentiation/experiment.py --dist-url tcp://${master}:8998 --master-port 8998 --master-addr ${master} --rank $rank --num-nodes $num_nodes $additional_args


#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task 8
#SBATCH --mem=32GB
#SBATCH -p qTRDGPU
#SBATCH --gres=gpu:1
#SBATCH -A psy53c17
#SBATCH --oversubscribe
#SBATCH --exclude=arctrdagn015,arctrdagn008,arctrdagn009,arctrdagn010
#SBATCH --time=120:00:00
echo In Master
master=`hostname -I | cut -f 1 -d " "`
foo=$1
rank=$2
num_nodes=$3
PORT=$4
additional_args=""
j=0
t=2
j=0
for i in $@; do
     #echo Arg $j $i
     #echo $j > $t
     if (( j > 3 )); then
        additional_args="$additional_args$i "
        #echo additional_args $additional_args
     fi 
     j=$((j + 1))
done
echo additional_args $additional_args
eval "$(conda shell.bash hook)"
conda activate catalyst3.9
cd /data/users2/bbaker/projects/dist_autodiff
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/users2/bbaker43/anaconda3/lib
echo Host name is 
hostname
echo Port is $PORT
start=`date +%s`
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python distributed_auto_differentiation/experiment.py --dist-url tcp://${master}:$PORT --master-port $PORT --master-addr ${master} --rank $rank --num-nodes $num_nodes $additional_args
end=`date +%s`
runtime=$((end-start))
echo Total runtime on master was $runtime
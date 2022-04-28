#!/bin/bash
timestamp=$(date +%s)
name="test_pi_epsilon"
rm -r logs/${name}* -v
name="${name}_${timestamp}"
backend="gloo"
batch=64
epochs=50
kwargs="{}"
fold=5
for k in {0..5}
do
echo fold $k
bash slurm/run_slurm_batch.sh $name 4 "--lr 0.0001 --batch-size 64 --epochs $epochs --distributed-mode rankdad --model vit --dataset cifar10 --num-folds 5 --k $k --backend $backend --pi-use-sigma 1"
bash slurm/run_slurm_batch.sh $name 4 "--lr 0.0001 --batch-size 64 --epochs $epochs --distributed-mode rankdad --model vit --dataset cifar10 --num-folds 5 --k $k --backend $backend --pi-use-sigma 0"
done
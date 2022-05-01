#rm -r logs/cifar10_baseline* -v
name="powerSGD-baseline"
backend="gloo"
batch=64
epochs=2
kwargs="\"--lr 0.001 --batch-size 64 --epochs 10 --distributed-mode powersgd --dataset mnist --model mnistnet --backend gloo --N -1 --k 0\""
fold=5
for k in {1..5}
do
echo fold $k
bash slurm/run_slurm_batch.sh $name 4 "$kwargs"
#break
done
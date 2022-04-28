#rm -r logs/cifar10_baseline* -v
name="hcpgender_baseline"
backend="gloo"
batch=64
epochs=2
kwargs="\"--lr 0.001 --batch-size 31 --epochs 101 --distributed-mode rankdad --dataset hcp_gender --model icalstm --backend gloo --N -1 --k 0\""
fold=5
for k in {0..5}
do
echo fold $k
bash slurm/run_slurm_batch.sh $name 4 "$kwargs"
break
done
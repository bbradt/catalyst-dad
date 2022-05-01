#rm -r logs/cifar10_baseline* -v
name="hcpgender_pooled"
backend="nccl"
#batch=32
#epochs=31

fold=5
for k in {0..9}
do
kwargs="--lr 0.001 --batch-size 32 --epochs 31 --distributed-mode rankdad_ar --dataset hcp_gender --model icalstm --backend nccl --N -1 --seed 314159 --pi-numiterations 1 --pi-effective-rank 20"
kwargs="$kwargs"" --k "$k
echo fold $k
bash slurm/run_slurm_batch.sh $name 1 "\"""$kwargs""\""
#break
done
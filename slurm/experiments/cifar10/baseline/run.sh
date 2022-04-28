#rm -r logs/cifar10_baseline* -v
name="cifar10_baseline"
backend="nccl"
batch=64
epochs=2
kwargs="{}"
fold=5
for k in {0..5}
do
echo fold $k
bash slurm/run_slurm_batch.sh $name 4 0.0001 64 $epochs rankdad vit cifar10 10 $k $backend -1 $kwargs
bash slurm/run_slurm_batch.sh $name 4 0.0001 64 $epochs rankdad_ar vit cifar10 10 $k $backend -1 $kwargs
break
done
rm -r logs/cifar10_more_lr* -v
name="cifar10_more_lr"
epochs=50
batch=64
backend="gloo"
sites=4
lrs=( 0.0000001 0.000001 0.00001 0.0001 0.001 0.01 )

for k in {0..5}
do
for lr in "${lrs[@]}"
do
echo fold $k
#k=0
bash slurm/run_slurm_batch.sh $name $sites $lr $batch $epochs dsgd vit cifar10 5 $k $backend -1 "{}"
#bash slurm/run_slurm_batch.sh $name $sites $lr $batch $epochs rankdad vit cifar10 5 $k $backend -1 "{}"
bash slurm/run_slurm_batch.sh $name $sites $lr $batch $epochs rankdad_ar vit cifar10 5 $k $backend -1 "{}"
bash slurm/run_slurm_batch.sh $name $sites $lr $batch $epochs topk vit cifar10 5 $k $backend -1 "{}"
#bash slurm/run_slurm_batch.sh $name $sites $lr $batch $epochs dad vit cifar10 5 $k $backend -1 "{}"
done
done
rm -r logs/cifar10_batch_size* -v
name="cifar10_batch_size"
epochs=10
batches=(8 16 32 64 128)
backend="gloo"
sites=4
lr=0.0001
for k in {0..5}
do
for batch in "${batches[@]}"
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
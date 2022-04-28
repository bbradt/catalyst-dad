name="cifar10_more_epochs"
epochs=50
batch=64
lr=0.0001
backend="gloo"
for ((sites = 4; sites <= 18 ; sites+=2)); 
do
for k in {0..5}
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
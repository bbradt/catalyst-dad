rm -r logs/tiny-imagenet_increase_sites* -v
name="tiny-imagenet_increase_sites"
epochs=10
batch=64
lr=0.0001
backend="gloo"
for k in {0..5}
do
for ((sites = 4; sites <= 18 ; sites+=2)); 
do
echo fold $k
#k=0
bash slurm/run_slurm_batch.sh $name $sites $lr $batch $epochs dsgd vit tiny-imagenet 5 $k $backend -1 "{}"
#bash slurm/run_slurm_batch.sh $name $sites $lr $batch $epochs rankdad vit tiny-imagenet 5 $k $backend -1 "{}"
bash slurm/run_slurm_batch.sh $name $sites $lr $batch $epochs rankdad_ar vit tiny-imagenet 5 $k $backend -1 "{}"
bash slurm/run_slurm_batch.sh $name $sites $lr $batch $epochs topk vit tiny-imagenet 5 $k $backend -1 "{}"
#bash slurm/run_slurm_batch.sh $name $sites $lr $batch $epochs dad vit tiny-imagenet 5 $k $backend -1 "{}"
done
done
rm -r logs/imagenet_increase_sites* -v
for ((sites = 2 ; sites <= 18 ; sites+=2)); 
do
for k in {0..5}
do
echo fold $k
#k=0
bash slurm/run_slurm_batch.sh imagenet_increase_sites $sites 0.0001 64 10 dsgd vit imagenet 5 $k gloo -1
bash slurm/run_slurm_batch.sh imagenet_increase_sites $sites 0.0001 64 10 dad vit imagenet 5 $k gloo -1
bash slurm/run_slurm_batch.sh imagenet_increase_sites $sites 0.0001 64 10 rankdad vit imagenet 5 $k gloo -1
bash slurm/run_slurm_batch.sh imagenet_increase_sites $sites 0.0001 64 10 topk vit imagenet 5 $k gloo -1
done
done
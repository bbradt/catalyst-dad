rm -r logs/imagenet_baseline* -v
for k in {0..10}
do
echo fold $k
#k=0
bash slurm/run_slurm_batch.sh imagenet_baseline 4 0.001 16 10 dsgd vit imagenet 10 $k gloo -1
bash slurm/run_slurm_batch.sh imagenet_baseline 4 0.001 16 10 dad vit imagenet 10 $k gloo -1
bash slurm/run_slurm_batch.sh imagenet_baseline 4 0.001 16 10 rankdad vit imagenet 10 $k gloo -1
done
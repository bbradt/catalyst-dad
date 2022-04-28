rm -r logs/tiny_imagenet_baseline* -v
for k in {0..10}
do
echo fold $k
#k=0
bash slurm/run_slurm_batch.sh tiny_imagenet_baseline 4 0.000001 8 10 dsgd vit tiny-imagenet 10 $k gloo -1
bash slurm/run_slurm_batch.sh tiny_imagenet_baseline 4 0.000001 8 10 dad vit tiny-imagenet 10 $k gloo -1
bash slurm/run_slurm_batch.sh tiny_imagenet_baseline 4 0.000001 8 10 rankdad vit tiny-imagenet 10 $k gloo -1
done
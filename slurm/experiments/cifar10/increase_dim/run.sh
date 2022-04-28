rm -r logs/cifar10_increase_dim* -v
sites=4
dims=( 1024 512 256 128 )
for k in {0..5}
do
for dim in "${dims[@]}"
do
echo "******************"
echo DIM $dim
echo FOLD $k
#k=0
bash slurm/run_slurm_batch.sh cifar10_increase_dim $sites 0.0001 64 10 dsgd vit$dim cifar10 5 $k gloo -1
bash slurm/run_slurm_batch.sh cifar10_increase_dim $sites 0.0001 64 10 dad vit$dim cifar10 5 $k gloo -1
bash slurm/run_slurm_batch.sh cifar10_increase_dim $sites 0.0001 64 10 rankdad vit$dim cifar10 5 $k gloo -1
bash slurm/run_slurm_batch.sh cifar10_increase_dim $sites 0.0001 64 10 topk vit$dim cifar10 5 $k gloo -1
echo "******************"
done
done
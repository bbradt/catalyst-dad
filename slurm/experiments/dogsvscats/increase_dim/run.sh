rm -r logs/catsvsdogs_increase_dim* -v
sites=16
dims=( 1024 512 256 128 )
for dim in "${dims[@]}"
do
for k in {0..5}
do
echo "******************"
echo DIM $dim
echo FOLD $k
#k=0
bash slurm/run_slurm_batch.sh catsvsdogs_increase_dim $sites 0.0001 64 10 dsgd vit$dim catsvsdogs 5 $k gloo -1
bash slurm/run_slurm_batch.sh catsvsdogs_increase_dim $sites 0.0001 64 10 dad vit$dim catsvsdogs 5 $k gloo -1
bash slurm/run_slurm_batch.sh catsvsdogs_increase_dim $sites 0.0001 64 10 rankdad vit$dim catsvsdogs 5 $k gloo -1
bash slurm/run_slurm_batch.sh catsvsdogs_increase_dim $sites 0.0001 64 10 topk vit$dim catsvsdogs 5 $k gloo -1
echo "******************"
done
done
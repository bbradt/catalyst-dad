rm -r logs/mnist_increase_sites* -v
for ((sites = 4 ; sites <= 14 ; sites+=2)); 
do
for k in {0..5}
do
echo fold $k
#k=0
bash slurm/run_slurm_batch.sh mnist_increase_sites $sites 0.001 64 10 dsgd mnistnet mnist 5 $k gloo -1 "{}"
bash slurm/run_slurm_batch.sh mnist_increase_sites $sites 0.001 64 10 topk mnistnet mnist 5 $k gloo -1 "{}"
bash slurm/run_slurm_batch.sh mnist_increase_sites $sites 0.001 64 10 rankdad mnistnet mnist 5 $k gloo -1 "{}"
done
done
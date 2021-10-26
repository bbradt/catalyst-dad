rm -r logs/mnist_increase_sites* -v
for ((sites = 2 ; sites <= 18 ; sites+=2)); 
do
for k in {0..10}
do
echo fold $k
#k=0
bash run_slurm_batch.sh mnist_increase_sites $sites 0.001 64 10 dsgd mnistnet mnist 10 $k gloo
bash run_slurm_batch.sh mnist_increase_sites $sites 0.001 64 10 dad mnistnet mnist 10 $k gloo
bash run_slurm_batch.sh mnist_increase_sites $sites 0.001 64 10 rankdad mnistnet mnist 10 $k gloo
done
done
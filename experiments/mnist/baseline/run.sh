for k in {0..10}
do
echo fold $k
bash run_slurm_batch.sh mnist_baseline 4 0.001 32 10 dsgd mnistnet mnist 10 $k gloo
bash run_slurm_batch.sh mnist_baseline 4 0.001 32 10 dad mnistnet mnist 10 $k gloo
bash run_slurm_batch.sh mnist_baseline 4 0.001 32 10 rankdad mnistnet mnist 10 $k gloo
done
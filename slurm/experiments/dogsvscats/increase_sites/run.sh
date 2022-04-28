rm -r logs/catsvsdogs_increase_sites* -v
for ((sites = 2 ; sites <= 18 ; sites+=2)); 
do
for k in {0..5}
do
echo fold $k
#k=0
bash slurm/run_slurm_batch.sh catsvsdogs_increase_sites $sites 0.0001 64 10 dsgd vit catsvsdogs 5 $k gloo -1
bash slurm/run_slurm_batch.sh catsvsdogs_increase_sites $sites 0.0001 64 10 dad vit catsvsdogs 5 $k gloo -1
bash slurm/run_slurm_batch.sh catsvsdogs_increase_sites $sites 0.0001 64 10 rankdad vit catsvsdogs 5 $k gloo -1
bash slurm/run_slurm_batch.sh catsvsdogs_increase_sites $sites 0.0001 64 10 topk vit catsvsdogs 5 $k gloo -1
done
done
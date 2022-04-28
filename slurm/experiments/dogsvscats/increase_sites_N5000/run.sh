rm -r logs/catsvsdogsN5000_increase_sites_* -v
for ((sites = 2 ; sites <= 18 ; sites+=2)); 
do
for k in {0..10}
do
echo fold $k
#k=0
bash run_slurm_batch.sh catsvsdogsN5000_increase_sites $sites 0.0001 64 10 dsgd vit catsvsdogs 10 $k gloo 5000
bash run_slurm_batch.sh catsvsdogsN5000_increase_sites $sites 0.0001 64 10 dad vit catsvsdogs 10 $k gloo 5000
bash run_slurm_batch.sh catsvsdogsN5000_increase_sites $sites 0.0001 64 10 rankdad vit catsvsdogs 10 $k gloo 5000
done
done
rm -r logs/catsvsdogsN5000_baseline_* -v
sites=8
for k in {0..10}
do
echo fold $k
#k=0
bash run_slurm_batch.sh catsvsdogsN5000_baseline $sites 0.0001 64 10 dsgd vit catsvsdogs 10 $k gloo 5000
bash run_slurm_batch.sh catsvsdogsN5000_baseline $sites 0.0001 64 10 dad vit catsvsdogs 10 $k gloo 5000
bash run_slurm_batch.sh catsvsdogsN5000_baseline $sites 0.0001 64 10 rankdad vit catsvsdogs 10 $k gloo 5000
done

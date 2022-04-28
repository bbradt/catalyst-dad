rm -r logs/catsvsdogs_baseline* -v
for k in {0..10}
do
echo fold $k
#k=0
#bash run_slurm_batch.sh catsvsdogs_baseline 4 0.001 32 10 dsgd vit catsvsdogs 10 $k gloo
#bash run_slurm_batch.sh catsvsdogs_baseline 4 0.001 32 10 dad vit catsvsdogs 10 $k gloo
#bash run_slurm_batch.sh catsvsdogs_baseline 4 0.001 32 10 rankdad vit catsvsdogs 10 $k gloo
bash slurm/run_slurm_batch.sh catsvsdogs_baseline 4 0.001 32 10 topk vit catsvsdogs 10 $k gloo
done
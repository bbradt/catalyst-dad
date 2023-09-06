# These are all the parameters which can get looped over
expname="dAD-ReTesting-MLP"
ks=( 0 1 2 3 4 )
backends=( "gloo" )
modes=( "dsgd" "rankdad" "powersgd" "rankdad_oneway" "dad" )
#modes=( "powersgd" )
datasets=( "mnist" "fmnist" "fsl_control" )
#datasets=( "mnist" )
models=( "fsnet" )
lrs=( "1e-3" )
batches=( 32 )
epochs=( 10 )
ranks=( 2 4 8 16 32 )
hidden_dims=( "list([512,256,128,64,32,16])" )
iters=( "1" )
tols=( "0.0001" )
# Sites is ALWAYS the last variable iterated over for slurm purposes
sites=( 2 4 8 )
g=0
echo ***hidden_dims $hidden_dims
for site in "${sites[@]}"
do
for k in "${ks[@]}" # Cross-Validation
do
for backend in "${backends[@]}" # Distributed Backend 
do
for dataset in "${datasets[@]}" # Dataset
do
for model in "${models[@]}" # Model
do
for mode in "${modes[@]}" # Distributed Method
do
for lr in "${lrs[@]}" # Learning Rate
do
for batch in "${batches[@]}" # Batch Size
do
for epoch in "${epochs[@]}" # Epochs
do
for hidden_dim in "${hidden_dims[@]}" # Hidden Dims
do
for rank in "${ranks[@]}"
do
for iter in "${iters[@]}"
do
for tol in "${tols[@]}"
do
echo hidden_dim $hidden_dim
kwargs=""
kwargs+="--k "$k
kwargs+=" --name "$expname
kwargs+=" --backend "$backend
kwargs+=" --dataset "$dataset
kwargs+=" --model "$model
kwargs+=" --distributed-mode "$mode
kwargs+=" --lr "$lr
kwargs+=" --batch-size "$batch
kwargs+=" --epochs "$epoch
kwargs+=" --model-hidden-dims "$hidden_dim
kwargs+=" --pi-effective-rank "$rank" --psgd-rank "$rank
kwargs+=" --pi-numiterations "$iter
kwargs+=" --pi-tolerance "$tol
name=$expname"-"$mode"-k"$k"-d"$dataset
echo Name - $name
echo KWARGS - $kwargs
port=$((1234+g))
bash slurm/run_slurm_batch.sh $name $site $port "\"""$kwargs""\""
g=$((g+1))
#break
done
#break
done
#break
done
#break
done 
#break
done
#break
done
#break
done
#break
done
#break
done
#break
done
#break
done
#break
done
#break
done

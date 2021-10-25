name=$1
num_nodes=$2
lr=$3
batch_size=$4
epochs=$5
distributed_mode=$6
model=$7
dataset="${8}"
num_folds="${9}"
k="${10}"
backend="${11}"
real_nsites=$((num_nodes - 1))


for i in $(seq 0 $real_nsites); 
do
logname=$name"_d-"$dataset"_m-"$model"_dm-"$distributed_mode"_s-"${num_nodes}"_bs-"${batch_size}"_k-"${k}"_be-"${backend}
if [ $i -eq 0 ]
then

Submit_Output="$(sbatch -J ${logname} -o slurmlogs/master-${logname}.log -e slurmlogs/master-${logname}.err run_master.sh NA 0 ${num_nodes} ${lr} ${batch_size} ${epochs} ${logname} ${distributed_mode} ${model} ${dataset} ${num_folds} ${k} ${backend} >&1)"
echo Submit_Output $Submit_Output
JobId=`echo $Submit_Output | grep 'Submitted batch job' | awk '{print $4}'`
echo JobId $JobId
#sleep 60
Host=`scontrol show job ${JobId} | grep ' NodeList' | awk -F'=' '{print $2}' | nslookup | grep 'Address: ' | awk -F': ' '{print $2}'`
echo "waiting on host..."
while [ -z "$Host" ];
do 
    sleep 1;
    Host=`scontrol show job ${JobId} | grep ' NodeList' | awk -F'=' '{print $2}' | nslookup | grep 'Address: ' | awk -F': ' '{print $2}'`
done
echo host $Host
else
echo slave! $i
sbatch -J ${logname} -o slurmlogs/slave-$i-${logname}.log -e slurmlogs/slave-$i-${logname}.err run_slave.sh $Host $i ${num_nodes} ${lr} ${batch_size} ${epochs} ${logname} ${distributed_mode} ${model} ${dataset} ${num_folds} ${k} ${backend} >&1
fi
done

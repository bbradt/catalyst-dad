name=$1
num_nodes=$2
kwargs=$3
real_nsites=$((num_nodes - 1))

timestamp=$(date +%s)
for i in $(seq 0 $real_nsites); 
do

if [ $i -eq 0 ]
then

Submit_Output="$(sbatch -J ${name} -o slurm/logs/${name}-${timestamp}-master.log -e slurm/logs/${name}-${timestamp}-master.err slurm/run_master.sh NA 0 ${num_nodes} ${kwargs}>&1)"
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
sbatch -J ${name} -o slurm/logs/${name}-${timestamp}-slave-$i.log -e slurm/logs/${name}-${timestamp}-slave-$i.err slurm/run_slave.sh $Host $i ${num_nodes} ${kwargs}>&1
fi
done

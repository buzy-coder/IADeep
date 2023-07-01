# Description: delete all pods of IADeep 
scheduler=$1
if [ "$scheduler" != "" ]; then
    echo "Using ${scheduler} scheduler"
    sed -i "s/ENV SCHEDULER=.*/ENV SCHEDULER=${scheduler}/" $(pwd)/iadeep-scheduler-extender/Dockerfile
fi

kubectl delete pods --all -n iadeep
kubectl delete -f iadeep-local-coordinator/iadeep-local-coordinator.yaml
kubectl delete -f iadeep-local-coordinator/iadeep-local-coordinator-rbac.yaml
kubectl delete -f iadeep-tuner/iadeep-tuner-ds.yaml
kubectl delete -f iadeep-device-plugin/iadeep-device-plugin-ds.yaml
kubectl delete -f iadeep-device-plugin/iadeep-device-plugin-rbac.yaml
sleep 10
bash del-etcd.sh

cur_path="$PWD"
echo "current path is: " $cur_path
# Description: start all components of IADeep
cd $cur_path/benchmarks
bash build_image.sh
cd $cur_path/iadeep-device-plugin
bash build_image.sh
sleep 10
kubectl apply -f .

if [ "$scheduler" == "IADEEP" ]; then
    echo $scheduler
    cd $cur_path/iadeep-local-coordinator
    bash build_image.sh
    kubectl apply -f . 

    cd $cur_path/iadeep-tuner
    bash build_image.sh
    kubectl apply -f .
fi

cd $cur_path/iadeep-scheduler-extender
bash build_image.sh
cd $cur_path/iadeep-scheduler-extender/config
bash deploy-scheduler.sh

sleep 5

# monitor GPU on each worker node
# cd ../
# server=("cc232")
# for i in "${server[@]}"
# do
#     echo "Stop monitoring GPU on $i"
#     ssh -T wychen@$i < stop_monitor_gpu.sh
#     echo "Monitoring GPU on $i"
#     ssh -T wychen@$i < monitor_gpu.sh & 
# done

# submit jobs
# cd $cur_path/microsoft-job-generator
cd $cur_path
python3 microsoft-job-generator/submit_tasks.py --jobs=300
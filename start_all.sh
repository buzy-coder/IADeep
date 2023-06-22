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

# Description: start all components of IADeep
cd iadeep-device-plugin
bash build_image.sh
sleep 10
kubectl apply -f .

if [ "$scheduler" == "IADEEP" ]; then
    cd ../iadeep-local-coordinator
    bash build_image.sh
    kubectl apply -f . 

    cd ../iadeep-tuner
    bash build_image.sh
    kubectl apply -f .
fi

cd ../iadeep-scheduler-extender
bash build_image.sh
cd config
bash deploy-scheduler.sh

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
# cd ../microsoft-job-generator
# python3 submit_tasks.py
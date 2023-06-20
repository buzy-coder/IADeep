# Description: delete all pods of IADeep 
kubectl delete pods --all -n iadeep
sleep 10
bash del-etcd.sh

# Description: start all components of IADeep
cd iadeep-device-plugin
kubectl delete -f iadeep-device-plugin-ds.yaml
kubectl delete -f iadeep-device-plugin-rbac.yaml 
bash build_image.sh
sleep 10
kubectl apply -f .

cd ../iadeep-local-coordinator
kubectl delete -f iadeep-local-coordinator.yaml
kubectl delete -f iadeep-local-coordinator-rbac.yaml
bash build_image.sh
sleep 10
kubectl apply -f .

cd ../iadeep-scheduler-extender
bash build_image.sh
cd config
bash deploy-scheduler.sh

cd ../iadeep-tuner
bash build_image.sh
kubectl delete -f iadeep-tuner.yaml
kubectl apply -f .

# monitor GPU on each worker node
cd ../
server=("cc232")
for i in "${server[@]}"
do
    echo "Stop monitoring GPU on $i"
    ssh -T wychen@$i < stop_monitor_gpu.sh
    echo "Monitoring GPU on $i"
    ssh -T wychen@$i < monitor_gpu.sh & 
done

# submit jobs
cd microsoft-job-generator
python3 submit_tasks.py
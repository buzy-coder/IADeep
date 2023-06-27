# Overview
A common strategy for improving resource efficiency in training deep learning applications entails multiple tasks on a single GPU. To mitagate the interference caused by multiplexing, existing approaches are mainly focus on kernel-level reordering of kernel operations and hardware-level limitations on GPU streaming multiprocessors and GPU memory. However, all of them are not satisfactorily in optimizing completion time of tasks that arrive online.

Now there is a middleware-level GPU multiplexing solution on native Kubernetes: it is based on scheduler extenders and device plugin mechanism, so you can reuse the solution easily in your own Kubernetes cluster. 

## 1. Dictionary Description
- benchmarks: deep learning tasks.
- iadeep_yaml: yaml files for submitting tasks.
- iadeep-device-plugin: expose the GPU Memory and GPU count on the node of your cluster.
- iadeep-local-coordinator: manager on every worker node for maintaining a tuning process on each GPU device.
- iadeep-scheduler-extender: online scheduler on master node.
- microsft-job-generator: generator for various deep learning tasks to the cluster.
- iadeep-tuner: tuner for task configuration selection.

## 2. Prerequisites
- Kubernetes 1.18+
- NVIDIA drivers ~= 470.57
- Nvidia-docker version > 2.0 (see how to [install](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html))
- Docker configured with NVIDIA as [default runtime](https://github.com/NVIDIA/nvidia-docker/wiki/Advanced-topics#default-runtime).
- CUDNN 7

## 3. Installation Guide
### 3.1 Prepare GPU Worker Node
This guide assumes that the NVIDIA drivers and nvidia-docker2 have been installed.
```shell
sudo apt-get install nvidia-container-runtime
```
Enable the Nvidia runtime as your default runtime on your worker node. To do this, please edit the docker daemon config file which is usually present at /etc/docker/daemon.json:
```shell
{
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "path": "/usr/bin/nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
```
Then restart the docker daemon:
```shell
sudo systemctl restart docker
```

- In addition, create namespace iadeep and label each GPU node with gpushare=true
```
kubectl create ns iadeep
kubectl label node ${worker1-name} gpushare=true
kubectl label node ${worker2-name} gpushare=true
kubectl label node ${worker3-name} gpushare=true
kubectl label node ${worker4-name} gpushare=true
```

- Importantly, edit leader-elect=true in /etc/kubernetes/manifests/kube-scheduler.yaml to leader-elect=false to make iadeep-scheduler work.
```shell
- --leader-elect=false
```

### 3.2 Clone Repository
```shell
git clone https://github.com/buzy-coder/IADeep
```
- Establish etcd connections with each module in IADeep
```shell
# modify etcd_server_ip and etcd_port environment variables in benchmarks/Dockerfile, iadeep-local-coordinator/Dockerfile, iadeep-scheduler-extender/Dockerfile and microsoft-job-generator/etcdctl.py  to your own etcd ip address and port, for example:
ETCD_SERVER_IP=172.168.0.1
ETCD_PORT=2379
```

### 3.3 Deploy GPU Scheduler Extender in Master Node
- build image
```shell
cd iadeep-scheduler-extender
bash build_image.sh
``` 
- deploy scheduler extender
```shell
cd config
bash deploy-scheduler.sh
```

### 3.4 Deploy Device Plugin 
- build image
```shell
cd iadeep-device-plugin
bash build_image.sh
```
- deploy DaemonSet
```shell
kubectl apply -f device-plugin-rbac.yaml
kubectl apply -f device-plugin-ds.yaml
```

### 3.5 Deploy local coordinator
- build image
```shell
cd iadeep-local-coordinator
bash build_image.sh
```

- deploy DaemonSet
```shell
kubectl apply -f iadeep-local-coordinator-rbac.yaml
kubectl apply -f iadeep-local-coordinator.yaml
```

### 3.6 Deploy Tuner 
- build image
```
cd iadeep-tuner
bash build_image.sh
```

- deploy DaemonSet
```shell
kubectl apply -f iadeep-tuner-ds.yaml
```

## 4. Submit Tasks
- Prepare training datasets
```shell
# download the datasets in your specfic dictionary
bash download_datasets.sh
# Use NFS to mount the dataset dictionary in each GPU node for deep learning tasks to train, you can refer the config_nfs_mount.sh script to configure this setting.
```
- build benchmarks image
```shell
bash build_image.sh
```
- submit deep learning tasks
```shell
cd microsoft-job-generartor
python3 submit_tasks.py --scale=1 --jobs=100
```

## 5. Use a script to deploy the system and run DL jobs (Optional)
```
sudo bash start_all.sh IADEEP
```

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
- tuner: tuner for task configuration selection.

## 2. Prerequisites
- Kubernetes 1.18+
- NVIDIA drivers ~= 470.57
- Nvidia-docker version > 2.0 (see how to [install](https://github.com/NVIDIA/nvidia-docker) and it's [prerequisites](https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)#prerequisites))
- Docker configured with NVIDIA as [default runtime](https://github.com/NVIDIA/nvidia-docker/wiki/Advanced-topics#default-runtime).
- CUDNN 7

## 3. Installation Guide
### 3.1 Prepare GPU Worker Node
This guide assumes that the NVIDIA drivers and nvidia-docker2 have been installed.

Enable the Nvidia runtime as your default runtime on your node. To do this, please edit the docker daemon config file which is usually present at /etc/docker/daemon.json:
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

### 3.2 Clone Repository
```shell
git clone https://github.com/buzy-coder/IADeep
```
- Establish etcd connections with each module in IADeep
```shell
# modify etcd_server_ip and etcd_port environment variables in Dockerfile to your own ip address and port, for example:
ETCD_SERVER_IP=172.168.0.1
ETCD_PORT=2379
```
- In addition, create namespace iadeep and label each GPU node with gpushare=true
```
kubectl create ns iadeep
kubectl label node worker-0 gpushare=true
```

### 3.3 Deploy GPU Scheduler Extender in Master Node
- build image
```shell
cd iadeep-gpu/iadeep-scheduler-extender
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
- Prepare training dataset
```shell
# download the datasets in your specfic dictionary
mkdir /nfs/dataset
bash iadeep-gpu/benchmarks/download_datasets.sh
# Use NFS to mount the dataset dictionary in each GPU node for deep learning tasks to train
```
- build benchmarks image
```shell
bash build_image.sh
```
- submit deep learning tasks
```shell
cd microsoft-job-generartor
python3 submit_tasks.py --scale=1
```

## 5. Use a script to deploy the system and run DL jobs
```
sudo bash start_all.sh IADEEP
```

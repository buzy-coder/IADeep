import os
import time
import argparse
import subprocess
from etcdctl import ETCD_WRAPER

os.environ['GRPC_ENABLE_FORK_SUPPORT']="1"
os.environ['GRPC_POLL_STRATEGY']="poll"

def run(args, pod_name):

    init_batchsize = args.batch_size
    job_name = args.job_name
    command = f"python /workspace/workloads/{job_name}.py --batch_size {init_batchsize} --pod_name {pod_name}"
    print(command)
    subprocess.run(command, shell=True, executable='bash')

if __name__ == "__main__":
    start = time.time()
    pod_name = os.environ.get("POD_NAME")
    # pod_name = "yolov5"
    etcd_wraper = ETCD_WRAPER() 
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--job_name', type=str, default="resnet50")
    parser.add_argument('--batch_size', type=int, default=512)
    args = parser.parse_args()
    etcd_wraper.put(pod_name, 'start_time', str(start))
    etcd_wraper.put(pod_name, 'complete', "0")
    etcd_wraper.put(pod_name, "init_batch_size", args.batch_size)

    run(args, pod_name) 

    end = time.time()
    etcd_wraper.put(pod_name, 'end_time', end)
    jct = end-start        
    etcd_wraper.put(pod_name, "jct", jct)
    print(f'{pod_name} jct is: {jct}.') 
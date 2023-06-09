import time, random, subprocess, argparse
import pandas as pd
from time import sleep
from etcdctl import ETCD_WRAPER, ROOT_PATH
from typing import Dict

random.seed(0)

YAML_FOLDER = f"{ROOT_PATH}iadeep_yaml"

def task_selection(task_cdf: Dict[str, float]):
    coin = random.random()
    for task in task_cdf:
        if coin < task_cdf[task]:
            return task


def submit_task(task):
    result = subprocess.Popen(
        f"kubectl create -f {YAML_FOLDER}/{task}.yaml",
        shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    submit_time = time.time()
    out = (result.stdout.read()).decode()
    pod_name = out.split('/')[1].split(" ")[0]
    print(pod_name)
    etcd_wraper.put(pod_name, 'submit_time', submit_time)
    return pod_name, submit_time 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--scheduler', type=str, default="IADeep",
                        help='Used scheduler of experiment')
    parser.add_argument('--fitter', type=str, default="gpucb",
                        help='Used fitting method of experiment')
    parser.add_argument('--scale', type=float, default=1.0,
                        help='Scaling of submition rate')
    parser.add_argument('--jobs', type=int, default=50,
                        help='Number of jobs to submit')
    opt = parser.parse_args()
    jobs = opt.jobs
    etcd_wraper = ETCD_WRAPER() 
    # Construct task cdf
    # tasks = ["vgg16", "resnet50", "squeezenet", "neumf", "lstm", "adgcl", "yolov5", "bert"]
    # prob = [0.14, 0.14, 0.14, 0.12, 0.12, 0.12, 0.1, 0.1]
    # tasks = ["vgg16", "resnet50", "squeezenet", "neumf", "lstm", "yolov5", "bert"]
    # prob = [0.14, 0.14, 0.14, 0.12, 0.12, 0.1, 0.1]
    tasks = ["vgg16", "resnet50", "squeezenet", "neumf", "lstm"]
    prob = [0.14, 0.14, 0.14, 0.12, 0.12]
    prob = [x / sum(prob) for x in prob]
    cdf, cum = [], 0
    for p in prob:
        cdf.append(cum + p)
        cum += p
    task_cdf = dict(zip(tasks, cdf))
    # Get and modify task submission rate
    init_rate = pd.read_csv(f"{ROOT_PATH}/microsoft-job-generator/microsoft_trace_task_rate.csv", nrows=jobs)
    submit_times = init_rate["modified_time"].tolist()
    prev_submit_time, submit_intervals = 0, []
    for submit_time in submit_times:
        submit_intervals.append(submit_time - prev_submit_time)
        prev_submit_time = submit_time
    submit_intervals = [x / 1000 * opt.scale for x in submit_intervals]
    # Start to submit tasks
    now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time())) 
    # csv_name = f"{opt.fitter}_{opt.scheduler}_{opt.scale}_{now}.csv"
    csv_name = f"{opt.scheduler}_{opt.scale}_{now}.csv"
    submissions = []
    for interval in submit_intervals:
        task = task_selection(task_cdf)
        print(f"Next submission: {interval}s later")
        sleep(interval)
        pod_name, submit_time = submit_task(task)
        submissions.append({"pod_name": pod_name, "submit_time": submit_time})
        pd.DataFrame(submissions).to_csv(csv_name, index=False)

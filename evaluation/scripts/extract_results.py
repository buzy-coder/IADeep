import os
import argparse
import pandas as pd
from etcdctl import ETCD_WRAPER


def extract_results_from_etcd(file_path, client):
    # Extract results from etcd
    pod_names, submit_times, start_times, end_times, jcts, schedule_times, tuning_times, tuning_rounds = [], [], [], [], [], [], [], []
    for pod_name in client.get_all_pod_names():
        # Extract results from etcd
        submit_time = client.get(pod_name, "submit_time")
        start_time = client.get(pod_name, "start_time")
        end_time = client.get(pod_name, "end_time")
        jct = end_time - start_time
        schedule_time = client.get(pod_name, "schedule_time")
        tuning_time = client.get(pod_name, "tuning_time")
        tuning_time = tuning_time if tuning_time is not None else 0
        search_rounds = client.get(pod_name, "search_rounds")
        search_rounds = search_rounds if search_rounds is not None else 0

        # Append results to lists
        pod_names.append(pod_name)
        submit_times.append(submit_time)
        start_times.append(start_time)
        end_times.append(end_time)
        jcts.append(jct)
        schedule_times.append(schedule_time)
        tuning_times.append(tuning_time)
        search_rounds.append(search_rounds)

    # Save results to csv file
    df = pd.DataFrame({"pod_name": pod_names, "submit_time": submit_times, "start_time": start_times, "end_time": end_times, "jct": jcts, "schedule_time": schedule_times, "tuning_time": tuning_times, "search_rounds": search_rounds})
    df.to_csv(os.path.join(file_path, "results.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, default=".", help="Figure number in the paper to plot.")
    args = parser.parse_args()

    client = ETCD_WRAPER()

    extract_results_from_etcd(args.file_path, client)
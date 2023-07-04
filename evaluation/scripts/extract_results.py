import os
import argparse
import pandas as pd
from etcdctl import ETCD_WRAPER


def extract_results_from_etcd(file_path, client):
    # Extract results from etcd
    df = pd.read_csv(file_path)
    pod_names = df["pod_name"].tolist()
    for pod_name in pod_names:
        # Extract results from etcd
        submit_time = client.get(pod_name, "submit_time")
        start_time = client.get(pod_name, "start_time")
        end_time = client.get(pod_name, "end_time")
        jct = end_time - start_time
        schedule_time = client.get(pod_name, "schedule_time")
        tuning_time = client.get(pod_name, "tuning_time")
        tuning_time = tuning_time if tuning_time is not None else 0
        search_round = client.get(pod_name, "search_round")
        search_round = search_round if search_round is not None else 0

        df.loc[df["pod_name"] == pod_name, "start_time"] = start_time
        df.loc[df["pod_name"] == pod_name, "end_time"] = end_time   
        df.loc[df["pod_name"] == pod_name, "jct"] = end_time - submit_time
        df.loc[df["pod_name"] == pod_name, "search_rounds"] = search_round
        df.loc[df["pod_name"] == pod_name, "schedule_time"] = schedule_time 

    df.to_csv(file_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, default=".", help="Figure number in the paper to plot.")
    args = parser.parse_args()

    client = ETCD_WRAPER()

    extract_results_from_etcd(args.file_path, client)
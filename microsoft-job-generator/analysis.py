import os
import re
import argparse
import pandas as pd
from statistics import mean
from etcdctl import ETCD_WRAPER

client = ETCD_WRAPER()

TARGET_DIR = "."
YELLO = "\033[0;33m"
GREEN = "\033[0;32m"
GRAY = "\033[0;90m"
RED = "\033[1;91m"
END = "\033[0m"

parser = argparse.ArgumentParser(description="manual to this script")
parser.add_argument("--dir", "-d", type=str, default=".")


def time_formatter(seconds):
    secs = seconds % 60
    mins = (seconds // 60) % 60
    hrs = seconds // 3600
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"


def printer(pod_name, final_sco, target_sco, jct, makespan, success):
    formatter = "%17s"

    if final_sco is None:
        final_sco = "N/A"
        formatter += "%15s"
    else:
        formatter += "%15.4g"

    if target_sco is None:
        target_sco = "N/A"
        formatter += "%15s"
    else:
        formatter += "%15.4g"

    if jct is None:
        jct = "N/A"
    else:
        jct = time_formatter(int(jct))

    if makespan is None:
        makespan = "N/A"
    else:
        makespan = time_formatter(int(makespan))

    if success == "Unknown":
        success = f"        {GRAY}?{END}"
    elif success == "Error":
        success = f"        {RED}‼{END}"
    elif success == "Success":
        success = f"        {GREEN}√{END}"
    else:
        success = f"        {YELLO}x{END}"

    formatter += "%15s%15s"

    print(
        "│"
        + formatter % (pod_name, final_sco, target_sco, jct, makespan)
        + f"{success}  │"
    )


file_names = [
    x
    for x in os.listdir(TARGET_DIR)
    if re.compile("^[a-zA-Z]+\_[a-zA-Z]+\_\d+\.?\d*\_.*\.csv").match(x)
]

data = {}
for file_name in file_names:
    file_path = f"{TARGET_DIR}/{file_name}"
    data_df = pd.read_csv(file_path)
    fitter, scheduler = file_name.split("_")[:2]
    jcts = []
    tuning_costs = []
    makespans = []
    train_results = []
    width = 90
    print("╭" + "─" * (width - 2) + "╮")
    print("│" + f"{file_path:^{width - 2}s}" + "│")
    print("├" + "─" * (width - 2) + "┤")
    print(
        "│"
        + ("%17s%15s%15s%15s%15s%10s")
        % ("Pod Name", "Final Score", "Target Score", "JCT", "Makespan", "Success")
        + " │"
    )
    for idx, row in data_df.iterrows():
        pod_name, submit_time = row
        jct = client.get(pod_name, "jct")
        end_time = client.get(pod_name, "end_time")
        train_success = client.get(pod_name, "train_success")
        final_sco = client.get(pod_name, "final_sco")
        target_sco = client.get(pod_name, "target_sco")
        tuning_cost = client.get(pod_name, "tuning_consumption")
        makespan = None

        if jct is not None:
            jcts.append(jct)
        if tuning_cost is not None:
            tuning_costs.append(tuning_cost)
        if end_time is not None:
            makespan = float(end_time) - float(submit_time)
            makespans.append(makespan)
        if train_success is None:
            if jct is not None:
                success = "Error"
            else:
                success = "Unknown"
        elif train_success == 1:
            success = "Success"
        else:
            success = "Fail"
        train_results.append(success)

        printer(pod_name, final_sco, target_sco, jct, makespan, success)

    print("├" + "╌" * (width - 2) + "┤")
    error_count, success_count, failed_count = 0, 0, 0
    for train_result in train_results:
        error_count += 1 if train_result == "Error" else 0
        success_count += 1 if train_result == "Success" else 0
        failed_count += 1 if train_result == "Fail" else 0
    result = {
        "jct": mean(jcts),
        "tuning_cost": mean(tuning_costs),
        "makespan": mean([x for x in makespans if x > 0]),
        "train_result": {
            "success": success_count,
            "failed": failed_count,
            "error": error_count,
            "finished": len([x for x in train_results if x != "Unknown"]),
        },
    }
    print("│" + f"{'Summary':^{width - 2}s}" + "│")
    print("├" + "╌" * (width - 2) + "┤")
    print(
        f"│ {'JCT':^12s}{'Makespan':^12s}{'Tuning Cost':^12s}"
        f"{'Success':^14s}{'Failed':^12s}{'Error':^12s}{'Finished':^12s} │"
    )
    print(
        f"│ {result['jct']:^12.4f}{result['makespan']:^12.4f}{result['tuning_cost']:^12.4f}"
        f"{result['train_result']['success']:^14d}{result['train_result']['failed']:^12d}"
        f"{result['train_result']['error']:^12d}{result['train_result']['finished']:^12d} │"
    )
    print("╰" + "─" * (width - 2) + "╯")
    if scheduler in data:
        data[scheduler][fitter] = result
    else:
        data[scheduler] = {fitter: result}

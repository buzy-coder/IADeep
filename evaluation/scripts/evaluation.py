import matplotlib.pyplot as plt
import numpy as np
import argparse
import pandas as pd
import seaborn as sns

# Set the font size of the plots
sns.set(font_scale=1.1, style="white")


# plot CT and makespan in bar with baselines (Antman, MPS, Kernel. Est)(Fig. 7a)
def plot_ct_and_makespan(iadeep_path, antman_path, mps_path, kernel_est_path):
    # Load results
    iadeep_df = pd.read_csv(iadeep_path)
    antman_df = pd.read_csv(antman_path)
    mps_df = pd.read_csv(mps_path)
    kernel_est_df = pd.read_csv(kernel_est_path)

    iadeep_ct = iadeep_df["jct"].mean()
    antman_ct = antman_df["jct"].mean()
    mps_ct = mps_df["jct"].mean()
    kernel_est_ct = kernel_est_df["jct"].mean()

    iadeep_makespan = iadeep_df["end_time"].max() - iadeep_df["start_time"].min()
    antman_makespan = antman_df["end_time"].max() - antman_df["start_time"].min()
    mps_makespan = mps_df["end_time"].max() - mps_df["start_time"].min()
    kernel_est_makespan = kernel_est_df["end_time"].max() - kernel_est_df["start_time"].min()

    iadeep_res = [iadeep_ct, iadeep_makespan]
    antman_res = [antman_ct, antman_makespan]
    mps_res = [mps_ct, mps_makespan]
    kernel_est_res = [kernel_est_ct, kernel_est_makespan]

    # performance comprision
    antman_res = [antman_res[i]/iadeep_res[i] for i in range(len(antman_res))]
    mps_res = [mps_res[i]/iadeep_res[i] for i in range(len(mps_res))]
    kernel_est_res = [kernel_est_res[i]/iadeep_res[i] for i in range(len(kernel_est_res))]
    iadeep_res = [1.0, 1.0]

    x_labels = ["CT", "Makespan"]
    x = np.arange(len(x_labels))
    width = 0.17

    fig = plt.figure(figsize=(3,2.3), dpi=120)
    ax = fig.add_subplot(111)

    ax.bar(x + width, antman_res, width, label='Antman')
    ax.bar(x + width*2, mps_res, width, label='MPS')
    ax.bar(x + width*3, kernel_est_res, width, label='Kernel Est.')
    ax.bar(x + width*4, iadeep_res, width, label='IADeep')

    for a,b in zip(x,antman_res):
        ax.text(a+width, b, str(round(b,2)), ha='center', va='bottom', fontsize=12, rotation=30)  
    for a,b in zip(x,mps_res):
        ax.text(a+width*2, b, str(round(b,2)), ha='center', va='bottom', fontsize=12, rotation=30)
    for a,b in zip(x,kernel_est_res):
        ax.text(a+width*3, b, str(round(b,2)), ha='center', va='bottom', fontsize=12, rotation=30)    
    for a,b in zip(x,iadeep_res):
        ax.text(a+width*4, b, str(round(b,2)), ha='center', va='bottom', fontsize=12, rotation=30)

    ax.set_xticks(x + width*2.5)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0, 3.5)
    ax.legend(loc="best", ncol=2, handlelength=1.6, handletextpad=0.5, columnspacing=0.6)
    plt.tight_layout(pad=0.0, h_pad=0, w_pad=0, rect=None)
    plt.savefig('../pdf/jct_and_makespan.pdf')    


# plot CDF of CT (Fig. 7b)
def plot_ct_cdf(iadeep_path, antman_path, mps_path, kernel_est_path):
    # Load results
    iadeep_df = pd.read_csv(iadeep_path)
    antman_df = pd.read_csv(antman_path)
    mps_df = pd.read_csv(mps_path)
    kernel_est_df = pd.read_csv(kernel_est_path)

    iadeep_ct = iadeep_df["jct"].values
    antman_ct = antman_df["jct"].values
    mps_ct = mps_df["jct"].values
    kernel_est_ct = kernel_est_df["jct"].values

    iadeep_ct = np.sort(iadeep_ct)
    antman_ct = np.sort(antman_ct)
    mps_ct = np.sort(mps_ct)
    kernel_est_ct = np.sort(kernel_est_ct)

    iadeep_ct_cdf = np.arange(len(iadeep_ct))/float(len(iadeep_ct))
    antman_ct_cdf = np.arange(len(antman_ct))/float(len(antman_ct))
    mps_ct_cdf = np.arange(len(mps_ct))/float(len(mps_ct))
    kernel_est_ct_cdf = np.arange(len(kernel_est_ct))/float(len(kernel_est_ct))

    fig = plt.figure(figsize=(3,2.3), dpi=120)
    ax = fig.add_subplot(111)
    ax.plot(antman_ct/3600, antman_ct_cdf, label="Antman")
    ax.plot(mps_ct/3600, mps_ct_cdf, label="MPS")
    ax.plot(kernel_est_ct/3600, kernel_est_ct_cdf, label="Kernel Est.")
    ax.plot(iadeep_ct/3600, iadeep_ct_cdf, label="IADeep")
    ax.set_xlabel("Norm. CT")
    ax.set_ylabel("CDF")
    # ax.set_xlim(0, 100)
    # ax.set_ylim(0, 1.05)
    ax.grid(linestyle='-.', linewidth=0.5, alpha=0.5)
    ax.legend(loc="best", ncol=1, handlelength=1.6, handletextpad=0.5, columnspacing=0.6)
    plt.tight_layout(pad=0.0, h_pad=0, w_pad=0, rect=None)
    plt.savefig('../pdf/jct_cdf.pdf')

    
# plot SM and Memory Utilization (Fig. 8)
def plot_sm_mem_util(iadeep_util_path, antman_util_path, mps_util_path, kernel_est_util_path):
    metrics = ["sm", "mem"]
    metrics_labels = ["SM Utilization", "MEM Utilization"]
    # Load results
    iadeep_util_df = pd.read_csv(iadeep_util_path)
    antman_util_df = pd.read_csv(antman_util_path)
    mps_util_df = pd.read_csv(mps_util_path)
    kernel_est_util_df = pd.read_csv(kernel_est_util_path)

    for metric in metrics:
        iadeep_util = iadeep_util_df[metric].values
        antman_util = antman_util_df[metric].values
        mps_util = mps_util_df[metric].values
        kernel_est_util = kernel_est_util_df[metric].values

        fig = plt.figure(figsize=(3,2.3), dpi=120)
        ax = fig.add_subplot(111)
        ax.plot(antman_util, label="Antman")
        ax.plot(mps_util, label="MPS")
        ax.plot(kernel_est_util, label="Kernel Est.")
        ax.plot(iadeep_util, label="IADeep")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(metrics_labels[metrics.index(metric)])
        
        ax.legend(loc="best", ncol=1, handlelength=1.6, handletextpad=0.5, columnspacing=0.6)
        plt.tight_layout(pad=0.0, h_pad=0, w_pad=0, rect=None)
        plt.savefig('../pdf/' + metric + '_util.pdf')

# plot CT with various task arrival rates (Fig. 10a)
def plot_ct_various_arrival_rates(iadeep_dir, antman_dir, mps_dir, kernel_est_dir):
    various_arrival_rates = [1,2,3,4]
    x_labels = ["1x", "2x", "3x", "4x"]
    x = np.arange(len(various_arrival_rates))
    fig = plt.figure(figsize=(3,2.3), dpi=120)
    
    # plot CT
    ax = fig.add_subplot(111)
    iadeep_ct, antman_ct, mps_ct, kernel_est_ct = [], [], [], []
    for arrival_rate in various_arrival_rates:
        iadeep_df = pd.read_csv(f'{iadeep_dir}iadeep_{arrival_rate}.csv')
        antman_df = pd.read_csv(f'{antman_dir}antman_{arrival_rate}.csv')
        mps_df = pd.read_csv(f'{mps_dir}mps_{arrival_rate}.csv')
        kernel_est_df = pd.read_csv(f'{kernel_est_dir}kernel_est_{arrival_rate}.csv')

        iadeep_ct.append(iadeep_df["jct"].mean())
        antman_ct.append(antman_df["jct"].mean())
        mps_ct.append(mps_df["jct"].mean())
        kernel_est_ct.append(kernel_est_df["jct"].mean())

    ax.plot(x, antman_ct, label="Antman", marker='d', markersize=5)
    ax.plot(x, mps_ct, label="MPS", marker='p', markersize=5)
    ax.plot(x, kernel_est_ct, label="Kernel Est.", marker='X', markersize=5)
    ax.plot(x, iadeep_ct, label="IADeep", marker='h', markersize=5)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Relative Task Load")
    ax.set_ylabel("Norm. CT") 
    ax.legend(loc="best", ncol=1, handlelength=1.6, handletextpad=0.5, columnspacing=0.6)
    plt.tight_layout(pad=0.0, h_pad=0, w_pad=0, rect=None)
    plt.savefig('../pdf/jct_various_arrival_rates.pdf')


# plot makespan with various task arrival rates (Fig. 10b)
def plot_makespan_various_arrival_rates(iadeep_dir, antman_dir, mps_dir, kernel_est_dir):
    various_arrival_rates = [1,2,3,4]
    x_labels = ["1x", "2x", "3x", "4x"]
    x = np.arange(len(various_arrival_rates))

    # plot makespan
    fig = plt.figure(figsize=(3,2.3), dpi=120)
    ax = fig.add_subplot(111)
    iadeep_makespan, antman_makespan, mps_makespan, kernel_est_makespan = [], [], [], []
    for arrival_rate in various_arrival_rates:
        iadeep_df = pd.read_csv(f'{iadeep_dir}iadeep_{arrival_rate}.csv')
        antman_df = pd.read_csv(f'{antman_dir}antman_{arrival_rate}.csv')
        mps_df = pd.read_csv(f'{mps_dir}mps_{arrival_rate}.csv')
        kernel_est_df = pd.read_csv(f'{kernel_est_dir}kernel_est_{arrival_rate}.csv')

        iadeep_makespan.append(iadeep_df["end_time"].max()-iadeep_df["start_time"].min())
        antman_makespan.append(antman_df["end_time"].max()-antman_df["start_time"].min())
        mps_makespan.append(mps_df["end_time"].max()-mps_df["start_time"].min())
        kernel_est_makespan.append(kernel_est_df["end_time"].max()-kernel_est_df["start_time"].min())

    ax.plot(x, antman_makespan, label="Antman", marker='d', markersize=5)
    ax.plot(x, mps_makespan, label="MPS", marker='p', markersize=5)
    ax.plot(x, kernel_est_makespan, label="Kernel Est.", marker='X', markersize=5)
    ax.plot(x, iadeep_makespan, label="IADeep", marker='h', markersize=5)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Relative Task Load")
    ax.set_ylabel("Norm. Makespan")
    ax.legend(loc="best", ncol=1, handlelength=1.6, handletextpad=0.5, columnspacing=0.6)
    plt.tight_layout(pad=0.0, h_pad=0, w_pad=0, rect=None)
    plt.savefig('../pdf/makespan_various_arrival_rates.pdf')


# plot CDF of search rouds (Fig. 12a)
def plot_search_rounds_cdf(iadeep_path):
    iadeep_rounds = []

    iadeep_df = pd.read_csv(iadeep_path)
    iadeep_rounds = iadeep_df["search_rounds"].values

    # plot CDF of search rounds
    fig = plt.figure(figsize=(3,2.3), dpi=120)
    ax = fig.add_subplot(111)
    ax.hist(iadeep_rounds, bins=100, density=True, cumulative=True)
    ax.set_xlabel("Rounds")
    ax.set_ylabel("CDF")
    plt.tight_layout(pad=0.0, h_pad=0, w_pad=0, rect=None)
    plt.savefig('../pdf/search_rounds_cdf.pdf')


# plot scheduling cost (Fig. 14b)
def plot_scheduling_cost(iadeep_path):
    scheduling_time = []
    iadeep_df = pd.read_csv(iadeep_path)
    scheduling_time = iadeep_df["scheduling_time"].values

    fig = plt.figure(figsize=(3,2.3), dpi=120)
    ax = fig.add_subplot(111)
    ax.hist(scheduling_time, bins=100, density=True, cumulative=True)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("CDF")
    plt.tight_layout(pad=0.0, h_pad=0, w_pad=0, rect=None)
    plt.savefig('../pdf/scheduling_cost.pdf')


if __name__ == "__main__":
    iadeep_dir = "../data/iadeep/"
    antman_dir = "../data/antman/"
    mps_dir = "../data/mps/"
    kernel_est_dir = "../data/kernel_est/"
    dataset_dir = "../dataset/"
    iadeep_path = f"{dataset_dir}/IADeep_1.0_2023-07-03-05_56_14.csv"
    antman_path = f"{dataset_dir}/ANTMAN_1.0_2023-07-06-01_57_52.csv"
    mps_path = f"{dataset_dir}/ANTMAN_1.0_2023-07-06-01_57_52.csv"
    kernel_est_path = f"{dataset_dir}/ANTMAN_1.0_2023-07-06-01_57_52.csv"
    iadeep_util_path = f"{dataset_dir}/iadeep_util.csv"
    antman_util_path = f"{dataset_dir}/antman_util.csv"
    mps_util_path = f"{dataset_dir}/mps_util.csv"
    kernel_est_util_path = f"{dataset_dir}/kernel_est_util.csv"
   
    parser = argparse.ArgumentParser()
    parser.add_argument("--fig", type=str, default="all", help="The path to save the results")
    args = parser.parse_args()

    if args.fig == "7a":
        plot_ct_and_makespan(iadeep_path, antman_path, mps_path, kernel_est_path)
    elif args.fig == "7b":
        plot_ct_cdf(iadeep_path, antman_path, mps_path, kernel_est_path)
    elif args.fig == "8":
        plot_sm_mem_util(iadeep_util_path, antman_util_path, mps_util_path, kernel_est_util_path)        
    elif args.fig == "10a":
        plot_ct_various_arrival_rates(iadeep_dir, antman_dir, mps_dir, kernel_est_dir)
    elif args.fig == "10b":
        plot_makespan_various_arrival_rates(iadeep_dir, antman_dir, mps_dir, kernel_est_dir)
    elif args.fig == "12a":
        plot_search_rounds_cdf(iadeep_path)
    elif args.fig == "14b":
        plot_scheduling_cost(iadeep_path)
    elif args.fig == 'all':
        plot_ct_and_makespan(iadeep_path)
        plot_ct_cdf(iadeep_path)
        plot_sm_mem_util(iadeep_util_path, antman_util_path, mps_util_path, kernel_est_util_path)
        plot_ct_various_arrival_rates(iadeep_dir, antman_dir, mps_dir, kernel_est_dir)
        plot_makespan_various_arrival_rates(iadeep_dir, antman_dir, mps_dir, kernel_est_dir)
        plot_search_rounds_cdf(iadeep_path)
        plot_scheduling_cost(iadeep_path)    
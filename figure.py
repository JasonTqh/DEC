import pandas as pd
from matplotlib import pyplot as plt


def figure_Md(time_md):
    df = pd.DataFrame(time_md)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlabel("Md Layers")
    ax.set_ylabel("Training time")
    ax.plot(df.index, df[0], label='Md', color='red', linestyle='--')
    plt.show()
def figure_Me( time_me):
    df = pd.DataFrame(time_me)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlabel("Me Layers")
    ax.set_ylabel("Training time")
    ax.plot(df.index, df[0], label='Me', color='blue', linestyle='--')
    plt.show()
def figure_hiertrain_eec_FEEL(time_for_figure_hiertrain, time_for_figure_eec, time_for_figure_FEEL, bandwidth):
    df = pd.DataFrame(time_for_figure_hiertrain)
    df.insert(1, 1, time_for_figure_eec)
    df.insert(2, 2, time_for_figure_FEEL)
    print(df)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlabel("Bandwidth(Mbps)")
    ax.set_ylabel("Training time")
    ax.plot(bandwidth, df[0], label='HeirTrain', color='blue', linestyle='--')
    ax.plot(bandwidth, df[1], label='EEC', color='red', linestyle='--')
    ax.plot(bandwidth, df[2], label='FEEL', color='green', linestyle='--')
    ax.legend()
    plt.show()
def figure_md_min_training(min_total_time_md, all_cloud, all_edge, hiertrain ):
    df = pd.DataFrame(min_total_time_md)
    df.insert(1, 1, all_cloud)
    df.insert(2, 2, all_edge)
    df.insert(3, 3, hiertrain)
    print(df)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlabel("md layers")
    ax.set_ylabel("Training time")
    ax.plot(df.index, df[0], label='EEC', color='blue', linestyle='--')
    ax.plot(df.index, df[1], label='All-cloud', color='red', linestyle='--')
    ax.plot(df.index, df[2], label='All-edge', color='green', linestyle='--')
    ax.plot(df.index, df[3], label='hiertrain', color='black', linestyle='--')
    ax.legend()
    plt.show()
def figure_me_min_training(min_total_time_me, all_cloud, all_edge):
    df = pd.DataFrame(min_total_time_me)
    df.insert(1, 1, all_cloud)
    df.insert(2, 2, all_edge)
    print(df)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlabel("me layers")
    ax.set_ylabel("Training time")
    ax.plot(df.index, df[0], label='EEC', color='blue', linestyle='--')
    ax.plot(df.index, df[1], label='All-cloud', color='red', linestyle='--')
    ax.plot(df.index, df[2], label='All-edge', color='green', linestyle='--')
    ax.legend()
    plt.show()
def figure_eec_all_cloud_all_edge_bandwidth(time, time_all_cloud, time_all_edge, bandwidth):
    df = pd.DataFrame(time)
    df.insert(1, 1, time_all_cloud)
    df.insert(2, 2, time_all_edge)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlabel("bandwidth(Mbps)")
    ax.set_ylabel("Training time")
    ax.plot(bandwidth, df[0],  label='EEC', color='red', linestyle='--')
    ax.plot(bandwidth, df[1],  label='all_cloud', color='green', linestyle='--')
    ax.plot(bandwidth, df[2],  label='all_edge', color='blue', linestyle='--')
    ax.legend()
    plt.show()
def figure_eec_hiertrain_bandwidth(time, time_hiertrain, time_all_cloud, time_all_edge, bandwidth):
    df = pd.DataFrame(time)
    df.insert(1, 1, time_all_cloud)
    df.insert(2, 2, time_all_edge)
    df.insert(3, 3, time_hiertrain)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlabel("bandwidth(Mbps)")
    ax.set_ylabel("Training time")
    ax.plot(bandwidth, df[0],  label='EEC', color='red', linestyle='--')
    ax.plot(bandwidth, df[1],  label='all_cloud', color='green', linestyle='--')
    ax.plot(bandwidth, df[2],  label='all_edge', color='yellow', linestyle='--')
    ax.plot(bandwidth, df[3],  label='hiertrain', color='blue', linestyle='--')
    ax.legend()
    plt.show()
def figure_eec_edge_cpus( min_total_time_bandwidth_list , bandwidth):
    df = pd.DataFrame(min_total_time_bandwidth_list[0])
    df.insert(1, 1, min_total_time_bandwidth_list[1])
    df.insert(2, 2, min_total_time_bandwidth_list[2])
    df.insert(3, 3, min_total_time_bandwidth_list[3])
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlabel("bandwidth(Mbps)")
    ax.set_ylabel("Training time")
    ax.plot(bandwidth, df[0], label='2 CPU', color='red', linestyle='--')
    ax.plot(bandwidth, df[1], label='4 CPUs', color='green', linestyle='--')
    ax.plot(bandwidth, df[2], label='6 CPUs', color='yellow', linestyle='--')
    ax.plot(bandwidth, df[3], label='8 CPUs', color='blue', linestyle='--')
    ax.legend()
    plt.show()
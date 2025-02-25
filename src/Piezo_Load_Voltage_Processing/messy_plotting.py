#! /usr/bin/env python3

import argparse
import glob
import logging
import os
import pathlib
import re
import sys
import types
import csv

#import dateutil.parser as dup
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas.plotting
import seaborn as sns
from scipy.integrate import trapz
from pathlib import Path


#from kneed import KneeLocator

__version__ = "0.0.1"

"""
for gear_dir in data/gear-*; do python3 src/plot_viz.py -i "$gear_dir"/*; done

"""

def load_file_dmm(rn, rate):
    full_pattern = f"data/A-divider/dmm/*-{rate}-{rn}.txt"
    filepaths = glob.glob(full_pattern, recursive=True)
    if not filepaths:
        print(f"No file found for pattern {full_pattern}")
        return None  # or handle this case appropriately

    filepath = filepaths[0]
    times = []  # Ensure these are defined in this scope
    dcvs = []

    with open(filepath, 'r') as file:
        lines = file.readlines()

    for line in lines:
        if 'functions loading' in line:
            continue
        # Split the line into components based on spaces
        parts = line.strip().split(",")

        # Extract the time and dcv, ensuring there are '=' characters to split on
        time_part = parts[0]
        dcv_part = parts[1]

        if '=' in time_part and '=' in dcv_part:
            time_value = time_part.split('=')[1]
            dcv_value = dcv_part.split('=')[1]
            # Append to the respective lists
            times.append(time_value)
            dcvs.append(dcv_value)
        else:
            print(f"Malformed line skipped: {line}")

    # Create a DataFrame
    df = pd.DataFrame({
        'time': times,
        'dcv': dcvs
    })
    # Convert 'time' to timedelta
    df['time'] = pd.to_timedelta(df['time'])

    # Convert 'dcv' to float
    df['dcv'] = pd.to_numeric(df['dcv'], errors='coerce')
    return df

def load_file_TT(filepath):
    log = logging.getLogger(__name__)
    log.debug("in")


    df = None
    #log.warning(filepath.upper())
    if ".csv" in filepath:
        #log.warning("MEC")
        df = pd.read_csv(filepath,  skiprows=1 )
        df.drop(0, axis=0, inplace=True)
    df["Time"] = df["Time"].astype(float)
    df["Force"] = df["Force"].astype(float)
    df["Stroke"] = df["Stroke"].astype(float)
    df['Time'] = pd.to_timedelta(df['Time'], unit='s')
    df.rename(columns={'Time': 'time', 'Force': 'force', 'Stroke': 'stroke'}, inplace=True)

    # get size and run number and crimp type
    fname = os.path.basename(filepath)
    slugs = Path(fname).stem
    slugs = slugs.split("-")
    rate = slugs[0].split("mm")[0]
    run_num = slugs[-1]#.split(".")[0]
    load = slugs[0].split("_")[-1]
    load = load.split("N")[0]
    sensor = slugs[2]

    # get length
    dname = os.path.dirname(filepath)
    dname = dname.split("/")

    #print("length is {}".format(length))

    #log.warning(fdate)
    log.info(run_num)
    df.meta = types.SimpleNamespace()
    df.meta.filepath = filepath
    df.meta.test_run = run_num
    df.meta.rate = rate
    df.meta.load = load
    df.meta.sensor = sensor

    log.debug("out")
    return df


def load_file_multi(data_paths=None):
    log = logging.getLogger(__name__)
    log.debug("in multi")

    trans_df = []
    for filename in data_paths:
        print(filename)
        df = load_file_TT(filename)
        test_run_num = df.meta.test_run
        trans_df.append(df)

    log.debug("out multi")
    return trans_df, test_run_num


def find_knees(df, param_x , param_y):
    log = logging.getLogger(__name__)
    log.debug("in")
    sensitivity = [10, 100, 200, 400, 500]
    knees = []
    norm_knees = []
    x = df[param_x]
    y = df[param_y]
    max_y_index = df[df[param_y]==max(y)].index.values
    red =1
    max_y_index = np.average(max_y_index)
    num_range = np.arange(1, max_y_index, red)
    red_x = x[num_range]
    red_y = y[num_range]

    # test effect of sensitivity on data
    for s in sensitivity:
        kn = KneeLocator(red_x, red_y, S=s
                         , curve='convex'
                         , direction='increasing'
                         , online=True
                         , interp_method="interp1d"
         )
        knees.append(kn.knee)
        norm_knees.append(kn.norm_knee)
    #log.warning(np.round(knees, 2))
    #log.warning(np.round(norm_knees, 2))
    log.debug("out")
    return knees
def find_increase_points(df, column, threshold=0.005):
    baseline = np.mean(df[column][:10])  # Baseline from the first 10 data points
    deviations = (df[column] - baseline) > threshold  # Only positive deviations that exceed the threshold
    change_points = df[deviations].index  # Indices of these increases
    return change_points

def find_decrease_points(df, column, threshold=0.005):
    baseline = np.mean(df[column][:10])  # Baseline from the first 10 data points
    deviations = (df[column] - baseline) < -threshold  # Only negative deviations that exceed the threshold
    change_points = df[deviations].index  # Indices of these decreases
    return change_points
def find_first_contiguous_decrease_points(df, column, threshold=0.05):
    baseline = np.mean(df[column][:10])  # Baseline from the first 10 data points
    deviations = (df[column] - baseline) < -threshold  # Only negative deviations that exceed the threshold
    change_points = df[deviations].index  # Indices of these decreases

    # Check if there are any change points
    if not change_points.empty:
        # Find the first contiguous sequence of decrease points
        first_seq_start = change_points[0]
        for i in range(1, len(change_points)):
            if change_points[i] != change_points[i-1] + 1:
                first_seq_end = change_points[i-1]
                return range(first_seq_start, first_seq_end + 1)
        # If all points are contiguous till the end of change_points
        return range(first_seq_start, change_points[-1] + 1)

    return []

def find_first_contiguous_increase_points(df, column, threshold=0.05):
    baseline = np.mean(df[column][:10])  # Baseline from the first 10 data points
    deviations = (df[column] - baseline) > threshold  # Only positive deviations that exceed the threshold
    change_points = df[deviations].index  # Indices of these increases
    
    # Check if there are any change points
    if not change_points.empty:
        # Find the first contiguous sequence of increase points
        first_seq_start = change_points[0]
        for i in range(1, len(change_points)):
            if change_points[i] != change_points[i-1] + 1:
                first_seq_end = change_points[i-1]
                return range(first_seq_start, first_seq_end + 1)
        # If all points are contiguous till the end of change_points
        return range(first_seq_start, change_points[-1] + 1)
    
    return []

def extract_load(filename):
    # Use regular expression to find the load value in N
    match = re.search(r'\d+N', filename)
    if match:
        # Convert the extracted load value (without 'N') to integer
        return int(match.group()[:-1])
    return 0

def extract_rate(filename):
    # Use regular expression to find the rate value in mm/min
    match = re.search(r'(\d+)mm_min', filename)
    if match:
        # Convert the extracted rate value to integer
        return int(match.group(1))
    return 0

def plot_data_single_trace(param_y="T", param_x="T", data_paths=None):
    """
    plot one data trace from a file

    simple plotting example
    only load data from first file matched in data_paths
    """
    log = logging.getLogger(__name__)
    log.debug("in")

    plot_param_options = {
        "time":{'label':"Time", 'hi':300, 'lo':0, 'lbl':"Time (sec)"},
        "force":{'label':"Force", 'hi':2600, 'lo':0, 'lbl':"Force (N)"},
        "stroke":{'label':"Stroke", 'hi':310, 'lo':0, 'lbl':"Stroke (mm)"},
        "stress":{'label':"Stress", 'hi':50, 'lo':0, 'lbl':"Stress (MPa)"},
        "strain":{'label':"Strain", 'hi':0.15, 'lo':0, 'lbl':"Strain (mm/mm)"},
    }

    df_shi = load_file_TT(glob.glob(data_paths[0])[0])
    test_run_num = df_shi.meta.test_run
    test_rate = df_shi.meta.sensor
    df_dmm = load_file_dmm(test_run_num, test_rate)
    print(df_dmm)
    #change_points_shi = find_change_points(df_shi, 'force', threshold=0.1)  # Customize column and threshold
    #change_points_dmm = find_change_points(df_dmm, 'dcv', threshold=0.1)  # Customize column and threshold
    
    ## For simplicity, align data from the first change point
    #if not change_points_shi.empty and not change_points_dmm.empty:
    #    start_time_shi = df_shi.iloc[change_points_shi[0]]['time']
    #    start_time_dmm = df_dmm.iloc[change_points_dmm[0]]['time']
    #    # Sync data based on detected start times
    #    df_shi_sync = df_shi[df_shi['time'] >= start_time_shi]
    #    df_dmm_sync = df_dmm[df_dmm['time'] >= start_time_dmm]
    #
    #    # Optionally, trim datasets to the same length (if needed)
    #    min_length = min(len(df_shi_sync), len(df_dmm_sync))
    #    df_shi_sync = df_shi_sync.head(min_length)
    #    df_dmm_sync = df_dmm_sync.head(min_length)
    #
    #    # Output synchronized data
    #    print(df_shi_sync)
    #    print(df_dmm_sync)
    
    #log.warning(df.columns)
    x = df_shi[param_x]
    y = df_shi[param_y]
    xx = df_dmm[param_x]
    yy = df_dmm["dcv"]
    #knee = find_knees(df, param_x, param_y)
    #log.warning(np.round(knee,3))

    figsize = 4  # 12 has been default
    figdpi = 600
    hwratio = 4./3
    fig = plt.figure(figsize=(figsize * hwratio, figsize), dpi=figdpi)
    ax = fig.add_subplot(111)
    #with sns.axes_style("whitegrid"):
    with sns.axes_style("darkgrid"):
        line = "{}-vs-{}".format(param_y, param_x)
        ax.plot(x, y, label=line)#, 'r.')
        ax.plot(xx, yy)#, 'r.')

        ax.set_xlabel(plot_param_options[param_x]['lbl'])
        ax.set_ylabel(plot_param_options[param_y]['lbl'])

        ax.legend(frameon=True)

        logging.info(sys._getframe().f_code.co_name)
        plot_fn = "out/plot--single-run-{}-vs-{}.png".format(
            test_run_num, param_y, param_x)
        logging.info("write plot to: {}".format(plot_fn))
        fig.savefig(plot_fn, bbox_inches='tight')
    return

def plot_data_multi_trace(param_y="T", param_x="T", data_paths=None):
    log = logging.getLogger(__name__)
    log.debug("in")

    plot_param_options = {
        "time":{'label':"Time", 'hi':300, 'lo':0, 'lbl':"Time (sec)"},
        "force":{'label':"Force", 'hi':3000, 'lo':0, 'lbl':"Force (N)"},
        "stroke":{'label':"Stroke", 'hi':310, 'lo':0, 'lbl':"Stroke (mm)"},
        "stress":{'label':"Stress", 'hi':50, 'lo':0, 'lbl':"Stress (Pa)"},
        "strain":{'label':"Strain", 'hi':0.1, 'lo':0, 'lbl':"Strain "},
    }

    multi_df , test_trial = load_file_multi(data_paths)

    figsize = 4  # 12 has been default
    figdpi = 600
    hwratio = 4./3
    fig = plt.figure(figsize=(figsize * hwratio, figsize), dpi=figdpi)
    ax = fig.add_subplot(111)
    colors = {
        15: 'red',
        30: 'blue',
        45: 'green',
        60: 'orange',
        75: 'purple',
        90: 'cyan',
        105: 'magenta',
        120: 'yellow' }
    rates_lab = {
        15: '15 mm/min',
        30: '30 mm/min',
        45: '45 mm/min',
        60: '60 mm/min',
        75: '75 mm/min',
        90: '90 mm/min',
        105: '105 mm/min',
        120: '120 mm/min' }
    #colors = {
    #    25: 'red',
    #    50: 'blue',
    #    75: 'green',
    #    100: 'orange',
    #    125: 'purple',
    #    150: 'cyan',
    #    175: 'magenta',
    #    200: 'yellow',
    #    225: 'teal',  
    #    250: 'violet',
    #    275: 'lime',  
    #    300: 'maroon'}

    #rates_lab = {
    #    25:  '25 N',
    #    50:  '50 N',
    #    75:  '75 N',
    #    100: '100 N',
    #    125: '125 N',
    #    150: '150 N',
    #    175: '175 N',
    #    200: '200 N',
    #    225: '125 N',
    #    250: '150 N',
    #    275: '175 N',
    #    300: '300 N'}


    used_labels = set()

    with sns.axes_style("darkgrid"):
        max_values = []
        for line in multi_df:
            print(line)
            x = line[param_x]
            if param_x == "time":
                x = x.dt.total_seconds()
            y = line[param_y]
            print(x[10])
            test_run_num = line.meta.test_run
            rate = int(line.meta.rate)
            label = rates_lab[rate] if rate not in used_labels else ""
            ax.plot(x, y, label=label, color=colors[rate])
            if rate not in used_labels:
                used_labels.add(rate)
            max_values.append(max(y))
        ax.set_xlabel('Time (sec)')
        ax.set_ylabel('Force (N)')# color=color)

        ax.legend(frameon=True)
        logging.info(sys._getframe().f_code.co_name)
        plot_fn = "out/plot-multi-{}-vs-{}.png".format(
             param_y, param_x)
        logging.info("write plot to: {}".format(plot_fn))
        fig.savefig(plot_fn, bbox_inches='tight')

    return


def plot_force_vs_displacement(param_y="force", param_x="dcv", data_paths=None):
    log = logging.getLogger(__name__)
    log.debug("in")

    df_shi = load_file_TT(glob.glob(data_paths[0])[0])
    test_run_num = df_shi.meta.test_run
    test_rate = df_shi.meta.rate
    df_dmm = load_file_dmm(test_run_num, test_rate)

    if 'time' in df_shi and 'time' in df_dmm:

        # Find significant changes
        increase_points = find_first_contiguous_increase_points(df_shi, 'force', 0.1)
        decrease_points = find_first_contiguous_decrease_points(df_dmm, 'dcv', 0.1)

        df_shi = df_shi.loc[increase_points]
        df_dmm = df_dmm.loc[decrease_points]
        df_shi['time'] = df_shi['time'] - df_shi['time'].iloc[0]
        df_dmm['time'] = df_dmm['time'] - df_dmm['time'].iloc[0]
        df_shi.set_index('time', inplace=True)
        df_dmm.set_index('time', inplace=True)
        df_shi.sort_values('time', inplace=True)
        df_dmm.sort_values('time', inplace=True)

        df_combined = pd.concat([df_shi, df_dmm], axis=1)
        df_combined.sort_values('time', inplace=True)
        df_combined.interpolate(method='linear', inplace=True)
        df_combined = df_combined.dropna()
        max_force_index = df_combined['force'].idxmax()

        # Step 2: Truncate the DataFrame at this index
        df_combined = df_combined.loc[:max_force_index]


        figsize = 4  # 12 has been default
        figdpi = 600
        hwratio = 4./3
        fig = plt.figure(figsize=(figsize * hwratio, figsize), dpi=figdpi)
        ax1 = fig.add_subplot(111)
        with sns.axes_style("darkgrid"):
            #figsize = 6
            #fig, ax1 = plt.subplots(figsize=(figsize * 4./3, figsize))
            #sns.set_style("darkgrid")

            # First axis
            color = 'tab:red'
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Force (N)', color=color)
            ax1.plot(df_combined.index, df_combined['force'], color=color,linestyle='--')
            ax1.tick_params(axis='y', labelcolor=color)

            # Create a second y-axis for the dcv
            ax2 = ax1.twinx()
            color = 'tab:blue'
            ax2.set_ylabel('DCV (V)', color=color)  # we already handled the x-label with ax1
            ax2.plot(df_combined.index, df_combined['dcv'], color=color)
            ax2.tick_params(axis='y', labelcolor=color)

            fig.tight_layout()  # to make sure that the labels do not overlap
            # Save the plot
            plot_filename = f"out/test_plot_{test_run_num}.png"
            plt.savefig(plot_filename, format='png')

            print(f"Plot saved to {plot_filename}")

def plot_force_vs_dcv(param_y="force", param_x="dcv", data_paths=None):
    log = logging.getLogger(__name__)
    log.debug("in")

    df_shi = load_file_TT(glob.glob(data_paths[0])[0])
    test_run_num = df_shi.meta.test_run
    test_rate = df_shi.meta.rate
    df_dmm = load_file_dmm(test_run_num, test_rate)

    if 'time' in df_shi and 'time' in df_dmm:

        # Find significant changes
        increase_points = find_first_contiguous_increase_points(df_shi, 'force', 0.1)
        decrease_points = find_first_contiguous_decrease_points(df_dmm, 'dcv', 0.1)

        df_shi = df_shi.loc[increase_points]
        df_dmm = df_dmm.loc[decrease_points]
        df_shi['time'] = df_shi['time'] - df_shi['time'].iloc[0]
        df_dmm['time'] = df_dmm['time'] - df_dmm['time'].iloc[0]
        df_shi.set_index('time', inplace=True)
        df_dmm.set_index('time', inplace=True)
        df_shi.sort_values('time', inplace=True)
        df_dmm.sort_values('time', inplace=True)

        df_combined = pd.concat([df_shi, df_dmm], axis=1)
        df_combined.sort_values('time', inplace=True)
        df_combined.interpolate(method='linear', inplace=True)
        df_combined = df_combined.dropna()
        max_force_index = df_combined['force'].idxmax()

        # Step 2: Truncate the DataFrame at this index
        df_combined = df_combined.loc[:max_force_index]


        figsize = 4  # 12 has been default
        figdpi = 600
        hwratio = 4./3
        fig = plt.figure(figsize=(figsize * hwratio, figsize), dpi=figdpi)
        ax1 = fig.add_subplot(111)
        with sns.axes_style("darkgrid"):
            ax1.set_xlabel('Force (N)')
            ax1.set_ylabel('DCV (V)')  # we already handled the x-label with ax1
            ax1.plot(df_combined['force'], np.abs(df_combined['dcv']))


            fig.tight_layout()  # to make sure that the labels do not overlap
            # Save the plot
            plot_filename = f"out/test_plot_f_vs_v_{test_run_num}.png"
            plt.savefig(plot_filename, format='png')

            print(f"Plot saved to {plot_filename}")
def plot_force_vs_dcv_multi(param_y="force", param_x="dcv", data_paths=None):
    log = logging.getLogger(__name__)
    log.debug("in")
    print(data_paths)

    figsize = 4  # 12 has been default
    figdpi = 600
    hwratio = 4./3
    fig = plt.figure(figsize=(figsize * hwratio, figsize), dpi=figdpi)
    ax1 = fig.add_subplot(111)
    colors = {
        'A': 'red',
        'B': 'blue',
        'C': 'green' }
    #colors = {
    #    15: 'red',
    #    30: 'blue',
    #    45: 'green',
    #    60: 'orange',
    #    75: 'purple',
    #    90: 'cyan',
    #    105: 'magenta',
    #    120: 'yellow' }
    #rates_lab = {
    #    15: '15 mm/min',
    #    30: '30 mm/min',
    #    45: '45 mm/min',
    #    60: '60 mm/min',
    #    75: '75 mm/min',
    #    90: '90 mm/min',
    #    105: '105 mm/min',
    #    120: '120 mm/min' }
    #colors = {
    #    25: 'red',
    #    50: 'blue',
    #    75: 'green',
    #    100: 'orange',
    #    125: 'purple',
    #    150: 'cyan',
    #    175: 'magenta',
    #    200: 'yellow',
    #    225: 'teal',  
    #    250: 'violet',
    #    275: 'lime',  
    #    300: 'maroon'}

    #rates_lab = {
    #    25:  '25 N',
    #    50:  '50 N',
    #    75:  '75 N',
    #    100: '100 N',
    #    125: '125 N',
    #    150: '150 N',
    #    175: '175 N',
    #    200: '200 N',
    #    225: '125 N',
    #    250: '150 N',
    #    275: '175 N',
    #    300: '300 N'}

    used_labels = set()

    with sns.axes_style("darkgrid"):
        ax1.set_xlabel('Force (N)')
        ax1.set_ylabel('DCV (V)')  # we already handled the x-label with ax1
        for data_path in data_paths: 
            print(data_path)
            df_shi = load_file_TT(data_path)
            test_run_num = df_shi.meta.test_run
            test_rate = df_shi.meta.rate
            test_rate = int(test_rate)
            test_sensor = df_shi.meta.sensor

            df_dmm = load_file_dmm(test_run_num, test_rate)
            print(df_dmm)
            sys.exit()

            if 'time' in df_shi and 'time' in df_dmm:
            
                # Find significant changes
                increase_points = find_first_contiguous_increase_points(df_shi, 'force', 0.1)
                decrease_points = find_first_contiguous_decrease_points(df_dmm, 'dcv', 0.1)

                df_shi = df_shi.loc[increase_points]
                df_dmm = df_dmm.loc[decrease_points]
                df_shi['time'] = df_shi['time'] - df_shi['time'].iloc[0]
                df_dmm['time'] = df_dmm['time'] - df_dmm['time'].iloc[0]
                df_shi.set_index('time', inplace=True)
                df_dmm.set_index('time', inplace=True)
                df_shi.sort_values('time', inplace=True)
                df_dmm.sort_values('time', inplace=True)

                df_combined = pd.concat([df_shi, df_dmm], axis=1)
                df_combined.sort_values('time', inplace=True)
                df_combined.interpolate(method='linear', inplace=True)
                df_combined = df_combined.dropna()
                max_force_index = df_combined['force'].idxmax()

                # Step 2: Truncate the DataFrame at this index
                df_combined = df_combined.loc[:max_force_index]
                #label = rates_lab[test_rate] if test_rate not in used_labels else ""
                label = f"Sensor {test_sensor}"

                ax1.plot(df_combined['force'], np.abs(df_combined['dcv']), label=label, color=colors[test_sensor])
                ax1.scatter(df_combined['force'][-1], np.abs(df_combined['dcv'][-1]), color=colors[test_sensor])
                if test_rate not in used_labels:
                    used_labels.add(test_rate)

    #ax1.legend(frameon=True)
    ax1.legend(loc='upper left', ncol=2, frameon=True)

    fig.tight_layout()  # to make sure that the labels do not overlap
    # Save the plot
    plot_filename = f"out/test_plot_f_vs_v_multi.png"
    plt.savefig(plot_filename, format='png')


def setup_logging(verbosity):
    log_fmt = ("%(levelname)s - %(module)s - "
               "%(funcName)s @%(lineno)d: %(message)s")
    # addl keys: asctime, module, name
    logging.basicConfig(filename=None,
                        format=log_fmt,
                        level=logging.getLevelName(verbosity))

    return

def parse_command_line():
    parser = argparse.ArgumentParser(description="Analyse sensor data")
    parser.add_argument("-V", "--version", "--VERSION", action="version",
                        version="%(prog)s {}".format(__version__))
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        dest="verbosity", help="verbose output")
    # -h/--help is auto-added
    parser.add_argument("-d", "--dir", dest="dirs",
                        # action="store",
                        nargs='+',
                        default=None, required=False, help="directories with data files")
    parser.add_argument("-i", "--in", dest="input",
                        # action="store",
                        nargs='+',
                        default=None, required=False, help="path to input")
    ret = vars(parser.parse_args())
    ret["verbosity"] = max(0, 30 - 10 * ret["verbosity"])

    return ret

def main():
    cmd_args = parse_command_line()
    setup_logging(cmd_args['verbosity'])

    #  plt.style.use('ggplot')
    #  plt.style.use('seaborn-colorblind')
    #  plt.style.use('seaborn-paper')
    #  plt.style.use('seaborn-whitegrid')
    #  plt.style.use('seaborn-darkgrid')
    #  plt.style.use('whitegrid') #not appl0.15cable?

    #data_paths=cmd_args['input']
    #load_file_TT(glob.glob(data_paths[0])[0])
    #plot_force_vs_displacement(param_y="force", param_x="dcv",data_paths=cmd_args['input'])
    sorted_files = sorted(cmd_args['input'], key=extract_load)
    sorted_files = sorted(cmd_args['input'], key=extract_rate)
    plot_force_vs_dcv_multi(param_y="force", param_x="dcv",data_paths=sorted_files)
    #plot_data_single_trace(param_y="force", param_x="time", data_paths=cmd_args['input'])
    #plot_data_multi_trace(param_y="force", param_x="stroke", data_paths=sorted_files)
    plot_data_multi_trace(param_y="force", param_x="time", data_paths=sorted_files)


if "__main__" == __name__:
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("exited")




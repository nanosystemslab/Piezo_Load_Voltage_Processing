#! /usr/bin/env python3

import argparse
import glob
import logging
import os
from pathlib import Path
import re
import sys
import types
import csv

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas.plotting
import seaborn as sns
from scipy.integrate import trapz


__version__ = "0.0.1"


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


def load_file_dmm(base_file_path, rn, rate):
    base_file_path = Path(base_file_path).parent.parent.resolve()
    full_pattern = f"{base_file_path}/dmm/*-{rate}*{rn}.txt"
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
    log.info(run_num)
    df.meta = types.SimpleNamespace()
    df.meta.filepath = filepath
    df.meta.test_run = run_num
    df.meta.rate = rate
    df.meta.load = load
    df.meta.sensor = sensor

    log.debug("out")
    return df


def load_estimated_comsol_simulation(filepath):
    base_file_path = Path(filepath).parent.parent.resolve()
    file_path = base_file_path / "COMSOL--Force-Displacement.csv"
    df = pd.read_csv(file_path)
    print(df)
    force = df["Force"]
    disp = df["Displacement"]
    return force, disp


def plot_force_vs_dcv_multi(param_y="force", param_x="dcv", data_paths=None):
    log = logging.getLogger(__name__)
    log.debug("in")
    print(data_paths)

    figsize = 4  # 12 has been default
    figdpi = 600
    hwratio = 4./3
    fig = plt.figure(figsize=(figsize * hwratio, figsize), dpi=figdpi)
    ax1 = fig.add_subplot(111)

    with sns.axes_style("darkgrid"):
        ax1.set_xlabel('Displacement (mm)')  
        ax1.set_ylabel('Force (N)')

        data_list = []

        for data_path in data_paths: 
            # Load Shidmazu data
            df_shi = load_file_TT(data_path)
            test_run_num = df_shi.meta.test_run
            test_rate = df_shi.meta.rate
            test_rate = int(test_rate)
            test_sensor = df_shi.meta.sensor
            test_file_path = df_shi.meta.filepath

            # Append processed data to the list
            data_list.append(df_shi[['force', 'stroke']])
     
        # Convert the list into a single DataFrame
        df_all = pd.concat(data_list, axis=1)
        df_all.dropna(inplace=True)  # Ensure complete data
        
        # Filter columns for force and stroke
        df_force = df_all.filter(like="force")   # Columns containing "force"
        df_stroke = df_all.filter(like="stroke")   # Columns containing "stroke"
        
        # Compute statistics along rows (or adjust axis as needed)
        df_force_mean = df_force.mean(axis=1)
        df_force_std  = df_force.std(axis=1)       # Compute standard deviation for force
        avg_force_std = df_force_std.mean()
        
        df_stroke_mean = df_stroke.mean(axis=1)
        
        # Plot the experimental average with x = stroke and y = force
        ax1.plot(df_stroke_mean, df_force_mean,
                 label="Experimental Average", color="blue", marker="D", markevery=5)
        
        # Fill between the average ± standard deviation for force
        ax1.fill_between(df_stroke_mean,
                         df_force_mean - df_force_std,
                         df_force_mean + df_force_std,
                         color='blue', alpha=0.3,
                         label=f"Standard Deviation {avg_force_std:.2f}")
        # Load COMSOL Simulation Data
        sim_force, sim_disp = load_estimated_comsol_simulation(test_file_path)

        # Plot the experimental average with x = stroke and y = force
        ax1.plot(sim_disp, sim_force,
                 label="COMSOL Simulation", color="red", marker="o", markevery=5)

        ax1.legend(loc='lower right', ncol=1, frameon=True)

        fig.tight_layout()  # to make sure that the labels do not overlap
        # Save the plot
        plot_filename = f"out/plot_force_vs_displacement_multi.png"
        plt.savefig(plot_filename, format='png')


def main():
    cmd_args = parse_command_line()
    setup_logging(cmd_args['verbosity'])

    sorted_files = sorted(cmd_args['input'], key=extract_load)
    sorted_files = sorted(cmd_args['input'], key=extract_rate)
    plot_force_vs_dcv_multi(param_y="force", param_x="dcv",data_paths=sorted_files)

if "__main__" == __name__:
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("exited")

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

def load_estimated_electrical_simulation(filepath, x):
    base_file_path = Path(filepath).parent.parent.resolve()
    #file_path = base_file_path / "Estimated_Electrical_Simulation.csv"
    # df = pd.read_csv(file_path)
    # 
    # # Compute polynomial coefficients and create a polynomial function:
    # degree = 3
    # coeffs = np.polyfit(df["x"], df[" y"], degree)
    # poly_fit = np.poly1d(coeffs)
    # 
    # # Generate a smooth curve for the polynomial fit:
    # force_fit = x
    # dcv_fit = poly_fit(force_fit)

    file_path = base_file_path / "single_pzt_shimadzu_test_model.txt"
    df = pd.read_csv(file_path, sep="\t")

    return df 


def average_percentage_difference(dcv_fit, df_dcv_mean):
    # Convert to numpy arrays in case they aren't already
    dcv_fit = np.array(dcv_fit)
    df_dcv_mean = np.array(df_dcv_mean)
    
    # Compute percentage differences; add a small epsilon to avoid division by zero if necessary.
    epsilon = 1e-8
    percentage_diff = np.abs(dcv_fit - df_dcv_mean) / (df_dcv_mean + epsilon) * 100
    
    # Compute the average percentage difference
    avg_percent_diff = np.mean(percentage_diff)
    return avg_percent_diff


def average_percent_difference(sim_disp, sim_force, df_stroke_mean, df_force_mean):
    """
    Compute the average percentage difference between simulation and experimental data.
    
    The function interpolates the simulation displacement and force onto the experimental
    force values and then computes the average percentage difference relative to the experimental
    stroke and force values.
    
    Parameters:
        sim_disp (array-like): Simulated displacement (or stroke) values.
        sim_force (array-like): Simulated force values (independent variable for simulation).
        df_stroke_mean (array-like): Experimental stroke values.
        df_force_mean (array-like): Experimental force values (independent variable for experiment).
        
    Returns:
        avg_stroke_diff (float): Average percentage difference for displacement/stroke.
        avg_force_diff (float): Average percentage difference for force.
    """
    # Convert inputs to numpy arrays
    sim_disp = np.array(sim_disp)
    sim_force = np.array(sim_force)
    df_stroke_mean = np.array(df_stroke_mean)
    df_force_mean = np.array(df_force_mean)
    
    # Interpolate simulation displacement to experimental force values
    sim_disp_interp = np.interp(df_force_mean, sim_force, sim_disp)
    stroke_pct_diff = np.abs(sim_disp_interp - df_stroke_mean) / np.abs(df_stroke_mean) * 100
    avg_stroke_diff = np.mean(stroke_pct_diff)
    
    # Interpolate simulation force to experimental force values
    sim_force_interp = np.interp(df_force_mean, sim_force, sim_force)
    force_pct_diff = np.abs(sim_force_interp - df_force_mean) / np.abs(df_force_mean) * 100
    avg_force_diff = np.mean(force_pct_diff)
    
    return avg_stroke_diff, avg_force_diff

def rmse_calculation(sim_disp, sim_force, df_stroke_mean, df_force_mean):
    """
    Compute the Root Mean Square Error (RMSE) between simulation and experimental data.
    
    The function interpolates the simulation displacement and force onto the experimental
    force values and then computes the RMSE for both stroke and force values.
    
    Parameters:
        sim_disp (array-like): Simulated displacement (or stroke) values.
        sim_force (array-like): Simulated force values (independent variable for simulation).
        df_stroke_mean (array-like): Experimental stroke values.
        df_force_mean (array-like): Experimental force values (independent variable for experiment).
        
    Returns:
        stroke_rmse (float): RMSE for displacement/stroke.
        force_rmse (float): RMSE for force.
    """
    # Convert inputs to numpy arrays
    sim_disp = np.array(sim_disp)
    sim_force = np.array(sim_force)
    df_stroke_mean = np.array(df_stroke_mean)
    df_force_mean = np.array(df_force_mean)
    
    # Interpolate simulation displacement to experimental force values
    sim_disp_interp = np.interp(df_force_mean, sim_force, sim_disp)
    stroke_rmse = np.sqrt(np.mean((sim_disp_interp - df_stroke_mean)**2))
    
    # Interpolate simulation force to experimental force values
    sim_force_interp = np.interp(df_force_mean, sim_force, sim_force)
    force_rmse = np.sqrt(np.mean((sim_force_interp - df_force_mean)**2))
    
    return stroke_rmse, force_rmse

def plot_force_vs_dcv_multi(param_y="force", param_x="dcv", data_paths=None):
    log = logging.getLogger(__name__)
    log.debug("in")

    figsize = 4  # 12 has been default
    figdpi = 600
    hwratio = 4./3
    fig = plt.figure(figsize=(figsize * hwratio, figsize), dpi=figdpi)
    ax1 = fig.add_subplot(111)

    with sns.axes_style("darkgrid"):
        ax1.set_xlabel('Force (N)')
        ax1.set_ylabel('DCV (V)')  # we already handled the x-label with ax1

        data_list = []

        for data_path in data_paths: 
            # Load Shidmazu data
            df_shi = load_file_TT(data_path)
            test_run_num = df_shi.meta.test_run
            test_rate = df_shi.meta.rate
            test_rate = int(test_rate)
            test_sensor = df_shi.meta.sensor
            test_file_path = df_shi.meta.filepath

            # Load DMM Data
            df_dmm = load_file_dmm(test_file_path, test_run_num, test_rate)

            if 'time' in df_shi and 'time' in df_dmm:
                # Find significant changes
                increase_points = find_first_contiguous_increase_points(df_shi, 'force', 0.1)
                decrease_points = find_first_contiguous_decrease_points(df_dmm, 'dcv', 0.01)

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
                threshold_force = 250  # Set truncation threshold
                truncate_index = df_combined[df_combined['force'] >= threshold_force].first_valid_index()

                # Step 2: Truncate the DataFrame at this index
                df_combined = df_combined.loc[:truncate_index]
                df_combined["dcv"] = np.abs(df_combined["dcv"])
                df_combined["time"] = df_combined.index

                # Append processed data to the list
                data_list.append(df_combined[['force', 'dcv', 'time']])

     
        dataset_lengths = [len(df) for df in data_list]
        # Convert the list into a single DataFrame
        df_all = pd.concat(data_list, axis=1)
        #df_all.interpolate(method='linear', inplace=True)
        df_all.dropna(inplace=True)  # Ensure complete data
        
        # Compute statistics
        df_force = df_all.filter(like="force")  # Select all columns containing "force"
        df_dcv = df_all.filter(like="dcv")      # Select all columns containing "dcv"
        df_time = df_all.filter(like="time")   # Columns containing "time"
        
        df_force_mean = df_force.mean(axis=1)
        df_dcv_mean = df_dcv.mean(axis=1)
        df_dcv_std = df_dcv.std(axis=1)

        # Compute mean time
        df_time_mean = df_time.mean(axis=1)
        df_time_mean = df_time_mean.dt.total_seconds()
        
        # Ensure consistent indices
        df_force_mean, df_dcv_mean = df_force_mean.align(df_dcv_mean, join='inner')
        df_force_mean, df_dcv_std = df_force_mean.align(df_dcv_std, join='inner')
        df_dcv_mean, df_dcv_std = df_dcv_mean.align(df_dcv_std, join='inner')
        
        # Sort by force to ensure smooth plotting
        sorted_indices = df_force_mean.argsort()
        df_force_mean = df_force_mean.iloc[sorted_indices]
        df_dcv_mean = df_dcv_mean.iloc[sorted_indices]
        df_dcv_std = df_dcv_std.iloc[sorted_indices]
        avg_dcv_std = df_dcv_std.mean()

        # Plot
        ax1.plot( df_force_mean,  df_dcv_mean, label="Experimental Average", color="blue", marker="D", markevery=5)
        ax1.fill_between(df_force_mean, df_dcv_mean - df_dcv_std, df_dcv_mean + df_dcv_std,
                 color='blue', alpha=0.3, label=f"Standard Deviation {avg_dcv_std:.2f}")

        # Estimated Values From Electrical Simulation 
        df_elec_sim = load_estimated_electrical_simulation(test_file_path, df_force_mean)
        
        df_force = pd.DataFrame({
            'Force Mean (N)': df_force_mean,
            'Time Mean (s)': df_time_mean,
            'EXP DCV': df_dcv_mean
        }).reset_index(drop=True)

        # Merge on "Time Mean (s)"
        #df_combined = pd.merge(df_elec_sim, df_force, left_on="time", right_on="Time Mean (s)", how="inner")
        df_combined = pd.merge_asof(df_elec_sim, df_force, left_on="time", right_on="Time Mean (s)")

        # Find the first index where Force reaches its max value
        max_force_value = df_combined["Force Mean (N)"].max()
        max_force_index = df_combined[df_combined["Force Mean (N)"] == max_force_value].index[0]
        
        # Keep only rows up to (and including) the first max force occurrence
        df_combined = df_combined.loc[:max_force_index].reset_index(drop=True)
        
        df_combined = df_combined.drop_duplicates(subset=["Force Mean (N)"], keep="first").reset_index(drop=True)

        df_combined["% Difference"] = np.abs(df_combined["V(opamp_out)"] - df_combined["EXP DCV"]) / df_combined["EXP DCV"] * 100

        # Plot simulation data
        ax1.plot(df_combined["Force Mean (N)"], df_combined["V(opamp_out)"],
                 label="Circuit Simulation", color="red",
                 marker="o", markevery=5, linewidth=2)

        # Calculate metrics using the functions
        # For the functions, we need to match the parameter names:
        # sim_disp -> V(opamp_out) (simulation voltage)
        # sim_force -> Force Mean (N) (force values)
        # df_stroke_mean -> EXP DCV (experimental voltage)
        # df_force_mean -> Force Mean (N) (same force values)
        
        # Calculate average percentage difference
        avg_voltage_diff, avg_force_diff = average_percent_difference(
            sim_disp=df_combined["V(opamp_out)"],
            sim_force=df_combined["Force Mean (N)"],
            df_stroke_mean=df_combined["EXP DCV"],
            df_force_mean=df_combined["Force Mean (N)"]
        )
        
        # Calculate RMSE
        voltage_rmse, force_rmse = rmse_calculation(
            sim_disp=df_combined["V(opamp_out)"],
            sim_force=df_combined["Force Mean (N)"],
            df_stroke_mean=df_combined["EXP DCV"],
            df_force_mean=df_combined["Force Mean (N)"]
        )

        # Add metrics to legend as invisible plots
        ax1.plot([], [], ' ', label=f"Avg Diff: {avg_voltage_diff:.2f}%")
        ax1.plot([], [], ' ', label=f"RMSE: {voltage_rmse:.3f} V")

        ax1.set_xlim(0, 235)
        ax1.legend(loc='lower right', ncol=1, frameon=True)

        fig.tight_layout()  # to make sure that the labels do not overlap
        # Save the plot
        plot_filename = f"out/plot_force_vs_voltage_multi.png"
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

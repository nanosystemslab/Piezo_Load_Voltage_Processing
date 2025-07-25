#!/usr/bin/env python3

import argparse
import glob
import logging
import os
from pathlib import Path
import re
import sys
import types

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

__version__ = "0.0.1"


class SensorAnalyzer:
    """Main class for sensor data analysis"""
    
    def __init__(self, verbosity=30):
        self.setup_logging(verbosity)
        self.log = logging.getLogger(__name__)
    
    def setup_logging(self, verbosity):
        """Setup logging configuration"""
        log_fmt = ("%(levelname)s - %(module)s - "
                   "%(funcName)s @%(lineno)d: %(message)s")
        logging.basicConfig(filename=None, format=log_fmt, level=verbosity)

    @staticmethod
    def extract_load(filename):
        """Extract load value from filename (e.g., '100N' -> 100)"""
        match = re.search(r'\d+N', filename)
        return int(match.group()[:-1]) if match else 0

    @staticmethod
    def extract_rate(filename):
        """Extract rate value from filename (e.g., '5mm_min' -> 5)"""
        match = re.search(r'(\d+)mm_min', filename)
        return int(match.group(1)) if match else 0

    def find_first_contiguous_points(self, df, column, threshold, direction='increase'):
        """Find first contiguous sequence of points above/below threshold"""
        baseline = np.mean(df[column][:10])
        
        if direction == 'increase':
            deviations = (df[column] - baseline) > threshold
        else:  # decrease
            deviations = (df[column] - baseline) < -threshold
            
        change_points = df[deviations].index
        
        if not change_points.empty:
            first_seq_start = change_points[0]
            for i in range(1, len(change_points)):
                if change_points[i] != change_points[i-1] + 1:
                    first_seq_end = change_points[i-1]
                    return range(first_seq_start, first_seq_end + 1)
            return range(first_seq_start, change_points[-1] + 1)
        return []

    def load_file_dmm(self, base_file_path, run_number, rate):
        """Load DMM (Digital Multimeter) data file"""
        base_path = Path(base_file_path).parent.parent.resolve()
        pattern = f"{base_path}/dmm/*-{rate}*{run_number}.txt"
        filepaths = glob.glob(pattern, recursive=True)
        
        if not filepaths:
            self.log.warning(f"No DMM file found for pattern {pattern}")
            return None

        filepath = filepaths[0]
        times, dcvs = [], []

        with open(filepath, 'r') as file:
            for line in file:
                if 'functions loading' in line:
                    continue
                    
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    time_part, dcv_part = parts[0], parts[1]
                    if '=' in time_part and '=' in dcv_part:
                        times.append(time_part.split('=')[1])
                        dcvs.append(dcv_part.split('=')[1])

        df = pd.DataFrame({'time': times, 'dcv': dcvs})
        df['time'] = pd.to_timedelta(df['time'])
        df['dcv'] = pd.to_numeric(df['dcv'], errors='coerce')
        return df

    def load_file_TT(self, filepath):
        """Load Tensile Test data file"""
        df = pd.read_csv(filepath, skiprows=1).drop(0, axis=0)
        
        # Convert data types
        for col in ["Time", "Force", "Stroke"]:
            df[col] = df[col].astype(float)
        
        df['Time'] = pd.to_timedelta(df['Time'], unit='s')
        df.rename(columns={'Time': 'time', 'Force': 'force', 'Stroke': 'stroke'}, inplace=True)

        # Extract metadata from filename
        fname = os.path.basename(filepath)
        slugs = Path(fname).stem.split("-")
        
        df.meta = types.SimpleNamespace()
        df.meta.filepath = filepath
        df.meta.test_run = slugs[-1]
        df.meta.rate = slugs[0].split("mm")[0]
        df.meta.load = slugs[0].split("_")[-1].split("N")[0]
        df.meta.sensor = slugs[2]

        return df

    def load_electrical_simulation(self, filepath):
        """Load electrical simulation data"""
        base_path = Path(filepath).parent.parent.resolve()
        file_path = base_path / "single_pzt_shimadzu_test_model.txt"
        return pd.read_csv(file_path, sep="\t")

    @staticmethod
    def average_percent_difference(sim_values, exp_values):
        """Calculate average percentage difference between simulation and experimental data"""
        sim_values = np.array(sim_values)
        exp_values = np.array(exp_values)
        
        pct_diff = np.abs(sim_values - exp_values) / np.abs(exp_values) * 100
        return np.mean(pct_diff)

    @staticmethod
    def rmse_calculation(sim_values, exp_values):
        """Calculate Root Mean Square Error"""
        sim_values = np.array(sim_values)
        exp_values = np.array(exp_values)
        return np.sqrt(np.mean((sim_values - exp_values)**2))

    def process_data_files(self, data_paths):
        """Process multiple data files and combine them"""
        data_list = []

        for data_path in data_paths:
            # Load data files
            df_shi = self.load_file_TT(data_path)
            df_dmm = self.load_file_dmm(
                df_shi.meta.filepath, 
                df_shi.meta.test_run, 
                int(df_shi.meta.rate)
            )

            if df_dmm is None or 'time' not in df_shi or 'time' not in df_dmm:
                continue

            # Find significant changes
            increase_points = self.find_first_contiguous_points(df_shi, 'force', 0.1, 'increase')
            decrease_points = self.find_first_contiguous_points(df_dmm, 'dcv', 0.01, 'decrease')

            # Process and combine data
            df_shi = df_shi.loc[increase_points]
            df_dmm = df_dmm.loc[decrease_points]
            
            # Normalize time and set as index
            for df in [df_shi, df_dmm]:
                df['time'] = df['time'] - df['time'].iloc[0]
                df.set_index('time', inplace=True)
                df.sort_values('time', inplace=True)

            # Combine and interpolate
            df_combined = pd.concat([df_shi, df_dmm], axis=1)
            df_combined.sort_values('time', inplace=True)
            df_combined.interpolate(method='linear', inplace=True)
            df_combined = df_combined.dropna()

            # Truncate at force threshold
            threshold_force = 250
            truncate_index = df_combined[df_combined['force'] >= threshold_force].first_valid_index()
            if truncate_index:
                df_combined = df_combined.loc[:truncate_index]

            df_combined["dcv"] = np.abs(df_combined["dcv"])
            df_combined["time"] = df_combined.index
            data_list.append(df_combined[['force', 'dcv', 'time']])

        return data_list

    def plot_force_vs_voltage(self, data_paths, output_dir="out"):
        """Main plotting function with interpolated simulation data"""
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Process data
        data_list = self.process_data_files(data_paths)
        if not data_list:
            self.log.error("No valid data files found")
            return

        # Combine all data
        df_all = pd.concat(data_list, axis=1)
        df_all.dropna(inplace=True)

        # Compute statistics
        df_force = df_all.filter(like="force")
        df_dcv = df_all.filter(like="dcv")
        df_time = df_all.filter(like="time")

        df_force_mean = df_force.mean(axis=1)
        df_dcv_mean = df_dcv.mean(axis=1)
        df_dcv_std = df_dcv.std(axis=1)
        df_time_mean = df_time.mean(axis=1).dt.total_seconds()

        # Align indices and sort by force
        df_force_mean, df_dcv_mean = df_force_mean.align(df_dcv_mean, join='inner')
        df_force_mean, df_dcv_std = df_force_mean.align(df_dcv_std, join='inner')
        
        sorted_indices = df_force_mean.argsort()
        df_force_mean = df_force_mean.iloc[sorted_indices]
        df_dcv_mean = df_dcv_mean.iloc[sorted_indices]
        df_dcv_std = df_dcv_std.iloc[sorted_indices]

        # Setup plot
        figsize = 4
        figdpi = 600
        hwratio = 4./3
        fig, ax = plt.subplots(figsize=(figsize * hwratio, figsize), dpi=figdpi)

        with sns.axes_style("darkgrid"):
            ax.set_xlabel('Force (N)')
            ax.set_ylabel('DCV (V)')

            # Plot experimental data
            avg_dcv_std = df_dcv_std.mean()
            ax.plot(df_force_mean, df_dcv_mean, 
                   label="Experimental Average", color="blue", marker="D", markevery=5)
            ax.fill_between(df_force_mean, df_dcv_mean - df_dcv_std, df_dcv_mean + df_dcv_std,
                           color='blue', alpha=0.3, label=f"Standard Deviation {avg_dcv_std:.2f}")

            # Load simulation data
            df_elec_sim = self.load_electrical_simulation(data_paths[0])
            
            # Interpolate simulation voltage to experimental force values
            # This ensures we use the same x-axis (force) for comparison
            sim_voltage_interp = np.interp(
                df_force_mean.values,  # x values where we want interpolation (exp force)
                df_elec_sim["force"].values,  # x values from simulation
                df_elec_sim["V(opamp_out)"].values  # y values from simulation
            )
            
            # Create simulation data aligned with experimental force values
            df_sim_aligned = pd.DataFrame({
                'Force (N)': df_force_mean.values,
                'V(opamp_out)': sim_voltage_interp
            })

            # Plot simulation data (now perfectly aligned with experimental x-axis)
            ax.plot(df_sim_aligned["Force (N)"], df_sim_aligned["V(opamp_out)"],
                   label="Circuit Simulation", color="red", marker="o", 
                   markevery=5, linewidth=2)

            # Calculate metrics using the SAME x-axis base (experimental force values)
            # Now both arrays have exactly the same length and force values
            avg_voltage_diff = self.average_percent_difference(
                sim_voltage_interp,  # Simulation voltages at experimental force points
                df_dcv_mean.values   # Experimental voltages at same force points
            )
            
            voltage_rmse = self.rmse_calculation(
                sim_voltage_interp,  # Simulation voltages at experimental force points
                df_dcv_mean.values   # Experimental voltages at same force points
            )

            # Add metrics to legend
            ax.plot([], [], ' ', label=f"Avg Diff: {avg_voltage_diff:.2f}%")
            ax.plot([], [], ' ', label=f"RMSE: {voltage_rmse:.3f} V")

            ax.set_xlim(0, 235)
            ax.legend(loc='lower right', ncol=1, frameon=True)

        fig.tight_layout()
        
        # Save plot
        plot_filename = f"{output_dir}/plot_force_vs_voltage_multi.png"
        plt.savefig(plot_filename, format='png')
        self.log.info(f"Plot saved to {plot_filename}")
        plt.close()


def parse_command_line():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Analyse sensor data")
    parser.add_argument("-V", "--version", action="version",
                       version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="count", default=0,
                       dest="verbosity", help="verbose output")
    parser.add_argument("-i", "--input", dest="input", nargs='+',
                       required=True, help="path to input files")
    parser.add_argument("-o", "--output", dest="output", default="out",
                       help="output directory (default: out)")
    
    args = vars(parser.parse_args())
    args["verbosity"] = max(0, 30 - 10 * args["verbosity"])
    return args


def main():
    """Main function"""
    try:
        cmd_args = parse_command_line()
        
        # Initialize analyzer
        analyzer = SensorAnalyzer(cmd_args['verbosity'])
        
        # Sort input files
        sorted_files = sorted(cmd_args['input'], key=SensorAnalyzer.extract_rate)
        sorted_files = sorted(sorted_files, key=SensorAnalyzer.extract_load)
        
        # Generate plot
        analyzer.plot_force_vs_voltage(sorted_files, cmd_args.get('output', 'out'))
        
        return 0
        
    except KeyboardInterrupt:
        print("Interrupted by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

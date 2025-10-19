"""
EPSC Analysis Module

This module provides functions for analyzing excitatory postsynaptic currents (EPSCs)
from CSV files and generating event tables.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Optional


def generate_event_tables_with_trace(csv_dir: str) -> None:
    """
    For each CSV in the directory, creates a simplified table with:
    - event number
    - trace number
    - event magnitude (pA)

    Saves the table as '<original_filename>_event_table_with_trace.csv'
    in the same directory.
    """
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)

            if "trace" not in df.columns or "peak amp (pA)" not in df.columns:
                print(f"Skipping {csv_file} — missing required columns.")
                continue

            event_table = pd.DataFrame({
                "event number": range(len(df)),
                "trace": df["trace"],
                "event magnitude (pA)": df["peak amp (pA)"]
            })

            base_name = os.path.splitext(os.path.basename(csv_file))[0]
            output_path = os.path.join(csv_dir, f"{base_name}_event_table_with_trace.csv")
            event_table.to_csv(output_path, index=False)
            print(f"Saved: {output_path}")

        except Exception as e:
            print(f"Error processing {csv_file}: {e}")


def plot_cumulative_column_by_trace(df: pd.DataFrame, column: str, trace_column: str, 
                                  trace_ranges: List[tuple], labels: List[str], 
                                  colors: List[str], title: str, save_path: str, 
                                  xlim: Optional[tuple] = None) -> None:
    """
    Generate and save cumulative probability plots as points filtered by trace number ranges.
    """
    plt.figure(figsize=(10, 6))

    for (start, end), label, color in zip(trace_ranges, labels, colors):
        data = df[(df[trace_column] >= start) & (df[trace_column] <= end)][column].dropna()
        if len(data) == 0:
            continue
        sorted_data = np.sort(data)
        cum_prob = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        plt.scatter(sorted_data, cum_prob, label=label, color=color, s=10)

    plt.xlabel(column)
    plt.ylabel("Cumulative Probability")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if xlim:
        plt.xlim(xlim)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def process_directory_for_trace_plots(directory_path: str, output_dir: Optional[str] = None) -> None:
    """
    Processes all CSV files in a directory and generates cumulative plots based on trace ranges
    for:
    - 'peak amp (pA)'
    - Interevent interval (ms), derived from 'Inst. Freq. (Hz)'
    """
    if output_dir is None:
        output_dir = directory_path
    
    csv_files = glob.glob(os.path.join(directory_path, "*.csv"))
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            
            # Check for required columns
            required_cols = ["trace", "peak amp (pA)", "Inst. Freq. (Hz)"]
            if not all(col in df.columns for col in required_cols):
                print(f"Skipping {csv_file} — missing required columns.")
                continue
            
            base_name = os.path.splitext(os.path.basename(csv_file))[0]
            
            # Define trace ranges and labels
            trace_ranges = [(1, 10), (11, 20), (21, 30)]
            labels = ["Traces 1-10", "Traces 11-20", "Traces 21-30"]
            colors = ["blue", "red", "green"]
            
            # Plot peak amplitude
            plot_cumulative_column_by_trace(
                df, "peak amp (pA)", "trace", trace_ranges, labels, colors,
                f"Cumulative Distribution of Peak Amplitude - {base_name}",
                os.path.join(output_dir, f"{base_name}_peak_amp_cumulative.png"),
                xlim=(0, 100)
            )
            
            # Calculate interevent interval from frequency
            df["interevent_interval_ms"] = 1000 / df["Inst. Freq. (Hz)"]
            
            # Plot interevent interval
            plot_cumulative_column_by_trace(
                df, "interevent_interval_ms", "trace", trace_ranges, labels, colors,
                f"Cumulative Distribution of Interevent Interval - {base_name}",
                os.path.join(output_dir, f"{base_name}_interevent_interval_cumulative.png"),
                xlim=(0, 2000)
            )
            
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")


def analyze_epsc_events(csv_file: str) -> Dict[str, float]:
    """
    Analyze EPSC events from a CSV file and return summary statistics.
    
    Parameters:
        csv_file (str): Path to CSV file containing EPSC data
        
    Returns:
        dict: Summary statistics including mean amplitude, frequency, etc.
    """
    try:
        df = pd.read_csv(csv_file)
        
        if "peak amp (pA)" not in df.columns:
            raise ValueError("CSV file must contain 'peak amp (pA)' column")
        
        stats = {
            "mean_amplitude": df["peak amp (pA)"].mean(),
            "std_amplitude": df["peak amp (pA)"].std(),
            "median_amplitude": df["peak amp (pA)"].median(),
            "num_events": len(df),
            "min_amplitude": df["peak amp (pA)"].min(),
            "max_amplitude": df["peak amp (pA)"].max()
        }
        
        if "Inst. Freq. (Hz)" in df.columns:
            stats["mean_frequency"] = df["Inst. Freq. (Hz)"].mean()
            stats["std_frequency"] = df["Inst. Freq. (Hz)"].std()
        
        return stats
        
    except Exception as e:
        print(f"Error analyzing EPSC events in {csv_file}: {e}")
        return {}


# Alias for backward compatibility
generate_event_tables = generate_event_tables_with_trace

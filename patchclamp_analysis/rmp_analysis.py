"""
Resting Membrane Potential (RMP) Analysis Module

This module provides functions for analyzing resting membrane potential
from current clamp recordings and holding current from voltage clamp recordings.
"""

import numpy as np
import matplotlib.pyplot as plt
import pyabf
import os
from typing import Optional, List, Tuple, Dict


def analyze_rmp(abf_file: str, time_window: Optional[List[float]] = None) -> Optional[float]:
    """
    Analyzes the resting membrane potential (RMP) from a current clamp ABF file
    and plots voltage, current, and RMP trends across sweeps.

    Parameters:
        abf_file (str): Path to the ABF file.
        time_window (list, optional): Time window [start, end] in milliseconds for RMP calculation.
                                      If None, the user is prompted to enter a time window.

    Returns:
        float: The average resting membrane potential across all sweeps.
    """
    # Load the ABF file
    abf = pyabf.ABF(abf_file)

    # Check if this is a current clamp recording
    if "mV" not in abf.sweepLabelY:
        print(f"Skipping {abf_file}: Not a current clamp recording.")
        return None

    print(f"Processing {abf_file} as a current clamp recording...")

    # Prompt user for time window if not provided
    if time_window is None:
        start_time = float(input("Enter start of time window (ms): "))
        end_time = float(input("Enter end of time window (ms): "))
        time_window = [start_time, end_time]

    print(f"Using time window: {time_window[0]} ms to {time_window[1]} ms")

    # Initialize lists to store data
    resting_membrane_potentials = []

    # Loop through all sweeps
    for sweep in range(abf.sweepCount):
        abf.setSweep(sweep)

        # Calculate RMP within the time window
        time_idx = (abf.sweepX * 1000 >= time_window[0]) & (abf.sweepX * 1000 <= time_window[1])
        rmp = np.mean(abf.sweepY[time_idx])
        resting_membrane_potentials.append(rmp)

    # Calculate the average RMP across all sweeps
    avg_rmp = np.mean(resting_membrane_potentials)

    # Print results for verification
    print(f"Resting Membrane Potentials: {resting_membrane_potentials}")
    print(f"Average RMP: {avg_rmp:.2f} mV")

    # Plot Resting Membrane Potential vs. Sweep Number
    plt.figure(figsize=(6, 4))
    sweep_numbers = np.arange(1, abf.sweepCount + 1)
    plt.plot(sweep_numbers, resting_membrane_potentials, marker="o", linestyle="-", color="g", label="RMP per Sweep")

    # Add a horizontal line for the average RMP
    plt.axhline(avg_rmp, color="black", linestyle="--", linewidth=2, label=f"Avg RMP: {avg_rmp:.2f} mV")

    # Add text annotation for average RMP
    plt.text(sweep_numbers[-1], avg_rmp + 1, f"Avg RMP: {avg_rmp:.2f} mV",
             verticalalignment="bottom", horizontalalignment="right", fontsize=10, color="black")

    plt.title("Resting Membrane Potential Over Sweeps")
    plt.ylabel("RMP (mV)")
    plt.xlabel("Sweep Number")
    plt.legend()
    plt.grid()
    plt.show()

    return avg_rmp


def analyze_holding_current_segmented(abf_file: str, time_window: List[float], sweep_window: List[int]) -> Dict[str, float]:
    """
    Analyzes holding current from a voltage clamp ABF file and returns the average 
    holding current before, during, and after an exclusion window.

    Parameters:
        abf_file (str): Path to ABF file.
        time_window (list): [start, end] in milliseconds for holding current calculation within each sweep.
        sweep_window (list): [start_sweep, end_sweep] for exclusion.

    Returns:
        dict: average holding current values for each segment.
    """
    abf = pyabf.ABF(abf_file)

    # Check if this is a voltage clamp recording
    if not any(unit in abf.sweepLabelY for unit in ["pA", "nA"]):
        raise ValueError(f"{abf_file} does not appear to be a voltage clamp recording.")

    start_time, end_time = time_window
    start_excl, end_excl = sweep_window

    before_current, during_current, after_current = [], [], []

    for sweep in range(abf.sweepCount):
        abf.setSweep(sweep)
        time_idx = (abf.sweepX * 1000 >= start_time) & (abf.sweepX * 1000 <= end_time)
        holding_current = np.mean(abf.sweepY[time_idx])

        if sweep < start_excl - 1:
            before_current.append(holding_current)
        elif start_excl - 1 <= sweep <= end_excl - 1:
            during_current.append(holding_current)
        else:
            after_current.append(holding_current)

    return {
        "before": np.mean(before_current) if before_current else 0,
        "during": np.mean(during_current) if during_current else 0,
        "after": np.mean(after_current) if after_current else 0
    }


def process_current_clamp_files(directory: str, time_window: List[float] = [200, 220], plot: bool = True) -> Dict[str, Tuple[str, float]]:
    """
    Identify and analyze all current clamp ABF files in a directory.

    Parameters:
        directory (str): The path to the folder containing ABF files.
        time_window (list): Time window [start, end] in ms for RMP calculation.
        plot (bool): Whether to generate a plot.

    Returns:
        dict: {filename: (protocol_name, avg_rmp)}
    """
    results = {}
    
    for file in os.listdir(directory):
        if file.endswith(".abf"):
            file_path = os.path.join(directory, file)
            try:
                abf = pyabf.ABF(file_path)
                
                if "mV" in abf.sweepLabelY:
                    avg_rmp = analyze_rmp(file_path, time_window)
                    if avg_rmp is not None:
                        results[file] = (abf.protocol, avg_rmp)
            except Exception as e:
                print(f"Error processing {file}: {e}")
    
    return results

"""
Plotting Module

This module provides functions for plotting patch clamp data from ABF files.
"""

import os
import pyabf
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional


def plot_current_clamp_abf(abf_path: str) -> None:
    """
    Given an ABF file, verify it is current clamp, then plot:
    - All voltage sweeps (Y) in one subplot.
    - All command currents (X) in a second subplot.
    """
    abf = pyabf.ABF(abf_path)

    # Check for current clamp mode
    if abf.adcUnits[1] != "pA":
        print(f"Skipping {os.path.basename(abf_path)}: not a current clamp file.")
        return

    # Prepare figure
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(f"Current Clamp Sweeps: {os.path.basename(abf_path)}")

    # Overlay all sweeps
    for sweep in range(abf.sweepCount):
        abf.setSweep(sweep)
        axs[0].plot(abf.sweepX * 1000, abf.sweepY, alpha=0.6)  # convert X to ms
        axs[1].plot(abf.sweepX * 1000, abf.sweepC, alpha=0.6)

    # Labels and formatting
    axs[0].set_ylabel("Membrane Voltage (mV)")
    axs[1].set_ylabel("Command Current (pA)")
    axs[1].set_xlabel("Time (ms)")
    axs[0].set_title("Voltage Responses")
    axs[1].set_title("Injected Currents")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def plot_voltage_clamp_abf(abf_path: str) -> None:
    """
    Given an ABF file, verify it is voltage clamp, then plot:
    - All current sweeps (Y) in one subplot.
    - All command voltages (X) in a second subplot.
    """
    abf = pyabf.ABF(abf_path)

    # Check for voltage clamp mode
    if not any(unit in abf.sweepLabelY for unit in ["pA", "nA"]):
        print(f"Skipping {os.path.basename(abf_path)}: not a voltage clamp file.")
        return

    # Prepare figure
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(f"Voltage Clamp Sweeps: {os.path.basename(abf_path)}")

    # Overlay all sweeps
    for sweep in range(abf.sweepCount):
        abf.setSweep(sweep)
        axs[0].plot(abf.sweepX * 1000, abf.sweepY, alpha=0.6)  # convert X to ms
        axs[1].plot(abf.sweepX * 1000, abf.sweepC, alpha=0.6)

    # Labels and formatting
    axs[0].set_ylabel("Current (pA)")
    axs[1].set_ylabel("Command Voltage (mV)")
    axs[1].set_xlabel("Time (ms)")
    axs[0].set_title("Current Responses")
    axs[1].set_title("Command Voltages")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def plot_avg_waveform(abf_path: str, channel: int = 0) -> None:
    """
    Plot the average waveform across all sweeps for a given channel.
    
    Parameters:
        abf_path (str): Path to ABF file
        channel (int): Channel to plot (default 0)
    """
    abf = pyabf.ABF(abf_path)
    
    # Collect all sweeps
    all_sweeps = []
    for sweep in range(abf.sweepCount):
        abf.setSweep(sweep, channel=channel)
        all_sweeps.append(abf.sweepY)
    
    # Calculate average
    avg_waveform = np.mean(all_sweeps, axis=0)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(abf.sweepX * 1000, avg_waveform, linewidth=2)
    plt.xlabel("Time (ms)")
    plt.ylabel(abf.sweepLabelY)
    plt.title(f"Average Waveform - {os.path.basename(abf_path)}")
    plt.grid(True)
    plt.show()


def plot_first_sweep(abf_path: str, channel: int = 0) -> None:
    """
    Plot the first sweep of an ABF file.
    
    Parameters:
        abf_path (str): Path to ABF file
        channel (int): Channel to plot (default 0)
    """
    abf = pyabf.ABF(abf_path)
    abf.setSweep(0, channel=channel)
    
    plt.figure(figsize=(10, 6))
    plt.plot(abf.sweepX * 1000, abf.sweepY)
    plt.xlabel("Time (ms)")
    plt.ylabel(abf.sweepLabelY)
    plt.title(f"First Sweep - {os.path.basename(abf_path)}")
    plt.grid(True)
    plt.show()


def plot_30ms_window(abf_path: str, start_time: float = 0, channel: int = 0) -> None:
    """
    Plot a 30ms window of data from an ABF file.
    
    Parameters:
        abf_path (str): Path to ABF file
        start_time (float): Start time in ms (default 0)
        channel (int): Channel to plot (default 0)
    """
    abf = pyabf.ABF(abf_path)
    
    # Collect all sweeps
    all_sweeps = []
    for sweep in range(abf.sweepCount):
        abf.setSweep(sweep, channel=channel)
        # Find time indices for 30ms window
        time_idx = (abf.sweepX * 1000 >= start_time) & (abf.sweepX * 1000 <= start_time + 30)
        window_data = abf.sweepY[time_idx]
        window_time = abf.sweepX[time_idx] * 1000
        all_sweeps.append((window_time, window_data))
    
    # Plot all sweeps
    plt.figure(figsize=(10, 6))
    for i, (time, data) in enumerate(all_sweeps):
        plt.plot(time, data, alpha=0.7, label=f"Sweep {i+1}")
    
    plt.xlabel("Time (ms)")
    plt.ylabel(abf.sweepLabelY)
    plt.title(f"30ms Window Starting at {start_time}ms - {os.path.basename(abf_path)}")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_membrane_voltage_vs_time_minutes(abf_path: str, channel: int = 0) -> None:
    """
    Plot membrane voltage as a function of time in minutes for current clamp recordings.
    
    Parameters:
        abf_path (str): Path to ABF file
        channel (int): Channel to plot (default 0)
    """
    abf = pyabf.ABF(abf_path)
    
    # Check if this is a current clamp recording
    if "mV" not in abf.sweepLabelY:
        print(f"Skipping {os.path.basename(abf_path)}: Not a current clamp recording.")
        return
    
    # Collect all voltage data and calculate cumulative time
    all_voltages = []
    all_times_minutes = []
    cumulative_time_minutes = 0
    
    for sweep in range(abf.sweepCount):
        abf.setSweep(sweep, channel=channel)
        
        # Convert time to minutes and add cumulative offset
        sweep_time_minutes = abf.sweepX / 60  # Convert seconds to minutes
        sweep_time_minutes += cumulative_time_minutes
        
        all_voltages.extend(abf.sweepY)
        all_times_minutes.extend(sweep_time_minutes)
        
        # Update cumulative time for next sweep
        cumulative_time_minutes += abf.sweepX[-1] / 60
    
    # Convert to numpy arrays for plotting
    all_times_minutes = np.array(all_times_minutes)
    all_voltages = np.array(all_voltages)
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(all_times_minutes, all_voltages, linewidth=0.8, alpha=0.8)
    plt.xlabel("Time (minutes)")
    plt.ylabel("Membrane Voltage (mV)")
    plt.title(f"Membrane Voltage vs Time - {os.path.basename(abf_path)}")
    plt.grid(True, alpha=0.3)
    
    # Add some statistics
    mean_voltage = np.mean(all_voltages)
    std_voltage = np.std(all_voltages)
    duration_minutes = all_times_minutes[-1] - all_times_minutes[0]
    
    plt.text(0.02, 0.98, f"Duration: {duration_minutes:.2f} min\n"
                         f"Mean Voltage: {mean_voltage:.2f} mV\n"
                         f"Std Dev: {std_voltage:.2f} mV\n"
                         f"Number of Sweeps: {abf.sweepCount}",
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    print(f"File: {os.path.basename(abf_path)}")
    print(f"Total duration: {duration_minutes:.2f} minutes")
    print(f"Mean membrane voltage: {mean_voltage:.2f} ± {std_voltage:.2f} mV")
    print(f"Number of sweeps: {abf.sweepCount}")

"""
Current-Voltage (I-V) Analysis Module

This module provides functions for analyzing current-voltage relationships
from patch clamp recordings.
"""

import numpy as np
import matplotlib.pyplot as plt
import pyabf
from typing import List, Tuple, Dict, Optional


def analyze_current_voltage_relationship(abf_file: str, voltage_range: Optional[List[float]] = None) -> Dict[str, np.ndarray]:
    """
    Analyze current-voltage relationship from a voltage clamp recording.
    
    Parameters:
        abf_file (str): Path to ABF file
        voltage_range (list, optional): Voltage range [start, end] in mV
        
    Returns:
        dict: Dictionary containing voltages, currents, and analysis results
    """
    abf = pyabf.ABF(abf_file)
    
    # Check if this is a voltage clamp recording
    if not any(unit in abf.sweepLabelY for unit in ["pA", "nA"]):
        raise ValueError(f"{abf_file} does not appear to be a voltage clamp recording.")
    
    voltages = []
    currents = []
    
    # Extract voltage and current data from each sweep
    for sweep in range(abf.sweepCount):
        abf.setSweep(sweep)
        
        # Get command voltage (usually from sweepC or protocol)
        if hasattr(abf, 'sweepC') and len(abf.sweepC) > 0:
            voltage = np.mean(abf.sweepC)
        else:
            # Fallback: try to extract from protocol
            voltage = 0  # Default if can't determine
        
        # Get steady-state current
        if voltage_range:
            start_time, end_time = voltage_range
            time_idx = (abf.sweepX * 1000 >= start_time) & (abf.sweepX * 1000 <= end_time)
            current = np.mean(abf.sweepY[time_idx])
        else:
            # Use last 20% of sweep for steady-state
            start_idx = int(0.8 * len(abf.sweepY))
            current = np.mean(abf.sweepY[start_idx:])
        
        voltages.append(voltage)
        currents.append(current)
    
    voltages = np.array(voltages)
    currents = np.array(currents)
    
    # Calculate slope (conductance)
    if len(voltages) > 1:
        slope = np.polyfit(voltages, currents, 1)[0]
        conductance = slope * 1000  # Convert to nS
    else:
        slope = 0
        conductance = 0
    
    return {
        'voltages': voltages,
        'currents': currents,
        'slope': slope,
        'conductance_nS': conductance,
        'reversal_potential': -slope * voltages[0] if slope != 0 else 0
    }


def plot_iv_curve(abf_file: str, voltage_range: Optional[List[float]] = None, 
                 save_path: Optional[str] = None) -> None:
    """
    Plot current-voltage relationship from an ABF file.
    
    Parameters:
        abf_file (str): Path to ABF file
        voltage_range (list, optional): Voltage range [start, end] in mV
        save_path (str, optional): Path to save the plot
    """
    results = analyze_current_voltage_relationship(abf_file, voltage_range)
    
    plt.figure(figsize=(8, 6))
    plt.plot(results['voltages'], results['currents'], 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Voltage (mV)')
    plt.ylabel('Current (pA)')
    plt.title(f'I-V Curve - {abf_file.split("/")[-1]}')
    plt.grid(True, alpha=0.3)
    
    # Add slope information
    plt.text(0.05, 0.95, f'Slope: {results["slope"]:.2f} pA/mV\n'
                         f'Conductance: {results["conductance_nS"]:.2f} nS\n'
                         f'Reversal Potential: {results["reversal_potential"]:.1f} mV',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def calculate_input_resistance(abf_file: str, current_injection: float, 
                             voltage_range: Optional[List[float]] = None) -> float:
    """
    Calculate input resistance from current clamp recording.
    
    Parameters:
        abf_file (str): Path to ABF file
        current_injection (float): Current injection in pA
        voltage_range (list, optional): Voltage range [start, end] in ms
        
    Returns:
        float: Input resistance in MΩ
    """
    abf = pyabf.ABF(abf_file)
    
    # Check if this is a current clamp recording
    if "mV" not in abf.sweepLabelY:
        raise ValueError(f"{abf_file} does not appear to be a current clamp recording.")
    
    voltages = []
    
    # Extract voltage response from each sweep
    for sweep in range(abf.sweepCount):
        abf.setSweep(sweep)
        
        if voltage_range:
            start_time, end_time = voltage_range
            time_idx = (abf.sweepX * 1000 >= start_time) & (abf.sweepX * 1000 <= end_time)
            voltage = np.mean(abf.sweepY[time_idx])
        else:
            # Use last 20% of sweep for steady-state
            start_idx = int(0.8 * len(abf.sweepY))
            voltage = np.mean(abf.sweepY[start_idx:])
        
        voltages.append(voltage)
    
    voltages = np.array(voltages)
    
    # Calculate voltage change
    voltage_change = np.max(voltages) - np.min(voltages)
    
    # Calculate input resistance (R = ΔV / ΔI)
    input_resistance = voltage_change / current_injection  # MΩ
    
    return input_resistance

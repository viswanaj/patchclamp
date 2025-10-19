"""
Patch Clamp Analysis Package

A comprehensive toolkit for analyzing patch clamp electrophysiology data.
"""

from .rmp_analysis import analyze_rmp, analyze_holding_current_segmented
from .epsc_analysis import process_epsc_csv_files, generate_event_tables
from .plotting import plot_current_clamp_abf, plot_voltage_clamp_abf, plot_membrane_voltage_vs_time_minutes
from .iv_analysis import analyze_current_voltage_relationship

__version__ = "1.0.0"
__author__ = "Jayashri Viswanathan"

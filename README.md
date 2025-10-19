# Patch Clamp Analysis Toolkit

A comprehensive Python package for analyzing patch clamp electrophysiology data. This toolkit provides organized modules for different types of electrophysiological analysis including resting membrane potential (RMP), excitatory postsynaptic currents (EPSCs), current-voltage relationships, and data visualization.

## Features

- **RMP Analysis**: Analyze resting membrane potential from current clamp recordings
- **EPSC Analysis**: Process and analyze excitatory postsynaptic currents from CSV data
- **I-V Analysis**: Calculate current-voltage relationships and input resistance
- **Plotting**: Comprehensive plotting functions for ABF files and analysis results
- **Machine Learning**: EPSC analysis with machine learning capabilities

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from patchclamp_analysis import analyze_rmp, plot_current_clamp_abf, analyze_epsc_events

# Analyze resting membrane potential
rmp = analyze_rmp('path/to/your/file.abf', time_window=[200, 220])

# Plot current clamp data
plot_current_clamp_abf('path/to/your/file.abf')

# Analyze EPSC events from CSV
stats = analyze_epsc_events('path/to/your/data.csv')
```

## Module Structure

- `rmp_analysis.py`: Resting membrane potential and holding current analysis
- `epsc_analysis.py`: EPSC event analysis and CSV processing
- `plotting.py`: Visualization functions for ABF files
- `iv_analysis.py`: Current-voltage relationship analysis

## Notebooks

The repository contains several Jupyter notebooks demonstrating different analysis workflows:

- `analysis.ipynb`: Main RMP analysis notebook
- `analysis-VC.ipynb`: Voltage clamp analysis
- `CC-IV.ipynb`: Current-voltage analysis
- `sEPSC_ML.ipynb`: Machine learning for EPSC analysis
- `AutomatedGranuleAvgEPSC.ipynb`: Automated EPSC analysis
- `segmented_rmp_analysis_vc.ipynb`: Segmented voltage clamp analysis

## Contributing

This toolkit consolidates multiple analysis notebooks into organized, reusable modules. The code has been cleaned up and organized for better maintainability and usability.

## License

This project is open source and available under the MIT License.

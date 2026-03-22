# Rocket-OS Simualtion Results

This directory contains the output artifacts and raw telemetry logs from the pre-publication validation of the **Resilient Autonomy** avionics architecture.

## Directory Structure

- `anomaly/` 
  Contains `detection_results.csv`, the output of the T-BAD (Transformer-Based Anomaly Detection) benchmark evaluating LSTM, Threshold, and Transformer architectures against 300 fault injection scenarios.

- `simplex/`
  Contains `simplex_log.csv`, demonstrating the Simplex safety monitor's arbitration behavior and 9-column telemetry capture showcasing the exact `t_breach` and `t_switch` sequences.

- `gnc/`
  Contains `EKF_results.csv` logging the 6-DOF tracking error of the Extended Kalman Filter against ground truth, and `trajectory.csv` capturing the 13 kinematic state variables of the G-FOLD optimization profile.

- `latency/`
  Contains scheduling traces validating the ARINC 653 partitioned stack constraint. Latency data proves that the safety wrapper adds <1.5 ms overhead and G-FOLD optimizations resolve within 67ms average bounded limits.

- `figures/`
  Contains the 17 generation G-series pipeline plots (`G01` to `G17`), including Monte Carlo landing accuracy histograms, ROC curves, architecture latency charts, and ROC evaluation matrices used in the main manuscript.

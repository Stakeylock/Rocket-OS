#!/usr/bin/env python
"""
Generate a high-fidelity 12-state baseline trajectory logs for the simulation.
Fulfills Phase 1.7 of the pre-publication roadmap:
Log all 12 state variables every step [dt=0.01].
"""

import os
import sys
import pandas as pd

# Ensure rocket_ai_os is in the path
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

from rocket_ai_os.sim.scenarios import FullMissionScenario

def main():
    print("Running Full Mission Scenario for baseline trajectory generation...")
    scenario = FullMissionScenario(inject_faults=False)
    
    # Run the full simulation until landing or crash
    result = scenario.run()
    
    # Process trajectory
    records = []
    for state in result.trajectory_log:
        records.append(state.to_12_state_dict())
        
    df = pd.DataFrame(records)
    
    # Output to results/
    out_dir = os.path.join(ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "trajectory.csv")
    
    df.to_csv(out_path, index=False)
    
    print(f"Scenario finished with success: {result.success}")
    print(f"Total steps: {len(df)}")
    if len(df) > 0:
        print(f"Mission duration: {df['t'].iloc[-1]:.2f} s")
    print(f"Metrics: {result.metrics}")
    print(f"Trajectory saved to {out_path}")

if __name__ == "__main__":
    main()

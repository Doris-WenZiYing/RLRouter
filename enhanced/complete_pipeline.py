#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import sys
import time
from pathlib import Path

def run_script(script_name, description):
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"Script: {script_name}")
    print(f"{'='*50}")
    
    if not Path(script_name).exists():
        print(f"Error: {script_name} not found")
        return False
    
    start_time = time.time()
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, timeout=3600)
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"Success: {description} completed ({elapsed:.1f}s)")
            if result.stdout:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 10:
                    print("...output truncated...")
                    print('\n'.join(lines[-10:]))
                else:
                    print(result.stdout)
            return True
        else:
            print(f"Failed: {description}")
            if result.stderr:
                print("Error:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"Timeout: {description} took too long")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    print("BFS Alpha/Beta RWA Pipeline")
    print("=" * 40)
    
    steps = [
        ("enhanced_datamake.py", "1. Generate Multi-Topology Data"),
        ("enhanced_ilp_solver.py", "2. Solve with BFS Alpha/Beta ILP"),
        ("enhanced_dnn_training.py", "3. Train BFS Alpha/Beta DNN"),
        ("enhanced_predictor.py", "4. Test and Evaluate")
    ]
    
    total_start = time.time()
    
    for script, description in steps:
        success = run_script(script, description)
        if not success:
            print(f"\nPipeline failed at: {description}")
            print(f"Try running manually: python {script}")
            return False
    
    total_time = time.time() - total_start
    
    print(f"\n{'='*50}")
    print("Checking Results:")
    print(f"{'='*50}")
    
    key_files = [
        "bfs_alpha_beta_dataset_all_topologies.csv",
        "bfs_alpha_beta_label_mappings.pkl", 
        "bfs_alpha_beta_training_results/bfs_alpha_beta_dnn_model.keras",
        "bfs_alpha_beta_scaler.pkl"
    ]
    
    all_good = True
    for file_path in key_files:
        if Path(file_path).exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path}")
            all_good = False
    
    print(f"\nTotal time: {total_time/60:.1f} minutes")
    
    if all_good:
        print("Pipeline completed successfully!")
        print("BFS Alpha/Beta system ready for use")
    else:
        print("Some files missing - check individual steps")
    
    return all_good

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPipeline cancelled by user")
    except Exception as e:
        print(f"Pipeline error: {e}")
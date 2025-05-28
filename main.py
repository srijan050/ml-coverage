import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from collections import defaultdict, Counter
import random
import time

from dut import SimpleDUT
from cds import CoverageDirectedTestSelection
from tests import TestSimulator
from exp import Experiment

def main():
    # Create experiment
    experiment = Experiment()
    
    # Run CDS method
    # Start with 100 random tests, then do 20 CDS iterations with 10 tests per iteration
    experiment.run_cds_iteration(500, 100, 50)
    
    # Run random selection with equivalent number of tests
    total_cds_tests = experiment.tests_simulated_cds
    experiment.run_random_selection(total_cds_tests)

    print(experiment.dut_cds.coverage_points)
    
    # Plot and analyze results
    plt = experiment.plot_results()
    plt.show()
    
    # Print detailed results
    print("\n--- Experiment Results ---")
    print(f"Total tests with CDS: {experiment.tests_simulated_cds}")
    print(f"Final coverage with CDS: {experiment.coverage_progress_cds[-1][1]:.2f}%")
    print(f"Total tests with Random: {experiment.tests_simulated_random}")
    print(f"Final coverage with Random: {experiment.coverage_progress_random[-1][1]:.2f}%")
    
    # Calculate tests needed to reach specific coverage levels
    coverage_levels = [90, 95, 98]
    
    print("\nTests required to reach coverage levels:")
    print("Coverage | CDS Tests | Random Tests | Savings")
    print("---------|-----------|-------------|--------")
    
    for level in coverage_levels:
        # Find tests needed for CDS
        cds_tests = None
        for tests, coverage in experiment.coverage_progress_cds:
            if coverage >= level:
                cds_tests = tests
                break
        
        # Find tests needed for Random
        random_tests = None
        for tests, coverage in experiment.coverage_progress_random:
            if coverage >= level:
                random_tests = tests
                break
        
        if cds_tests is not None and random_tests is not None:
            savings = ((random_tests - cds_tests) / random_tests) * 100
            print(f"{level:7}% | {cds_tests:9} | {random_tests:11} | {savings:6.2f}%")
        else:
            print(f"{level:7}% | {'N/A':9} | {'N/A':11} | {'N/A':6}")

if __name__ == "__main__":
    main()
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

class Experiment:
    """
    Framework to run CDS experiments and compare with random selection
    """
    def __init__(self, dut_class=SimpleDUT):
        # Create DUT for CDS and random testing
        self.dut_cds = dut_class()
        self.dut_random = dut_class()
        
        # Create simulators
        self.simulator_cds = TestSimulator(self.dut_cds)
        self.simulator_random = TestSimulator(self.dut_random)
        
        # Create CDS controller
        self.cds = CoverageDirectedTestSelection(self.dut_cds, self.simulator_cds)
        
        # Generate a large pool of tests for selection
        self.test_pool = self.generate_test_pool(10000)
        
        # Performance tracking
        self.coverage_progress_cds = []
        self.coverage_progress_random = []
        self.tests_simulated_cds = 0
        self.tests_simulated_random = 0
    
    def generate_test_pool(self, num_tests):
        """Generate a pool of tests to select from"""
        tests = []
        for _ in range(num_tests):
            # Create test with random values for each field
            test = {
                'input_interface': random.choice(self.dut_cds.input_interface_values),
                'data_size': random.choice(self.dut_cds.data_size_values),
                'output_active': random.choice(self.dut_cds.output_active_values),
                'data_bin': random.randint(0, self.dut_cds.data_bin_max)
            }
            tests.append(test)
        return tests
    
    def run_random_selection(self, num_tests):
        """Run verification using random test selection"""
        print(f"\nRunning Random Selection with {num_tests} tests...")
        
        # Randomly select tests from pool
        selected_tests = random.sample(self.test_pool, num_tests)
        
        # Simulate tests
        self.simulator_random.test_database.extend(selected_tests)
        self.simulator_random.simulate_tests(selected_tests)
        
        # Update metrics
        self.tests_simulated_random += num_tests
        coverage = self.dut_random.get_coverage_percentage()
        self.coverage_progress_random.append((self.tests_simulated_random, coverage))
        
        print(f"  Coverage after Random Selection: {coverage:.2f}%")
        return coverage
    
    def run_cds_iteration(self, num_initial_tests, num_iterations, tests_per_iteration):
        """
        Run CDS verification process
        - Start with random tests until initial threshold
        - Then apply CDS for subsequent iterations
        """
        print(f"\nRunning CDS with {num_initial_tests} initial tests + {num_iterations} CDS iterations...")
        
        # Phase 1: Initial random testing
        initial_tests = random.sample(self.test_pool, num_initial_tests)
        self.simulator_cds.test_database.extend(initial_tests)
        self.simulator_cds.simulate_tests(initial_tests)
        
        # Update metrics
        self.tests_simulated_cds += num_initial_tests
        coverage = self.dut_cds.get_coverage_percentage()
        self.coverage_progress_cds.append((self.tests_simulated_cds, coverage))
        
        print(f"  Coverage after initial random testing: {coverage:.2f}%")
        
        # Phase 2: CDS iterations
        for iteration in range(num_iterations):
            print(f"\nCDS Iteration {iteration+1}/{num_iterations}")
            
            # Train classifiers based on current results
            self.cds.train_classifiers_for_uncovered_groups()
            
            # Get candidate tests (those not already in test_database)
            candidate_tests = self.test_pool.copy()
            
            # Select promising tests using CDS
            selected_indices = self.cds.select_next_tests(candidate_tests, tests_per_iteration)
            selected_tests = [candidate_tests[i] for i in selected_indices]
            
            # Simulate selected tests
            self.simulator_cds.test_database.extend(selected_tests)
            self.simulator_cds.simulate_tests(selected_tests)
            
            # Update metrics
            self.tests_simulated_cds += len(selected_tests)
            coverage = self.dut_cds.get_coverage_percentage()
            self.coverage_progress_cds.append((self.tests_simulated_cds, coverage))
            
            print(f"  Coverage after CDS iteration {iteration+1}: {coverage:.2f}%")
            
            # If we've reached 100% coverage, we can stop
            if coverage >= 99.99:
                print("  Reached (approximately) 100% coverage. Stopping CDS iterations early.")
                break
        
        return coverage
    
    def plot_results(self):
        """Plot coverage progress for both methods"""
        plt.figure(figsize=(10, 6))
        
        # Plot CDS coverage progress
        cds_tests, cds_coverage = zip(*self.coverage_progress_cds)
        plt.plot(cds_tests, cds_coverage, 'b-o', label='CDS')
        
        # Plot Random coverage progress
        random_tests, random_coverage = zip(*self.coverage_progress_random)
        plt.plot(random_tests, random_coverage, 'r-o', label='Random')
        
        plt.xlabel('Number of Tests Simulated')
        plt.ylabel('Coverage (%)')
        plt.title('Coverage Progress: CDS vs Random Selection')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Calculate and display savings
        if self.coverage_progress_random and self.coverage_progress_cds:
            final_random_coverage = self.coverage_progress_random[-1][1]
            
            # Find how many CDS tests were needed to reach the same coverage
            tests_for_same_coverage = None
            for tests, coverage in self.coverage_progress_cds:
                if coverage >= final_random_coverage:
                    tests_for_same_coverage = tests
                    break
            
            if tests_for_same_coverage is not None:
                saving_percentage = ((self.tests_simulated_random - tests_for_same_coverage) / 
                                    self.tests_simulated_random) * 100
                
                plt.text(0.5, 0.01, 
                         f'CDS saved {saving_percentage:.1f}% of tests to reach {final_random_coverage:.1f}% coverage',
                         transform=plt.gca().transAxes,
                         ha='center', va='bottom',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        return plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from collections import defaultdict, Counter
import random
import time

class TestSimulator:
    """
    Handles test generation and simulation for the DUT
    """
    def __init__(self, dut):
        self.dut = dut
        self.test_database = []  # Store generated tests
        self.coverage_database = {}  # Store coverage results for each test
        
    def generate_test_stimuli(self, num_tests):
        """Generate a collection of test stimuli"""
        tests = []
        for _ in range(num_tests):
            # Create test with random values for each field
            test = {
                'input_interface': random.choice(self.dut.input_interface_values),
                'data_size': random.choice(self.dut.data_size_values),
                'output_active': random.choice(self.dut.output_active_values),
                'data_bin': random.randint(0, self.dut.data_bin_max)
            }
            tests.append(test)
        
        # Add tests to test database
        for test in tests:
            self.test_database.append(test)
        
        return tests
    
    def simulate_test(self, test):
        """
        Simulate a single test and determine which coverage points it hits
        This is a simplified model of how the test might exercise certain functionality
        """
        # Get test parameters
        input_interface = test['input_interface']
        data_size = test['data_size']
        output_active = test['output_active']
        data_bin = test['data_bin']
        
        # Track which coverage points are hit by this test
        hit_points = []
        
        # Check GROUP1 coverage points (Memory interface)
        if input_interface == 0:
            for ds in self.dut.data_size_values:
                if ds == data_size:
                    for out_act in self.dut.output_active_values:
                        if out_act == output_active:
                            # Check if data_bin falls within defined ranges
                            for bin_range in [(0, 100), (101, 500), (501, 1000)]:
                                if bin_range[0] <= data_bin <= bin_range[1]:
                                    point_id = f"g1_iface0_ds{ds}_out{out_act}_bin{bin_range[0]}-{bin_range[1]}"
                                    hit_points.append(point_id)
        
        # Check GROUP2 coverage points (Radar interface)
        if input_interface == 1:
            for ds in self.dut.data_size_values:
                if ds == data_size:
                    for out_act in self.dut.output_active_values:
                        if out_act == output_active:
                            # Check if data_bin falls within defined ranges
                            for bin_range in [(0, 200), (201, 1000), (1001, 5000)]:
                                if bin_range[0] <= data_bin <= bin_range[1]:
                                    point_id = f"g2_iface1_ds{ds}_out{out_act}_bin{bin_range[0]}-{bin_range[1]}"
                                    hit_points.append(point_id)
        
        # Check GROUP3 coverage points (Special cases)
        if (input_interface == 1 and data_size == 4) or (input_interface == 0 and data_size == 3):
            if 5001 <= data_bin <= 10000:
                point_id = f"g3_iface{input_interface}_ds{data_size}_special_bin5001-10000"
                hit_points.append(point_id)
        
        # Update DUT coverage model
        for point in hit_points:
            if point in self.dut.coverage_points:
                self.dut.coverage_points[point] = True
        
        # Store test coverage result in database
        test_id = len(self.coverage_database)
        self.coverage_database[test_id] = hit_points
        
        return hit_points
    
    def simulate_tests(self, tests):
        """Simulate a batch of tests"""
        results = []
        for test in tests:
            hit_points = self.simulate_test(test)
            results.append(hit_points)
        return results
    
    def get_test_features(self):
        """Convert test database to feature matrix for ML"""
        features = []
        for test in self.test_database:
            features.append([
                test['input_interface'],
                test['data_size'],
                test['output_active'],
                test['data_bin']
            ])
        return np.array(features)
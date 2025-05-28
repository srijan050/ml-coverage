import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from collections import defaultdict, Counter
import random
import time

class SimpleDUT:
    """
    A simplified Device Under Test (DUT) representing a basic signal processor
    with configuration fields similar to those mentioned in the paper.
    """
    def __init__(self):
        # Define possible values for each configuration field
        self.input_interface_values = [0, 1]  # 0: MEM, 1: Radar
        self.data_size_values = [1, 2, 3, 4]
        self.output_active_values = [0, 1]
        self.data_bin_max = 10000  # Simplified range for data_bin
        
        # Initialize coverage model
        self.initialize_coverage_model()
    
    def initialize_coverage_model(self):
        """Initialize the coverage model with coverage points and groups"""
        # Dictionary to track which coverage points have been hit
        self.coverage_points = {}
        
        # Define coverage groups (simplified version)
        self.coverage_groups = {}
        
        # Group 1: Input Interface = 0 (Memory) related functionality
        group1_points = []
        for ds in self.data_size_values:
            for out_act in self.output_active_values:
                # Define ranges for data_bin that would exercise these points
                for bin_range in [(0, 100), (101, 500), (501, 1000)]:
                    point_id = f"g1_iface0_ds{ds}_out{out_act}_bin{bin_range[0]}-{bin_range[1]}"
                    self.coverage_points[point_id] = False
                    group1_points.append(point_id)
        self.coverage_groups["GROUP1"] = group1_points
        
        # Group 2: Input Interface = 1 (Radar) related functionality
        group2_points = []
        for ds in self.data_size_values:
            for out_act in self.output_active_values:
                # Define ranges for data_bin that would exercise these points
                for bin_range in [(0, 200), (201, 1000), (1001, 5000)]:
                    point_id = f"g2_iface1_ds{ds}_out{out_act}_bin{bin_range[0]}-{bin_range[1]}"
                    self.coverage_points[point_id] = False
                    group2_points.append(point_id)
        self.coverage_groups["GROUP2"] = group2_points
        
        # Group 3: Cross-product functionality (complex interactions)
        group3_points = []
        for iface in self.input_interface_values:
            for ds in [3, 4]:  # Only larger data sizes
                if iface == 1 and ds == 4:  # Special radar high-data case
                    for bin_range in [(5001, 10000)]:
                        point_id = f"g3_iface{iface}_ds{ds}_special_bin{bin_range[0]}-{bin_range[1]}"
                        self.coverage_points[point_id] = False
                        group3_points.append(point_id)
        self.coverage_groups["GROUP3"] = group3_points
        
        # Metadata about coverage model
        self.total_coverage_points = len(self.coverage_points)
        self.coverage_per_group = {group: len(points) for group, points in self.coverage_groups.items()}
        
    def get_coverage_percentage(self):
        """Calculate the current coverage percentage"""
        covered = sum(1 for p in self.coverage_points.values() if p)
        return (covered / self.total_coverage_points) * 100
    
    def get_group_coverage_percentage(self, group_name):
        """Calculate coverage percentage for a specific group"""
        group_points = self.coverage_groups[group_name]
        covered = sum(1 for p in group_points if self.coverage_points[p])
        return (covered / len(group_points)) * 100
    
    def reset_coverage(self):
        """Reset all coverage points to not covered"""
        for point in self.coverage_points:
            self.coverage_points[point] = False
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from collections import defaultdict, Counter
import random
import time


class CoverageDirectedTestSelection:
    """
    Implements the Coverage-Directed Test Selection methodology using Random Forest
    """
    def __init__(self, dut, simulator):
        self.dut = dut
        self.simulator = simulator
        self.classifiers = {}  # One classifier per coverage group
    
    def get_uncovered_groups(self, threshold=100.0):
        """Get list of coverage groups that have not reached the threshold coverage"""
        uncovered_groups = []
        for group in self.dut.coverage_groups.keys():
            coverage = self.dut.get_group_coverage_percentage(group)
            if coverage < threshold:
                uncovered_groups.append(group)
        return uncovered_groups
    
    def prepare_training_data_for_group(self, group_name):
        """
        Prepare training data for a specific coverage group
        This implements the technique from the paper where we create a balanced dataset
        """
        # Get coverage points for this group
        group_points = self.dut.coverage_groups[group_name]
        
        # Find which tests hit at least one point in this group
        positive_examples = []
        for test_id, hit_points in self.simulator.coverage_database.items():
            if any(point in hit_points for point in group_points):
                positive_examples.append(test_id)
        
        # Find tests that didn't hit any points in this group (negative examples)
        negative_examples = []
        for test_id in range(len(self.simulator.test_database)):
            if test_id in self.simulator.coverage_database:
                hit_points = self.simulator.coverage_database[test_id]
                if not any(point in hit_points for point in group_points):
                    negative_examples.append(test_id)
        
        # Balance the dataset by sampling equal numbers of positive and negative examples
        num_samples = min(len(positive_examples), len(negative_examples))
        if num_samples == 0:
            # If no positive examples, we can't train a model
            return None, None, None
        
        if len(positive_examples) > num_samples:
            positive_examples = random.sample(positive_examples, num_samples)
        if len(negative_examples) > num_samples:
            negative_examples = random.sample(negative_examples, num_samples)
        
        # Get features for sampled tests
        X_positive = []
        for test_id in positive_examples:
            test = self.simulator.test_database[test_id]
            X_positive.append([
                test['input_interface'],
                test['data_size'],
                test['output_active'],
                test['data_bin']
            ])
        
        X_negative = []
        for test_id in negative_examples:
            test = self.simulator.test_database[test_id]
            X_negative.append([
                test['input_interface'],
                test['data_size'],
                test['output_active'],
                test['data_bin']
            ])
        
        # Combine positive and negative examples
        X = np.vstack((X_positive, X_negative))
        y = np.hstack((np.ones(len(X_positive)), np.zeros(len(X_negative))))
        
        # Track which tests were used for training
        used_test_ids = positive_examples + negative_examples
        
        return X, y, used_test_ids
    
    def train_classifiers_for_uncovered_groups(self):
        """Train classifiers for all uncovered coverage groups"""
        uncovered_groups = self.get_uncovered_groups()
        print(f"Training classifiers for {len(uncovered_groups)} uncovered groups")
        
        for group in uncovered_groups:
            # Prepare training data
            X, y, used_test_ids = self.prepare_training_data_for_group(group)
            
            if X is None or len(X) == 0:
                print(f"  No training data available for group {group}")
                continue
            
            print(f"  Training Random Forest for group {group} with {len(X)} examples")
            
            # Train Random Forest classifier
            clf = RandomForestClassifier(max_depth=3, random_state=42)
            clf.fit(X, y)
            
            # Store trained classifier
            self.classifiers[group] = clf
    
    def select_next_tests(self, candidate_tests, max_tests):
        """
        Select most promising tests based on classifier predictions
        Returns a list of selected test indices from candidate_tests
        """
        if not self.classifiers:
            print("No classifiers trained yet. Selecting random tests.")
            return random.sample(range(len(candidate_tests)), min(max_tests, len(candidate_tests)))
        
        # Get uncovered groups
        uncovered_groups = self.get_uncovered_groups()
        
        if not uncovered_groups:
            print("All groups have reached coverage target. Selecting random tests.")
            return random.sample(range(len(candidate_tests)), min(max_tests, len(candidate_tests)))
        
        # Convert candidate tests to feature matrix
        X_candidates = np.array([[
            test['input_interface'],
            test['data_size'],
            test['output_active'],
            test['data_bin']
        ] for test in candidate_tests])
        
        # For each uncovered group, use classifier to predict most promising tests
        selected_indices = []
        
        # Try to select one test per uncovered group
        for group in uncovered_groups[:max_tests]:
            if group in self.classifiers:
                clf = self.classifiers[group]
                
                # Get probability estimates for positive class
                probabilities = clf.predict_proba(X_candidates)[:, 1]
                
                # Find test with highest probability that hasn't been selected yet
                for _ in range(len(candidate_tests)):
                    if len(probabilities) == 0:
                        break
                    
                    best_idx = np.argmax(probabilities)
                    
                    if best_idx not in selected_indices:
                        selected_indices.append(best_idx)
                        break
                    else:
                        # If already selected, set probability to -1 to ignore in next argmax
                        probabilities[best_idx] = -1
        
        # If we still haven't selected enough tests, pick randomly
        remaining = max_tests - len(selected_indices)
        if remaining > 0:
            available_indices = [i for i in range(len(candidate_tests)) if i not in selected_indices]
            if available_indices:
                additional_indices = random.sample(available_indices, min(remaining, len(available_indices)))
                selected_indices.extend(additional_indices)
        
        return selected_indices
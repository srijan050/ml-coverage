# ML Coverage - Coverage-Directed Test Selection Framework

A machine learning-based framework for intelligent test selection in hardware verification, implementing Coverage-Directed Test Selection (CDS) methodology using Random Forest classifiers to optimize coverage efficiency.

## Overview

This system implements an ML-driven approach to hardware verification test selection that significantly reduces the number of tests needed to achieve target coverage compared to random selection. The framework uses Random Forest classifiers to predict which tests are most likely to hit uncovered functionality, enabling more efficient verification workflows.

## What It Does

The ML Coverage framework addresses the challenge of efficient test selection in hardware verification by:

- **Intelligent Test Selection**: Uses machine learning to predict which tests will hit uncovered coverage points
- **Coverage Tracking**: Maintains detailed coverage models with grouped coverage points
- **Comparative Analysis**: Benchmarks CDS performance against random test selection
- **Automated Training**: Dynamically trains classifiers based on test execution results

## How It Works

### Core Architecture

The system consists of four main components:

1. **SimpleDUT**: Device Under Test model with coverage tracking
2. **TestSimulator**: Test generation and execution engine  
3. **CoverageDirectedTestSelection**: ML-based test selection controller
4. **Experiment**: Framework for comparing CDS vs random selection

### Coverage Model

The system defines three coverage groups with specific criteria:

- **GROUP1**: Memory interface functionality (`input_interface == 0`)
- **GROUP2**: Radar interface functionality (`input_interface == 1`) 
- **GROUP3**: Special cross-product cases for complex interactions

### Machine Learning Pipeline

The CDS algorithm works through these steps:

1. **Initial Random Phase**: Executes random tests to build training data
2. **Classifier Training**: Trains Random Forest models for each uncovered coverage group
3. **Intelligent Selection**: Uses trained models to predict most promising tests
4. **Iterative Refinement**: Continuously updates models as new coverage data becomes available

### Feature Engineering

Each test case is represented as a 4-dimensional feature vector:
- `input_interface` (0-1): Memory vs Radar interface
- `data_size` (1-4): Data size configuration  
- `output_active` (0-1): Output activation state
- `data_bin` (0-10000): Data bin value

## Installation & Setup

### Prerequisites

```
pip install -r requirements.txt
```

## How to Run

### Basic Execution

Run the complete experiment comparing CDS vs random selection:

```
python main.py
```

This executes the default configuration:
- 500 initial random tests
- 100 CDS iterations  
- 50 tests per iteration
- Comparative analysis and visualization

### Custom Experiment Configuration

You can customize the experiment parameters by modifying the values in `main.py`:

```python
# Adjust these parameters for different experiment configurations
experiment.run_cds_iteration(
    num_initial_tests=500,    # Initial random tests
    num_iterations=100,       # CDS iterations
    tests_per_iteration=50    # Tests per CDS iteration
)
```

### Programmatic Usage

For custom implementations, use the core classes directly:

```python
from dut import SimpleDUT
from tests import TestSimulator
from cds import CoverageDirectedTestSelection

# Initialize components
dut = SimpleDUT()
simulator = TestSimulator(dut)
cds = CoverageDirectedTestSelection(dut, simulator)

# Generate and simulate initial tests
tests = simulator.generate_test_stimuli(100)
simulator.simulate_tests(tests)

# Train classifiers and select next tests
cds.train_classifiers_for_uncovered_groups()
candidate_tests = simulator.generate_test_stimuli(1000)
selected_indices = cds.select_next_tests(candidate_tests, 10)
```

## Output and Results

### Visualization

The system generates comparative plots showing coverage progress over time for both CDS and random selection methods.
Example run:
![image](https://github.com/user-attachments/assets/70ad13b1-eb04-4aa5-8442-c166491812d3)


### Performance Metrics

The framework provides detailed analysis including:

- Total tests executed for each method
- Final coverage percentages achieved
- Test savings at different coverage levels (90%, 95%, 98%)
- Efficiency comparisons between CDS and random selection

### Sample Output

![image](https://github.com/user-attachments/assets/6050d8eb-2fc3-4cf0-9c9d-b7db96ec22e3)


## Configuration Options

### Coverage Model Parameters

The DUT configuration can be customized by modifying the values in `SimpleDUT`

### ML Algorithm Parameters

Random Forest classifier settings can be adjusted, or it can be replaced by another ML Algorithm like Decision Trees, Gradient Boosting or Neural Networks.

## Performance Benefits

The CDS methodology typically achieves:
- 30-70% reduction in tests needed to reach target coverage
- More efficient exploration of hard-to-reach coverage points
- Faster convergence to high coverage levels
- Better resource utilization in verification workflows

## Conclusion

This implementation serves as a research framework demonstrating the effectiveness of ML-based test selection in hardware verification. The simplified DUT model and coverage groups provide a foundation that can be extended for more complex verification scenarios. The Random Forest classifier choice balances prediction accuracy with training efficiency, making it suitable for iterative verification workflows where models need frequent retraining as new coverage data becomes available.

Wiki pages you might want to explore:
- [Technical Reference (srijan050/ml-coverage)](https://deepwiki.com/srijan050/ml-coverage)

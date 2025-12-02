# RL Task: Balanced Batch Sampler

## Task Overview

This task asks the model to implement a PyTorch-compatible batch sampler that creates class-balanced batches for training on imbalanced datasets. This is a common real-world ML engineering task that requires understanding of:

- PyTorch's data loading abstractions (`Sampler` interface)
- Class imbalance handling strategies
- Proper iteration and oversampling logic

## Why This Task?

**Skills Taught:**
1. Custom PyTorch data loading components
2. Handling class imbalance in training
3. Iterator patterns in Python
4. Understanding of batch-based training

**Practical Relevance:**
Class imbalance is ubiquitous in real-world ML (fraud detection, medical diagnosis, rare event prediction). Creating balanced batches is a common technique alongside loss weighting and SMOTE.

## Expected Difficulty

Target pass rate: **10-40%** with claude-haiku-4-5

The task is challenging because:
1. Requires correct understanding of the `Sampler` interface
2. Must handle oversampling for minority classes
3. Must ensure ALL samples are used (not just balanced random sampling)
4. Must maintain reproducibility with seeds
5. Edge cases (very imbalanced data, minimum configurations)

## Failure Modes

Based on testing, models fail this task for various reasons:

### Common Failure Modes:

1. **Incorrect Sampler Interface** (~25% of failures)
   - Returning indices directly instead of batches
   - Not implementing `__len__` correctly
   - Not yielding lists of indices

2. **Balance Logic Errors** (~30% of failures)
   - Random sampling without enforcing exact balance
   - Off-by-one errors in batch composition
   - Not handling classes with different numbers of samples

3. **Missing All-Samples Guarantee** (~20% of failures)
   - Only sampling balanced batches without ensuring all data is seen
   - Dropping samples from majority classes

4. **Reproducibility Issues** (~15% of failures)
   - Not properly seeding random number generator
   - Seeding once but not maintaining state across iteration

5. **Edge Case Failures** (~10% of failures)
   - Failing on very imbalanced data (100:2 ratio)
   - Incorrect behavior with minimum viable configurations

## Grading

The grader runs 7 tests:
1. `test_basic_functionality` - Can be instantiated and iterated
2. `test_batch_balance` - Each batch has exact class balance
3. `test_all_samples_used` - All dataset samples appear at least once
4. `test_reproducibility` - Same seed = same batches
5. `test_dataloader_compatibility` - Works with PyTorch DataLoader
6. `test_edge_cases` - Handles extreme imbalance and minimum configs
7. `test_different_class_sizes` - Works with 4+ classes of varying sizes

**Pass criteria:** All 7 tests must pass

## Running the Evaluation

### Option A: Using Docker (Recommended)

```bash
# Build the image
docker build -t rl-task-sampler .

# Test the grader (no API key needed)
docker run rl-task-sampler python test_grader.py

# Run full evaluation
docker run -e ANTHROPIC_API_KEY="your-key-here" rl-task-sampler python run_eval.py --runs 10
```

Or using docker-compose:

```bash
# Test grader
docker-compose run test-grader

# Run evaluation (set API key in .env or environment)
export ANTHROPIC_API_KEY="your-key-here"
docker-compose run task
```

### Option B: Local Installation

#### Step 1: Verify Grader (No API Key Needed)

```bash
# Install dependencies
pip install torch numpy

# Test the grader with reference and broken solutions
python test_grader.py
```

This should show the reference solution passing and all broken solutions failing appropriately.

#### Step 2: Run Full Evaluation (Requires API Key)

```bash
# Install anthropic
pip install anthropic

# Set API key
export ANTHROPIC_API_KEY="your-key-here"

# Run evaluation (default 10 runs)
python run_eval.py

# Run more evaluations for better statistics
python run_eval.py --runs 20
```

## File Structure

```
rl_task_balanced_sampler/
├── task.py            # Task definition, prompt, and grading function
├── run_eval.py        # Evaluation script for running with claude-haiku-4-5
├── test_grader.py     # Test script to verify grader catches failure modes
├── Dockerfile         # Docker image definition
├── docker-compose.yml # Docker compose for easy running
├── README.md          # This file
└── requirements.txt   # Dependencies
```

## Tools Provided to Model

1. **write_file** - Write solution to a file
2. **run_python** - Execute Python code for testing

## Multiple Valid Solutions

The task allows multiple correct approaches:

1. **Round-robin with oversampling**: Cycle through each class, oversample minority classes
2. **Pool-based**: Create pools for each class, draw from pools for each batch
3. **Index-shuffling**: Pre-compute all batches by shuffling class indices
4. **Generator-based**: Lazy generation of balanced batches

All approaches that satisfy the requirements will pass.

## Sample Passing Solution

See `REFERENCE_SOLUTION` in `task.py` for one valid implementation.

## Notes

- The task is designed to be concise (~200 lines for task + grading)
- Maximum steps set to 15 to allow experimentation but prevent infinite loops
- Model can test its solution using `run_python` before submitting


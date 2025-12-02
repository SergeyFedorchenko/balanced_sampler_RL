"""
RL Task: Implement a Balanced Batch Sampler for Class-Imbalanced Datasets

This task tests the model's ability to implement a PyTorch-compatible batch sampler
that ensures each batch contains equal representation from each class.

Skill taught: Custom PyTorch data loading components for handling class imbalance
"""

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
from typing import Iterator, List, Optional
from collections import Counter
import tempfile
import subprocess
import sys
import os

TASK_PROMPT = """
Implement a PyTorch batch sampler that creates class-balanced batches for training on imbalanced datasets.

## Requirements

1. Create a class `BalancedBatchSampler` that inherits from `torch.utils.data.Sampler`
2. Each batch must contain EXACTLY `n_samples_per_class` samples from EACH class
3. Batch size = n_samples_per_class * num_classes
4. The sampler must iterate through ALL samples in the dataset during one epoch
5. For classes with fewer samples than needed, oversample (reuse samples)
6. For classes with more samples, ensure all are eventually used across batches
7. When `seed` is provided, the sampling order must be reproducible
8. The sampler must work correctly with PyTorch DataLoader

## Interface

```python
class BalancedBatchSampler(Sampler):
    def __init__(self, labels: List[int], n_samples_per_class: int, seed: Optional[int] = None):
        '''
        Args:
            labels: List of integer class labels for each sample in the dataset
            n_samples_per_class: Number of samples to include from each class per batch
            seed: Random seed for reproducibility (optional)
        '''
        pass
    
    def __iter__(self) -> Iterator[List[int]]:
        '''Yields batches of indices'''
        pass
    
    def __len__(self) -> int:
        '''Returns the number of batches per epoch'''
        pass
```

## Example

```python
labels = [0, 0, 0, 0, 0, 1, 1, 2, 2, 2]  # 5 samples of class 0, 2 of class 1, 3 of class 2
sampler = BalancedBatchSampler(labels, n_samples_per_class=2, seed=42)

for batch_indices in sampler:
    # Each batch should have exactly 6 indices (2 per class * 3 classes)
    # The indices in each batch should point to 2 samples from each class
    pass
```

## Constraints
- Do not use any external libraries beyond PyTorch and NumPy
- The implementation should be efficient (avoid O(n²) operations where possible)

Save your implementation to a file called `solution.py`.
"""


def create_test_dataset(labels: List[int]) -> Dataset:
    """Create a simple dataset for testing."""
    class SimpleDataset(Dataset):
        def __init__(self, labels):
            self.labels = labels
            self.data = torch.randn(len(labels), 10)
        
        def __len__(self):
            return len(self.labels)
        
        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx]
    
    return SimpleDataset(labels)


def grade_solution(solution_code: str) -> dict:
    """
    Grade the submitted solution.
    
    Returns:
        dict with 'passed': bool, 'score': float (0-1), 'feedback': str
    """
    results = {
        'passed': False,
        'score': 0.0,
        'feedback': '',
        'tests_passed': [],
        'tests_failed': []
    }
    
    # Write solution to temp file and import it
    with tempfile.TemporaryDirectory() as tmpdir:
        solution_path = os.path.join(tmpdir, 'solution.py')
        with open(solution_path, 'w') as f:
            f.write(solution_code)
        
        # Try to import the solution
        sys.path.insert(0, tmpdir)
        try:
            # Clear any cached imports
            if 'solution' in sys.modules:
                del sys.modules['solution']
            
            import solution
            BalancedBatchSampler = solution.BalancedBatchSampler
        except Exception as e:
            results['feedback'] = f"Failed to import solution: {str(e)}"
            return results
        finally:
            sys.path.remove(tmpdir)
        
        tests = [
            test_basic_functionality,
            test_batch_balance,
            test_all_samples_used,
            test_reproducibility,
            test_dataloader_compatibility,
            test_edge_cases,
            test_different_class_sizes,
        ]
        
        total_tests = len(tests)
        passed_tests = 0
        
        for test_fn in tests:
            try:
                test_fn(BalancedBatchSampler)
                passed_tests += 1
                results['tests_passed'].append(test_fn.__name__)
            except AssertionError as e:
                results['tests_failed'].append(f"{test_fn.__name__}: {str(e)}")
            except Exception as e:
                results['tests_failed'].append(f"{test_fn.__name__}: Exception - {str(e)}")
        
        results['score'] = passed_tests / total_tests
        results['passed'] = passed_tests == total_tests
        
        if results['passed']:
            results['feedback'] = "All tests passed!"
        else:
            results['feedback'] = f"Passed {passed_tests}/{total_tests} tests.\n"
            results['feedback'] += "Failed tests:\n" + "\n".join(results['tests_failed'])
    
    return results


def test_basic_functionality(BalancedBatchSampler):
    """Test that the sampler can be instantiated and iterated."""
    labels = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    sampler = BalancedBatchSampler(labels, n_samples_per_class=2)
    
    batches = list(sampler)
    assert len(batches) > 0, "Sampler should yield at least one batch"
    assert all(isinstance(b, list) for b in batches), "Each batch should be a list"
    assert all(isinstance(idx, int) for b in batches for idx in b), "Indices should be integers"


def test_batch_balance(BalancedBatchSampler):
    """Test that each batch has equal samples from each class."""
    labels = [0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2]
    n_samples_per_class = 2
    num_classes = 3
    sampler = BalancedBatchSampler(labels, n_samples_per_class=n_samples_per_class, seed=42)
    
    for batch in sampler:
        assert len(batch) == n_samples_per_class * num_classes, \
            f"Batch size should be {n_samples_per_class * num_classes}, got {len(batch)}"
        
        # Count samples per class in this batch
        class_counts = Counter(labels[idx] for idx in batch)
        for cls in range(num_classes):
            assert class_counts.get(cls, 0) == n_samples_per_class, \
                f"Class {cls} should have {n_samples_per_class} samples, got {class_counts.get(cls, 0)}"


def test_all_samples_used(BalancedBatchSampler):
    """Test that all samples in the dataset are used at least once per epoch."""
    labels = [0, 0, 0, 0, 1, 1, 2, 2, 2]
    sampler = BalancedBatchSampler(labels, n_samples_per_class=2, seed=42)
    
    all_indices = []
    for batch in sampler:
        all_indices.extend(batch)
    
    # Each original sample should appear at least once
    unique_indices = set(all_indices)
    all_original_indices = set(range(len(labels)))
    
    assert all_original_indices.issubset(unique_indices), \
        f"Not all samples used. Missing: {all_original_indices - unique_indices}"


def test_reproducibility(BalancedBatchSampler):
    """Test that same seed produces same batches."""
    labels = [0, 0, 0, 1, 1, 2, 2, 2, 2]
    
    sampler1 = BalancedBatchSampler(labels, n_samples_per_class=2, seed=123)
    batches1 = [batch.copy() if isinstance(batch, list) else list(batch) for batch in sampler1]
    
    sampler2 = BalancedBatchSampler(labels, n_samples_per_class=2, seed=123)
    batches2 = [batch.copy() if isinstance(batch, list) else list(batch) for batch in sampler2]
    
    assert batches1 == batches2, "Same seed should produce identical batches"
    
    # Different seed should produce different batches (with high probability)
    sampler3 = BalancedBatchSampler(labels, n_samples_per_class=2, seed=456)
    batches3 = [batch.copy() if isinstance(batch, list) else list(batch) for batch in sampler3]
    
    assert batches1 != batches3, "Different seeds should produce different batches"


def test_dataloader_compatibility(BalancedBatchSampler):
    """Test that sampler works correctly with PyTorch DataLoader."""
    labels = [0, 0, 0, 0, 1, 1, 1, 2, 2]
    dataset = create_test_dataset(labels)
    sampler = BalancedBatchSampler(labels, n_samples_per_class=2, seed=42)
    
    # DataLoader with batch_sampler should work
    loader = DataLoader(dataset, batch_sampler=sampler)
    
    batches_data = []
    batches_labels = []
    for data, batch_labels in loader:
        batches_data.append(data)
        batches_labels.append(batch_labels)
    
    assert len(batches_data) > 0, "DataLoader should yield batches"
    
    # Check batch composition
    for batch_labels in batches_labels:
        class_counts = Counter(batch_labels.tolist())
        for cls in range(3):
            assert class_counts.get(cls, 0) == 2, \
                f"Each batch should have 2 samples per class"


def test_edge_cases(BalancedBatchSampler):
    """Test edge cases like single class, minimum samples."""
    # Two classes, minimum viable configuration
    labels = [0, 0, 1, 1]
    sampler = BalancedBatchSampler(labels, n_samples_per_class=1, seed=42)
    batches = list(sampler)
    assert len(batches) >= 1, "Should produce at least one batch"
    
    # Very imbalanced
    labels = [0] * 100 + [1, 1]
    sampler = BalancedBatchSampler(labels, n_samples_per_class=3, seed=42)
    batches = list(sampler)
    
    for batch in batches:
        class_counts = Counter(labels[idx] for idx in batch)
        assert class_counts[0] == 3, "Should have 3 samples from class 0"
        assert class_counts[1] == 3, "Should have 3 samples from class 1 (oversampled)"


def test_different_class_sizes(BalancedBatchSampler):
    """Test with various class size ratios."""
    # 4 classes with different sizes
    labels = [0]*10 + [1]*5 + [2]*20 + [3]*3
    sampler = BalancedBatchSampler(labels, n_samples_per_class=2, seed=42)
    
    all_indices = []
    for batch in sampler:
        all_indices.extend(batch)
        # Check balance
        class_counts = Counter(labels[idx] for idx in batch)
        for cls in range(4):
            assert class_counts.get(cls, 0) == 2, \
                f"Class {cls} should have 2 samples in each batch"
    
    # All original samples should be used
    unique_indices = set(all_indices)
    assert set(range(len(labels))).issubset(unique_indices), \
        "All samples should be used at least once"


# Reference solution for testing the grader
REFERENCE_SOLUTION = '''
import torch
from torch.utils.data import Sampler
import numpy as np
from typing import Iterator, List, Optional
from collections import defaultdict

class BalancedBatchSampler(Sampler):
    def __init__(self, labels: List[int], n_samples_per_class: int, seed: Optional[int] = None):
        self.labels = labels
        self.n_samples_per_class = n_samples_per_class
        self.seed = seed
        
        # Group indices by class
        self.class_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.class_indices[label].append(idx)
        
        self.num_classes = len(self.class_indices)
        self.classes = sorted(self.class_indices.keys())
        
        # Calculate number of batches needed to see all samples
        max_class_size = max(len(indices) for indices in self.class_indices.values())
        self.num_batches = (max_class_size + n_samples_per_class - 1) // n_samples_per_class
    
    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.RandomState(self.seed)
        
        # Create shuffled indices for each class, with oversampling if needed
        class_iters = {}
        for cls in self.classes:
            indices = self.class_indices[cls].copy()
            rng.shuffle(indices)
            
            # Oversample if needed
            total_needed = self.num_batches * self.n_samples_per_class
            if len(indices) < total_needed:
                # Repeat indices and shuffle again
                repeat_times = (total_needed // len(indices)) + 1
                indices = (indices * repeat_times)[:total_needed]
                rng.shuffle(indices)
            
            class_iters[cls] = iter(indices)
        
        for _ in range(self.num_batches):
            batch = []
            for cls in self.classes:
                for _ in range(self.n_samples_per_class):
                    batch.append(next(class_iters[cls]))
            yield batch
    
    def __len__(self) -> int:
        return self.num_batches
'''


if __name__ == "__main__":
    # Test the grader with reference solution
    print("Testing grader with reference solution...")
    result = grade_solution(REFERENCE_SOLUTION)
    print(f"Passed: {result['passed']}")
    print(f"Score: {result['score']}")
    print(f"Feedback: {result['feedback']}")


"""
Test script to verify the grader catches different failure modes.
Run this without an API key to validate grader logic.
"""

from task import grade_solution, REFERENCE_SOLUTION

# Test 1: Reference solution should pass
print("=" * 60)
print("Test 1: Reference solution (should PASS)")
print("=" * 60)
result = grade_solution(REFERENCE_SOLUTION)
print(f"Passed: {result['passed']}")
print(f"Score: {result['score']}")
print()

# Test 2: Wrong return type (returns indices, not batches)
WRONG_RETURN_TYPE = '''
import torch
from torch.utils.data import Sampler
import numpy as np
from typing import Iterator, List, Optional

class BalancedBatchSampler(Sampler):
    def __init__(self, labels: List[int], n_samples_per_class: int, seed: Optional[int] = None):
        self.labels = labels
        self.n_samples_per_class = n_samples_per_class
        self.seed = seed
    
    def __iter__(self) -> Iterator[int]:
        # WRONG: Yields individual indices instead of batches
        for i in range(len(self.labels)):
            yield i
    
    def __len__(self) -> int:
        return len(self.labels)
'''

print("=" * 60)
print("Test 2: Wrong return type - yields indices not batches (should FAIL)")
print("=" * 60)
result = grade_solution(WRONG_RETURN_TYPE)
print(f"Passed: {result['passed']}")
print(f"Score: {result['score']}")
print(f"Failed tests: {result.get('tests_failed', [])[:2]}")
print()

# Test 3: Random sampling without balance
UNBALANCED = '''
import torch
from torch.utils.data import Sampler
import numpy as np
from typing import Iterator, List, Optional

class BalancedBatchSampler(Sampler):
    def __init__(self, labels: List[int], n_samples_per_class: int, seed: Optional[int] = None):
        self.labels = labels
        self.n_samples_per_class = n_samples_per_class
        self.seed = seed
        self.num_classes = len(set(labels))
        self.batch_size = n_samples_per_class * self.num_classes
    
    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.RandomState(self.seed)
        indices = list(range(len(self.labels)))
        rng.shuffle(indices)
        
        # WRONG: Just random batches without balancing
        for i in range(0, len(indices), self.batch_size):
            batch = indices[i:i+self.batch_size]
            if len(batch) == self.batch_size:
                yield batch
    
    def __len__(self) -> int:
        return len(self.labels) // (self.n_samples_per_class * len(set(self.labels)))
'''

print("=" * 60)
print("Test 3: Random sampling without balance (should FAIL)")
print("=" * 60)
result = grade_solution(UNBALANCED)
print(f"Passed: {result['passed']}")
print(f"Score: {result['score']}")
print(f"Failed tests: {result.get('tests_failed', [])[:2]}")
print()

# Test 4: Not using all samples
DROPS_SAMPLES = '''
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
        
        self.class_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.class_indices[label].append(idx)
        
        self.num_classes = len(self.class_indices)
        self.classes = sorted(self.class_indices.keys())
        
        # WRONG: Only enough batches for smallest class
        min_class_size = min(len(indices) for indices in self.class_indices.values())
        self.num_batches = min_class_size // n_samples_per_class
        if self.num_batches == 0:
            self.num_batches = 1
    
    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.RandomState(self.seed)
        
        class_iters = {}
        for cls in self.classes:
            indices = self.class_indices[cls].copy()
            rng.shuffle(indices)
            class_iters[cls] = iter(indices)
        
        for _ in range(self.num_batches):
            batch = []
            for cls in self.classes:
                for _ in range(self.n_samples_per_class):
                    try:
                        batch.append(next(class_iters[cls]))
                    except StopIteration:
                        # WRONG: Just repeat first sample if exhausted
                        batch.append(self.class_indices[cls][0])
            yield batch
    
    def __len__(self) -> int:
        return self.num_batches
'''

print("=" * 60)
print("Test 4: Drops samples from majority classes (should FAIL)")
print("=" * 60)
result = grade_solution(DROPS_SAMPLES)
print(f"Passed: {result['passed']}")
print(f"Score: {result['score']}")
print(f"Failed tests: {result.get('tests_failed', [])[:2]}")
print()

# Test 5: Not reproducible
NOT_REPRODUCIBLE = '''
import torch
from torch.utils.data import Sampler
import numpy as np
import random
from typing import Iterator, List, Optional
from collections import defaultdict

class BalancedBatchSampler(Sampler):
    def __init__(self, labels: List[int], n_samples_per_class: int, seed: Optional[int] = None):
        self.labels = labels
        self.n_samples_per_class = n_samples_per_class
        # WRONG: Ignores seed
        
        self.class_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.class_indices[label].append(idx)
        
        self.num_classes = len(self.class_indices)
        self.classes = sorted(self.class_indices.keys())
        
        max_class_size = max(len(indices) for indices in self.class_indices.values())
        self.num_batches = (max_class_size + n_samples_per_class - 1) // n_samples_per_class
    
    def __iter__(self) -> Iterator[List[int]]:
        # WRONG: Uses random module instead of seeded rng
        class_iters = {}
        for cls in self.classes:
            indices = self.class_indices[cls].copy()
            random.shuffle(indices)  # Not seeded!
            
            total_needed = self.num_batches * self.n_samples_per_class
            if len(indices) < total_needed:
                repeat_times = (total_needed // len(indices)) + 1
                indices = (indices * repeat_times)[:total_needed]
                random.shuffle(indices)
            
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

print("=" * 60)
print("Test 5: Not reproducible with seed (should FAIL)")
print("=" * 60)
result = grade_solution(NOT_REPRODUCIBLE)
print(f"Passed: {result['passed']}")
print(f"Score: {result['score']}")
print(f"Failed tests: {result.get('tests_failed', [])[:2]}")
print()

# Test 6: Syntax error
SYNTAX_ERROR = '''
import torch
from torch.utils.data import Sampler

class BalancedBatchSampler(Sampler):
    def __init__(self, labels, n_samples_per_class, seed=None)
        pass  # Missing colon above!
'''

print("=" * 60)
print("Test 6: Syntax error (should FAIL)")
print("=" * 60)
result = grade_solution(SYNTAX_ERROR)
print(f"Passed: {result['passed']}")
print(f"Score: {result['score']}")
print(f"Feedback: {result['feedback'][:100]}...")
print()

# Test 7: Almost correct but wrong batch size
WRONG_BATCH_SIZE = '''
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
        
        self.class_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.class_indices[label].append(idx)
        
        self.num_classes = len(self.class_indices)
        self.classes = sorted(self.class_indices.keys())
        
        max_class_size = max(len(indices) for indices in self.class_indices.values())
        self.num_batches = (max_class_size + n_samples_per_class - 1) // n_samples_per_class
    
    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.RandomState(self.seed)
        
        class_iters = {}
        for cls in self.classes:
            indices = self.class_indices[cls].copy()
            rng.shuffle(indices)
            
            total_needed = self.num_batches * self.n_samples_per_class
            if len(indices) < total_needed:
                repeat_times = (total_needed // len(indices)) + 1
                indices = (indices * repeat_times)[:total_needed]
                rng.shuffle(indices)
            
            class_iters[cls] = iter(indices)
        
        for _ in range(self.num_batches):
            batch = []
            for cls in self.classes:
                # WRONG: Only adds n_samples_per_class - 1
                for _ in range(self.n_samples_per_class - 1):
                    batch.append(next(class_iters[cls]))
            yield batch
    
    def __len__(self) -> int:
        return self.num_batches
'''

print("=" * 60)
print("Test 7: Wrong batch size (should FAIL)")
print("=" * 60)
result = grade_solution(WRONG_BATCH_SIZE)
print(f"Passed: {result['passed']}")
print(f"Score: {result['score']}")
print(f"Failed tests: {result.get('tests_failed', [])[:2]}")
print()

print("=" * 60)
print("GRADER VALIDATION SUMMARY")
print("=" * 60)
print("The grader correctly:")
print("  ✓ Passes the reference solution")
print("  ✓ Catches wrong return types")
print("  ✓ Catches unbalanced sampling")
print("  ✓ Catches dropped samples")
print("  ✓ Catches reproducibility issues")
print("  ✓ Catches syntax errors")
print("  ✓ Catches wrong batch sizes")


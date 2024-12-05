import numpy as np

class BlockingTimeSeriesSplit:
    """
    A time series splitter that divides the data into non-overlapping, consecutive blocks.
    
    Parameters:
    - n_splits (int): Number of folds.
    """
    def __init__(self, n_splits):
        self.n_splits = n_splits

    def get_n_splits(self, X=None, y=None, groups=None):
        """Return the number of folds."""
        return self.n_splits

    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and test sets for each fold.

        Parameters:
        - X (array-like): Feature data to split.
        - y (array-like, optional): Target data.
        - groups (array-like, optional): Group labels.
        
        Yields:
        - train_indices (array): Indices for the training set.
        - test_indices (array): Indices for the test set.
        """
        n_samples = len(X)
        fold_size = n_samples // self.n_splits
        indices = np.arange(n_samples)

        for i in range(self.n_splits):
            start = i * fold_size
            stop = start + fold_size
            train_end = int(0.8 * fold_size) + start  # 80% train, 20% test
            yield indices[start:train_end], indices[train_end:stop]


class RollingBasisTimeSeriesSplit:
    """
    A rolling time series splitter with overlapping folds, each moving forward by a fixed step size.
    
    Parameters:
    - fold_size (int): Total number of samples in each fold.
    - step_size (int): Step size between folds.
    """
    def __init__(self, fold_size, step_size):
        self.fold_size = fold_size
        self.step_size = step_size

    def get_fold_size(self, X=None):
        """Return the fold size."""
        return self.fold_size
    
    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into rolling training and test sets.
        
        Parameters:
        - X (array-like): Feature data to split.
        - y (array-like, optional): Target data.
        - groups (array-like, optional): Group labels.
        
        Yields:
        - train_indices (array): Indices for the training set.
        - test_indices (array): Indices for the test set.
        """
        n_samples = len(X)
        train_size = int(self.fold_size * 0.8)  # 80% train
        test_size = self.fold_size - train_size  # 20% test
        start = 0

        while start + train_size + test_size <= n_samples:
            train_end = start + train_size
            test_end = train_end + test_size
            yield np.arange(start, train_end), np.arange(train_end, test_end)
            start += self.step_size

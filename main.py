import time
from multiprocessing import Pool, cpu_count

import numpy as np
from sklearn.datasets import make_blobs
from scipy.optimize import linear_sum_assignment

# --- Efficient Accuracy Calculation ---
def get_accuracy(true_labels, predicted_labels):
    """
    Match arbitrary K-Means cluster IDs to true labels using the Hungarian
    Algorithm to find the best mapping and return accuracy.
    """
    # Create confusion matrix
    d_size = max(predicted_labels.max(), true_labels.max()) + 1
    w = np.zeros((d_size, d_size), dtype=np.int64)

    # Fast histogram 2d calculation: map (pred, true) pairs to counts
    for p, t in zip(predicted_labels, true_labels):
        w[p, t] += 1

    # Find the best assignment (maximize diagonal sum)
    row_ind, col_ind = linear_sum_assignment(w.max() - w)

    # Calculate accuracy based on optimal assignment
    correct_matches = w[row_ind, col_ind].sum()
    return correct_matches / len(predicted_labels)

# --- Worker Function (Must be top-level) ---
def parallel_assign_chunk(args):
    """Assign points in a chunk to nearest centroids and return labels."""
    data_chunk, centroids = args
    # Calculation: (N x 1 x F) - (1 x K x F)
    distances = np.sqrt(((data_chunk[:, np.newaxis] - centroids) ** 2).sum(axis=2))
    return np.argmin(distances, axis=1)

class HighPerformanceKMeans:
    """Benchmark class for sequential vs. parallel K-Means assignment."""

    def __init__(self):
        # TUNED PARAMETERS FOR SPEEDUP
        # Applying "Hard" math to make parallel worth it.
        # 1 Million points x 50 Dimensions = Heavy CPU load
        self.n_samples = 1_000_000
        self.n_features = 50
        self.n_clusters = 10
        self.max_iter = 5  # Keep iterations low for quick demo

        print(
            f"1. Generating Data ({self.n_samples} samples, "
            f"{self.n_features} features)..."
        )
        # Generate data with ground truth
        self.data, self.true_labels = make_blobs(
            n_samples=self.n_samples,
            n_features=self.n_features,
            centers=self.n_clusters,
            random_state=42,
        )
        self.data = self.data.astype(np.float32)  # Optimize memory

        # Initialize centroids (deterministic start)
        np.random.seed(42)
        indices = np.random.choice(self.n_samples, self.n_clusters, replace=False)
        self.initial_centroids = self.data[indices].copy()

    def run_sequential(self):
        """Run K-Means assignment sequentially and return (time, accuracy)."""
        print("\n--- Starting Sequential Execution ---")
        start = time.time()
        centroids = self.initial_centroids.copy()
        labels = None

        for _ in range(self.max_iter):
            # Calculate all 1M distances on one core
            distances = np.sqrt(((self.data[:, np.newaxis] - centroids) ** 2).sum(axis=2))
            labels = np.argmin(distances, axis=1)

            # Update
            new_centroids = np.array(
                [self.data[labels == j].mean(axis=0) for j in range(self.n_clusters)]
            )
            centroids = new_centroids

        total_time = time.time() - start
        acc = get_accuracy(self.true_labels, labels)
        return total_time, acc

    def run_parallel(self):
        """Run K-Means assignment using multiprocessing and return (time, accuracy)."""
        n_workers = max(1, cpu_count() - 1)
        print(f"\n--- Starting Parallel Execution ({n_workers} Workers) ---")

        start = time.time()
        centroids = self.initial_centroids.copy()

        # Pre-slice data to avoid copying inside loop
        chunks = np.array_split(self.data, n_workers)

        with Pool(processes=n_workers) as pool:
            for _ in range(self.max_iter):
                # Map
                tasks = [(chunk, centroids) for chunk in chunks]
                results = pool.map(parallel_assign_chunk, tasks)

                # Gather
                labels = np.concatenate(results)

                # Reduce (Update)
                new_centroids = np.array(
                    [self.data[labels == j].mean(axis=0) for j in range(self.n_clusters)]
                )
                centroids = new_centroids

        total_time = time.time() - start
        acc = get_accuracy(self.true_labels, labels)
        return total_time, acc

if __name__ == "__main__":
    benchmark = HighPerformanceKMeans()

    # 1. Run Sequential
    seq_time, seq_acc = benchmark.run_sequential()
    print(f"Sequential: {seq_time:.4f}s | Accuracy: {seq_acc*100:.2f}%")

    # 2. Run Parallel
    par_time, par_acc = benchmark.run_parallel()
    print(f"Parallel:   {par_time:.4f}s | Accuracy: {par_acc*100:.2f}%")

    # 3. Final Report
    speedup = seq_time / par_time
    print("\n" + "=" * 50)
    print(f"{'METRIC':<20} | {'SEQUENTIAL':<12} | {'PARALLEL':<12}")
    print("-" * 50)
    print(f"{'Time (s)':<20} | {seq_time:<12.4f} | {par_time:<12.4f}")
    print(f"{'Accuracy (%)':<20} | {seq_acc*100:<12.2f} | {par_acc*100:<12.2f}")
    print("-" * 50)
    print(f"SPEEDUP FACTOR: {speedup:.2f}x")
    print("=" * 50)
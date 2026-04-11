"""
Step 2 of the PURSUhInT pipeline:
Loads stored layer representations (hint{i}.pt files) and clusters them using
K-Means with R2_CCA or CKA as the distance metric.
Cluster center indices are saved as the selected hint points.
"""

import numpy as np
import os
import argparse
import torch
from tqdm import tqdm

from clustering.utils import logger
from clustering.utils.cka import cka, gram_linear, gram_rbf, r2_cca


def build_argparser():
    parser = argparse.ArgumentParser(description="PURSUhInT K-Means Clustering of Teacher Layer Representations")
    parser.add_argument("--hints_dir", required=True, help="Directory containing hint{i}.pt files.")
    parser.add_argument("--num_clusters", default=3, type=int, help="Number of hint points (clusters) to find.")
    parser.add_argument("--num_layers", default=18, type=int, help="Total number of sub-blocks in the teacher model.")
    parser.add_argument("--metric_name", default='r2', choices=['r2', 'cka_linear', 'cka_rbf'],
                        help="Distance metric for K-Means clustering.")
    parser.add_argument("--output_dir", default='./save/hints', help="Directory to write centroid result files.")
    parser.add_argument("--model_name", default='teacher', help="Model name prefix for the output centroid filename.")
    return parser


# Distance metric wrappers (similarity -> distance)
def cka_linear(x1, x2):
    return 1.0 - cka(gram_linear(x1), gram_linear(x2))


def cka_rbf(x1, x2):
    return 1.0 - cka(gram_rbf(x1), gram_rbf(x2))


def r2(x1, x2):
    return 1.0 - r2_cca(x1, x2)


def get_metric_func(metric_name):
    if metric_name == 'r2':
        return r2
    elif metric_name == 'cka_linear':
        return cka_linear
    elif metric_name == 'cka_rbf':
        return cka_rbf
    else:
        raise NotImplementedError(f"Metric '{metric_name}' is not implemented.")


class KMeans:
    """
    K-Means clustering on teacher layer representations.
    Uses cluster *center* points (not last points) as per the PURSUhInT paper.
    Initial seeds are set to first, center, and last layer indices to mitigate random seeding issues.
    """

    def __init__(self, k=3, max_iter=80, distance_metric=None, num_all_layers=18):
        self.k = k
        self.max_iter = max_iter
        self.distance_metric = distance_metric
        self.num_all_layers = num_all_layers

    def _fit_once(self, data):
        for _ in tqdm(range(self.max_iter), desc="K-Means iterations"):
            self.classifications = {j: [] for j in range(self.k)}
            self.indices_h = {j: [] for j in range(self.k)}

            # Assignment step: assign each layer to nearest centroid
            for index_h in range(self.num_all_layers):
                data_h = data[index_h]
                distances = [self.distance_metric(data_h, data[self.centroids[c]]) for c in self.centroids]
                classification = int(np.argmin(distances))
                self.classifications[classification].append(data_h)
                self.indices_h[classification].append(index_h)

            prev_centroids = dict(self.centroids)

            # Update step: move centroid to the layer closest to the mean index
            for c in range(self.k):
                members = self.indices_h[c]
                if len(members) == 0:
                    continue
                mean_idx = int(np.mean(members))
                new_center = min(members, key=lambda x: abs(x - mean_idx))
                self.centroids[c] = new_center
                logger.info(f"Cluster {c} center -> layer {new_center}")

            print(20 * '--')
            if self.centroids == prev_centroids:
                break  # Converged

    def fit(self, data):
        # Initialize centroids: first, center, and last layer (paper's seeding strategy)
        self.centroids = {}
        for t in range(self.k):
            self.centroids[t] = int(t * (self.num_all_layers - 1) / (self.k - 1))
        logger.info(f"Initial centroids: {self.centroids}")
        self._fit_once(data)
        return self

    def get_centroid_indices(self):
        """Returns sorted centroid layer indices (1-indexed for compatibility with train_student.py --hint_points)."""
        raw = sorted(self.centroids.values())
        # Convert to 1-indexed
        return [idx + 1 for idx in raw]


def main():
    parser = build_argparser()
    opt = parser.parse_args()

    # Load all layer representations
    X = []
    for i in range(1, opt.num_layers + 1):
        hint_path = os.path.join(opt.hints_dir, f'hint{i}.pt')
        if not os.path.isfile(hint_path):
            raise FileNotFoundError(f"Missing hint file: {hint_path}. Run store_hints.py first.")
        x = torch.load(hint_path, map_location='cpu', weights_only=False)
        if not isinstance(x, np.ndarray):
            x = x.cpu().detach().numpy()
        X.append(x)
    print(f"Loaded {len(X)} layer representations from {opt.hints_dir}")

    metric = get_metric_func(opt.metric_name)

    clf = KMeans(
        k=opt.num_clusters,
        distance_metric=metric,
        num_all_layers=opt.num_layers,
    )
    clf.fit(X)
    centroid_indices = clf.get_centroid_indices()

    print(f"\n{'='*50}")
    print(f"PURSUhInT selected hint points: {centroid_indices}")
    print(f"{'='*50}\n")

    # Save results
    os.makedirs(opt.output_dir, exist_ok=True)
    base_name = f"{opt.model_name}_{opt.num_clusters}clusters_{opt.metric_name}"

    # Save full cluster membership info
    membership_file = os.path.join(opt.output_dir, base_name + '_clusters.txt')
    with open(membership_file, 'w') as f:
        f.write(str(clf.indices_h))
    print(f"Cluster membership saved to: {membership_file}")

    # Save centroid indices in comma-separated format (for --hint_points)
    centroid_file = os.path.join(opt.output_dir, base_name + '_centroids.txt')
    with open(centroid_file, 'w') as f:
        f.write(','.join(str(c) for c in centroid_indices))
    print(f"Centroid hint points saved to: {centroid_file}")
    print(f"Use these as: --hint_points {','.join(str(c) for c in centroid_indices)}")


if __name__ == '__main__':
    main()

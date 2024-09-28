import random
import math
import numpy as np
from sklearn.datasets import make_blobs

class KMeansClustering:
    def __init__(self, n_clusters, init_method='random'):
        self.n_clusters = n_clusters
        self.init_method = init_method
        self.data, _ = make_blobs(n_samples=300, centers=n_clusters, random_state=42)
        self.centroids = np.empty((n_clusters, self.data.shape[1]))  # Initialize centroids as NumPy array
        self.labels = np.empty(self.data.shape[0], dtype=int)  # Initialize labels as NumPy array
        self.initialize_centroids()

    # Helper function to calculate Euclidean distance
    def euclidean_distance(self, point1, point2):
        return np.linalg.norm(point1 - point2)

    # Helper function to compute the mean of points in a cluster
    def calculate_mean(self, points):
        return np.mean(points, axis=0)

    # KMeans++ initialization
    def kmeans_plus_plus_init(self):
        centroids = []
        centroids.append(random.choice(self.data))  # Randomly pick the first centroid

        for _ in range(1, self.n_clusters):
            distances = []
            for point in self.data:
                # Compute distance to the closest centroid
                min_dist = min(self.euclidean_distance(point, centroid) for centroid in centroids)
                distances.append(min_dist ** 2)

            total_distance = sum(distances)
            probabilities = [dist / total_distance for dist in distances]
            cumulative_probabilities = np.cumsum(probabilities)
            r = random.random()

            for i, cumulative_prob in enumerate(cumulative_probabilities):
                if r <= cumulative_prob:
                    centroids.append(self.data[i])
                    break

        self.centroids = np.array(centroids)

    # Random initialization
    def random_init(self):
        self.centroids = np.array(random.sample(self.data.tolist(), self.n_clusters))

    # Initialize centroids based on the chosen method
    def initialize_centroids(self):
        if self.init_method == 'k-means++':
            self.kmeans_plus_plus_init()
        elif self.init_method == 'random':
            self.random_init()
        else:
          raise NotImplementedError(f"{self.init_method} initialization not implemented")

    # Assign each point to the nearest centroid
    def assign_clusters(self):
        clusters = [[] for _ in range(self.n_clusters)]
        self.labels = np.empty(self.data.shape[0], dtype=int)

        for i, point in enumerate(self.data):
            distances = [self.euclidean_distance(point, centroid) for centroid in self.centroids]
            closest_centroid_idx = distances.index(min(distances))
            clusters[closest_centroid_idx].append(point)
            self.labels[i] = closest_centroid_idx

        return clusters

    # Update centroids by calculating the mean of each cluster
    def update_centroids(self, clusters):
        self.centroids = np.array([self.calculate_mean(cluster) if len(cluster) > 0 else random.choice(self.data) for cluster in clusters])

    # Step through KMeans (one iteration)
    def step(self):
        clusters = self.assign_clusters()  # Assign points to clusters based on current centroids
        old_centroids = self.centroids.copy()
        self.update_centroids(clusters)  # Update centroids based on new cluster assignments

    # Run the KMeans algorithm until convergence
    def run_to_convergence(self, max_iter=300):
        for _ in range(max_iter):
            clusters = self.assign_clusters()  # Assign points to clusters
            old_centroids = self.centroids.copy()
            self.update_centroids(clusters)  # Update centroids

            # If centroids haven't changed, the algorithm has converged
            if np.array_equal(old_centroids, self.centroids):
                break

    # Generate new data (useful for re-running the algorithm on new data)
    def generate_new_data(self):
        self.data, _ = make_blobs(n_samples=300, centers=self.n_clusters, random_state=42)
        self.initialize_centroids()

# Example usage:
if __name__ == "__main__":
    # Initialize with KMeans++ method
    kmeans_clustering = KMeansClustering(n_clusters=3, init_method='kmeans++')
    
    # Perform iterative steps of KMeans
    for i in range(10):  # Example: take 10 steps
        print(f"Step {i + 1}")
        kmeans_clustering.step()
    
    # Run KMeans to convergence
    kmeans_clustering.run_to_convergence()
    print("Final centroids:", kmeans_clustering.centroids)

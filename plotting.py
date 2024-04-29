import numpy as np
import matplotlib.pyplot as plt

n = input('Number of Realizations: ')

def read_points(filename):
    points = np.loadtxt(filename)
    return points[:, 0], points[:, 1]

def read_density(filename):
    density_matrix = np.loadtxt(filename)
    return density_matrix

def plot_density_and_points(density_filename, point_cloud_filename):
    # Read density matrix
    density_matrix = read_density(density_filename)

    # Read point cloud
    x, y = read_points(point_cloud_filename)

    # Plot density
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(density_matrix.T, origin='lower', extent=[0, 1, 0, 1], cmap='viridis')
    plt.colorbar(label='Density')
    plt.title('Expectation')

    # Plot point cloud
    plt.subplot(1, 2, 2)
    plt.scatter(x, y, s=10)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Original Point Pattern')

    plt.tight_layout()
    plt.show()
    
def plot_points(point_cloud_filename):

    # Read point cloud
    x, y = read_points(point_cloud_filename)

    # Plot density
    plt.figure(figsize=(5, 5))

    plt.scatter(x, y, s=10)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Sampled Point Pattern')

    plt.tight_layout()
    plt.show()

# Example usage:
plot_density_and_points("density_matrix.txt", "original_point_cloud.txt")

for i in range(1, int(n)+1):
    point_cloud_filename = f"sampled_point_cloud_{i}.txt"
    plot_points(point_cloud_filename)



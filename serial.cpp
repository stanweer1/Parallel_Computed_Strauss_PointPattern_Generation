#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <fstream>

struct Point {
    double x;
    double y;
};

// Function to calculate Euclidean distance between two points
double distance(const Point& p1, const Point& p2) {
    return std::sqrt(std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y, 2));
}

// Function to generate a random point within a circle of radius `r` around a given point
Point generateRandomPointAround(const Point& center, double radius) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 2 * M_PI); // Angle in radians
    double angle = dis(gen);
    double r = radius * std::sqrt(dis(gen)); // Square root to ensure uniform distribution within the circle
    double newX = center.x + r * std::cos(angle);
    double newY = center.y + r * std::sin(angle);
    return {newX, newY};
}

// Function to ensure that a point is within the specified range
void clampPointToBounds(Point& point, double minX, double maxX, double minY, double maxY) {
    point.x = std::max(minX, std::min(maxX, point.x));
    point.y = std::max(minY, std::min(maxY, point.y));
}

// Function to calculate the average nearest neighbor distance
double calculateAverageNearestNeighborDistance(const std::vector<Point>& pointCloud) {
    double avgNearestNeighborDistance = 0.0;
    for (size_t i = 0; i < pointCloud.size(); ++i) {
        double minDistance = std::numeric_limits<double>::max();
        for (size_t j = 0; j < pointCloud.size(); ++j) {
            if (i != j) {
                double dist = distance(pointCloud[i], pointCloud[j]);
                if (dist < minDistance)
                    minDistance = dist;
            }
        }
        avgNearestNeighborDistance += minDistance;
    }
    avgNearestNeighborDistance /= pointCloud.size();
    return avgNearestNeighborDistance;
}

std::vector<Point> generateRandomPointCloud(int numPoints, double minX, double maxX, double minY, double maxY) {
    std::vector<Point> pointCloud;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> disX(minX, maxX);
    std::uniform_real_distribution<> disY(minY, maxY);
    for (int i = 0; i < numPoints; ++i) {
        pointCloud.push_back({disX(gen), disY(gen)});
    }
    return pointCloud;
}

// Function to generate a Strauss Point Process on a given point cloud
std::vector<Point> generateStraussPointProcess(const std::vector<Point>& pointCloud, double radius, double lambda, double minX, double maxX, double minY, double maxY) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    std::vector<Point> sampledPoints = pointCloud;

    // Burn-in period
    for (int burnIn = 0; burnIn < 10; ++burnIn) {
        for (auto& point : sampledPoints) {
            // Generate a new proposed point around the current point
            std::uniform_real_distribution<> disAngle(0.0, 2 * M_PI); // Angle in radians
            double angle = disAngle(gen);
            double r = radius * std::sqrt(dis(gen)); // Square root to ensure uniform distribution within the circle
            double newX = point.x + r * std::cos(angle);
            double newY = point.y + r * std::sin(angle);

            // Ensure the proposed point is within the specified range
            newX = std::max(minX, std::min(maxX, newX));
            newY = std::max(minY, std::min(maxY, newY));

            // Compute the acceptance probability
            int count = 0;
            for (const auto& otherPoint : sampledPoints) {
                double dist = distance({newX, newY}, otherPoint);
                if (dist <= radius && dist > 0)
                    count++;
            }
            double p_accept = lambda * std::exp(-count); // Using exponential decay for the probability

            // Accept or reject the proposed point
            if (dis(gen) < p_accept)
                point = {newX, newY}; // Replace the current point with the proposed point
        }
    }

    return sampledPoints;
}

// Function to write points to a file
void writePointsToFile(const std::vector<Point>& points, const std::string& filename) {
    std::ofstream outFile(filename);
    if (outFile.is_open()) {
        for (const auto& point : points)
            outFile << point.x << " " << point.y << std::endl;
        outFile.close();
    } else {
        std::cerr << "Unable to open file: " << filename << std::endl;
    }
}

// Function to calculate the probability density at a given point for the Strauss Point Process model
double calculateDensityAtPoint(const Point& point, const std::vector<Point>& pointCloud, double radius, double lambda) {
    int count = 0;
    for (const auto& otherPoint : pointCloud) {
        double dist = distance(point, otherPoint);
        if (dist <= radius && dist > 0) {
            count++;
        }
    }
    // The density decreases with increasing number of nearby points
    double density = lambda * std::exp(-count);
    return density;
}

// Function to generate the probability density matrix for the Strauss Point Process model
std::vector<std::vector<double>> generateDensityMatrix(const std::vector<Point>& pointCloud, double radius, double lambda, int resolution) {
    std::vector<std::vector<double>> densityMatrix(resolution, std::vector<double>(resolution, 0.0));

    // Calculate the step size for the grid
    double stepSize = 1.0 / resolution;

    // Iterate over each grid point and compute the density
    for (int i = 0; i < resolution; ++i) {
        for (int j = 0; j < resolution; ++j) {
            Point gridPoint = {i * stepSize, j * stepSize};
            densityMatrix[i][j] = calculateDensityAtPoint(gridPoint, pointCloud, radius, lambda);
        }
    }

    return densityMatrix;
}

// Function to write the density matrix to a file
void writeDensityMatrixToFile(const std::vector<std::vector<double>>& densityMatrix, const std::string& filename) {
    std::ofstream outFile(filename);
    if (outFile.is_open()) {
        for (const auto& row : densityMatrix) {
            for (const auto& density : row) {
                outFile << density << " ";
            }
            outFile << "\n";
        }
        outFile.close();
    } else {
        std::cerr << "Unable to open file: " << filename << std::endl;
    }
}

// Function to check if the sampled point pattern follows the Strauss Point Process model
bool validateStraussPointProcess(const std::vector<Point>& sampledPoints, double radius, double lambda) {
    for (size_t i = 0; i < sampledPoints.size(); ++i) {
        int count = 0;
        for (size_t j = 0; j < sampledPoints.size(); ++j) {
            if (i != j) {
                double dist = distance(sampledPoints[i], sampledPoints[j]);
                if (dist <= radius)
                    count++;
            }
        }
        double p_accept = lambda *  std::exp(-count);
        if (p_accept <= 0.5)
            return false; // The sampled point pattern violates the Strauss condition
    }
    return true;
}

int main(int argc, char** argv) {
    // Take sizes from the command line of numpoints and numpatterns and required threads
    if (argc != 3) {
        return 1;
    }

    int numPoints = atoi(argv[1]), numPatterns = atoi(argv[2]);
    double start_time, end_time; 

    // Defining my space to be a [0, 1] square
    double minX = 0.0, maxX = 1.0, minY = 0.0, maxY = 1.0;
    int resolution = 100; // Change the resolution as needed

    // Generate random point cloud
    std::vector<Point> pointCloud = generateRandomPointCloud(numPoints, minX, maxX, minY, maxY);
    
    std::string sampledFilename = "original_point_cloud.txt";
    writePointsToFile(pointCloud, sampledFilename);

    // Calculate average nearest neighbor distance
    double avgNearestNeighborDistance = calculateAverageNearestNeighborDistance(pointCloud);
    std::cout << "Average Nearest Neighbor Distance: " << avgNearestNeighborDistance << std::endl;

    // Estimate parameters for Strauss Point Process
    double radius = avgNearestNeighborDistance / 2.0; // Set radius as half of the average nearest neighbor distance
    double lambda = 1.0 / (M_PI * std::pow(radius, 2)); // Set lambda based on the area of the interaction circle


    std::vector<std::vector<double>> densityMatrix = generateDensityMatrix(pointCloud, radius, lambda, resolution);

    // Write the density matrix to a file
    std::string densityFilename = "density_matrix.txt";
    writeDensityMatrixToFile(densityMatrix, densityFilename);

    // Generate multiple Strauss Point Process patterns
    for (int i = 0; i < numPatterns; ++i) {
        // Generate Strauss Point Process
        std::vector<Point> sampledPointCloud = generateStraussPointProcess(pointCloud, radius, lambda, minX, maxX, minY, maxY);

        // Write sampled point cloud to file
        std::string sampledFilename = "sampled_point_cloud_" + std::to_string(i + 1) + ".txt";
        writePointsToFile(sampledPointCloud, sampledFilename);

        // Validate the sampled point pattern
        bool isValid = validateStraussPointProcess(sampledPointCloud, radius, lambda);

        std::cout << "Sampled Point Cloud " << i + 1 << ":\n";
        if (isValid) {
            for (const auto& point : sampledPointCloud)
                std::cout << "(" << point.x << ", " << point.y << ")\n";
            std::cout << "\nSampled point pattern follows the Strauss Point Process model.\n";
        } else {
            std::cout << "The sampled point pattern does not follow the Strauss Point Process model.\n";
        }
    }

    return 0;
}



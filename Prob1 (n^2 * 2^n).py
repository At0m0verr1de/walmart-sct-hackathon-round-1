import numpy as np
import csv
from functools import lru_cache
import time

start_time = time.time()

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # radius of Earth in kilometers
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda/2)**2
    res = R * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))
    return np.round(res, 2)

def create_distance_matrix(file_path):
    locations = []
    with open(file_path, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            locations.append((float(row['lat']), float(row['lng'])))
    num_locations = len(locations)
    distance_matrix = np.zeros((num_locations, num_locations))
    for i in range(num_locations):
        for j in range(num_locations):
            if i != j:
                distance_matrix[i][j] = haversine(locations[i][0], locations[i][1], locations[j][0], locations[j][1])
    return distance_matrix

def held_karp(dists):
    n = len(dists)
    @lru_cache(maxsize=None)
    def visit(subset, end):
        if subset == (1 << end):
            return (dists[0][end], [0, end])
        prev_subset = subset & ~(1 << end)
        results = []
        for prev_city in range(n):
            if subset & (1 << prev_city):
                prev_cost, prev_path = visit(prev_subset, prev_city)
                results.append((prev_cost + dists[prev_city][end], prev_path + [end]))
        return min(results)
    all_cities = (1 << n) - 1
    min_cost, path = min(
        (visit(all_cities, end)[0] + dists[end][0], visit(all_cities, end)[1])
        for end in range(1, n)
    )
    return min_cost, path

# Example usage
distance_matrix = create_distance_matrix('part_a_input_dataset_1.csv')
min_cost, path = held_karp(distance_matrix)
print("Minimum tour cost:", min_cost)
print("Path:", path)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time taken: {elapsed_time} seconds")

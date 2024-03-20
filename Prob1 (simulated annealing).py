import numpy as np
import random
import math
import csv
import time
from functools import lru_cache
start_time = time.time()


def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # radius of Earth in kilometers
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    res = R * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))
    return np.round(res, 2)


def create_distance_matrix(file_path):
    # Reading locations using NumPy (assuming lat, lng order in the file)
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1, usecols=(1, 2))
    num_locations = len(data)
    lat, lng = data[:, 1], data[:, 0]

    # Calculate distance matrix using broadcasting and vectorized operations
    lat = np.radians(lat)
    lng = np.radians(lng)
    delta_lat = lat[:, np.newaxis] - lat
    delta_lng = lng[:, np.newaxis] - lng
    a = np.sin(delta_lat / 2) ** 2 + np.cos(lat)[:, np.newaxis] * np.cos(lat) * np.sin(delta_lng / 2) ** 2
    distance_matrix = 6371 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Round the distances to two decimals
    distance_matrix = np.round(distance_matrix, 2)

    # No need to zero out the diagonal; it should already be close to zero from calculations
    return distance_matrix


def calculate_total_distance(route, distance_matrix):
    distances = distance_matrix[route[:-1], route[1:]]
    return np.sum(distances) + distance_matrix[route[-1], route[0]]  # Closing the loop


def swap_two_cities(tour):
    a, b = random.sample(range(len(tour)), 2)
    tour[a], tour[b] = tour[b], tour[a]
    return tour


def simulated_annealing(distance_matrix, initial_temperature=100, cooling_rate=0.99, final_temperature=1):
    current_tour = np.arange(len(distance_matrix))
    np.random.shuffle(current_tour)
    current_distance = calculate_total_distance(current_tour, distance_matrix)

    temperature = initial_temperature
    while temperature > final_temperature:
        candidate_tour = current_tour.copy()
        swap_two_cities(candidate_tour)
        candidate_distance = calculate_total_distance(candidate_tour, distance_matrix)

        delta_distance = candidate_distance - current_distance

        if delta_distance < 0 or random.random() < math.exp(-delta_distance / temperature):
            current_tour = candidate_tour
            current_distance = candidate_distance

        temperature *= cooling_rate

    return current_tour, current_distance


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


distance_matrix = create_distance_matrix('part_a_input_dataset_3.csv')

if math.sqrt(distance_matrix.size) < 14:
    tour, distance = held_karp(distance_matrix)
else:
    distance = float('inf')  # Initialize with an infinitely large value
    tour = None
    tour, distance = simulated_annealing(distance_matrix)
    for _ in range(100):
        tempTour, tempDistance = simulated_annealing(distance_matrix)
        if distance < distance:
            distance = tempDistance
            tour = tempTour


# tour_order_ids = [order_ids[i] for i in best_tour]
# print("order_id lng lat depot_lat depot_lng dlvr_seq_num")
# for seq_num, order_id in enumerate(optimal_path_order_ids, start=1):
#     order = orders_data[order_id]
#     print(f"{order_id} {order['lng']} {order['lat']} {order['depot_lat']} {order['depot_lng']} {seq_num}")


print("Best tour:", tour)
print("Total distance:", distance)

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Time taken to create the distance matrix: {elapsed_time} seconds")
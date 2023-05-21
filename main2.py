import random
import numpy as np
import time
from python_tsp.exact import solve_tsp_dynamic_programming, solve_tsp_brute_force
from tsp_solver.greedy import solve_tsp
from pyCombinatorial.algorithm import genetic_algorithm
from pyCombinatorial.algorithm import local_search_2_opt
from pyCombinatorial.algorithm import branch_and_bound
from pyCombinatorial.utils import util


def create_coordinates(num_points, max_x, max_y):
    """
    Creates a list of coordinates, between specified X and Y constraints.
    :param num_points: The number of points to create.
    :param max_x: Max X bound.
    :param max_y: Max Y bound.
    :return:
    """
    points = []
    for _ in range(num_points):
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)
        points.append([x, y])
    return points


def get_distance(path, points):
    """
    Calculates the distance of a path.
    :param path: An array of indices for points.
    :param points: A list of coordinate arrays.
    :return: The Manhattan distance of the path.
    """
    distance = 0
    for p1, p2 in zip(path, path[1:] + path[:1]):
        distance += abs(points[p1][0] - points[p2][0]) + abs(
            points[p1][1] - points[p2][1]
        )
    return distance


def get_distance_matrix(points):
    """
    Creates a distance matrix using a list of coordinates.
    :param points: The coordinates to formulate a matrix.
    :return: A distance matrix.
    """
    n = len(points)
    distance_matrix = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            distance_matrix[i][j] = abs(points[i][0] - points[j][0]) + abs(
                points[i][1] - points[j][1]
            )
    return np.array(distance_matrix)


def solve_tsp_hk(distance_matrix, points):
    """
    Solves the TSP for a distance matrix using the Held Karp algorithm.
    :param distance_matrix: The distance matrix to solve.
    :param points: The list of coordinates passed.
    :return: The distance of the optimised solution, and the time taken to solve.
    """
    print("HK: ")

    start = time.time()  # A measure of the current time before solver execution
    path, distance = solve_tsp_dynamic_programming(distance_matrix)
    print(path)
    end = time.time()  # A measure of the current time after solver execution
    elapsed_time = (
            end - start
    )  # The difference in times can be calculated to find the running time

    distance = get_distance(path, points)

    print(distance, elapsed_time)
    print("---------------------")
    return distance, elapsed_time


def solve_tsp_genetic(distance_matrix):
    """
    Solves the TSP for a distance matrix using the Held Karp algorithm.
    :param distance_matrix: The distance matrix to solve.
    :return: The distance of the optimised solution, and the time taken to solve.
    """
    print("Genetic:")

    start = time.time()
    path, distance = genetic_algorithm(distance_matrix, verbose=False)
    end = time.time()
    elapsed_time = end - start

    print(path)
    print(distance, elapsed_time)
    print("---------------------")
    return distance, elapsed_time


def solve_tsp_2opt(distance_matrix):
    """
    Solves the TSP for a distance matrix using the 2opt algorithm.
    :param distance_matrix: The distance matrix to solve.
    :return: The distance of the optimised solution, and the time taken to solve.
    """
    print("2opt: ")
    start = time.time()
    path, distance = local_search_2_opt(
        distance_matrix, util.seed_function(distance_matrix)
    )
    print(path)
    end = time.time()
    elapsed_time = end - start

    print(distance, elapsed_time)
    print("---------------------")
    return distance, elapsed_time


def solve_tsp_bnb(distance_matrix):
    """
    Solves the TSP for a distance matrix using the Branch and Bound algorithm.
    :param distance_matrix: The distance matrix to solve.
    :return: The distance of the optimised solution, and the time taken to solve.
    """
    print("BNB: ")
    start = time.time()
    path, distance = branch_and_bound(distance_matrix)
    end = time.time()
    elapsed_time = end - start
    print(path)
    print(distance, elapsed_time)
    print("---------------------")
    return distance, elapsed_time


def solve_tsp_bruteforce(distance_matrix, points):
    """
    Solves the TSP for a distance matrix using the Bruteforce algorithm.
    :param distance_matrix: The distance matrix to solve.
    :param points: The list of coordinates passed.
    :return: The distance of the optimised solution, and the time taken to solve.
    """
    print("Bruteforce:")
    start = time.time()
    path, distance = solve_tsp_brute_force(distance_matrix)
    end = time.time()
    elapsed_time = end - start

    distance = get_distance(path, points)
    print(path)
    print(distance, elapsed_time)
    print("---------------------")
    return distance, elapsed_time


def solve_tsp_greedy(distance_matrix, points):
    """
    Solves the TSP for a distance matrix using the Greedy algorithm.
    :param distance_matrix: The distance matrix to solve.
    :param points: The list of coordinates passed.
    :return: The distance of the optimised solution, and the time taken to solve.
    """
    print("Greedy:")
    start = time.time()
    path = solve_tsp(distance_matrix)
    end = time.time()
    elapsed_time = end - start

    print(path)
    distance = get_distance(path, points)
    print(distance, elapsed_time)
    print("---------------------")
    return distance, elapsed_time


def solve_tsps(sample_size, bound_x, bound_y, iterations):
    """
    Calls solver functions to compare solutions of the TSP using a range of different algorithms.
    :param sample_size: N size to use in solutions, number of cities.
    :param bound_x: Upper bound of the X axis.
    :param bound_y: Upper bound of the Y axis.
    :param iterations: Number of iterations to use for each algorithm.
    :return: Prints values of mean solution distance and mean time taken to console for each algorithm.
    """
    greedy_distance = 0.0
    greedy_time = 0.0
    hk_distance = 0.0
    hk_time = 0.0
    _2opt_distance = 0.0
    _2opt_time = 0.0
    bnb_distance = 0.0
    bnb_time = 0.0
    brute_distance = 0.0
    brute_time = 0.0
    genetic_distance = 0.0
    genetic_time = 0.0

    for i in range(
            iterations
    ):
        # Creates coordinates based on user-input and then iterates through the algorithms for a specified number of
        # iterations.
        points = create_coordinates(sample_size, bound_x, bound_y)
        distance_matrix = get_distance_matrix(points)

        d, t = solve_tsp_greedy(distance_matrix, points)
        greedy_distance += d
        greedy_time += t

        d, t = solve_tsp_hk(distance_matrix, points)
        hk_distance += d
        hk_time += t

        d, t = solve_tsp_2opt(distance_matrix)
        _2opt_distance += d
        _2opt_time += t

        d, t = solve_tsp_bnb(distance_matrix)
        bnb_distance += d
        bnb_time += t

        d, t = solve_tsp_bruteforce(distance_matrix, points)
        brute_distance += d
        brute_time += t

        d, t = solve_tsp_genetic(distance_matrix)
        genetic_distance += d
        genetic_time += t

    # Prints the mean distance and time.
    print("----------------------------------------------")
    print("Greedy:")
    print(greedy_distance / iterations, greedy_time / iterations)
    print("HK:")
    print(hk_distance / iterations, hk_time / iterations)
    print("2opt:")
    print(_2opt_distance / iterations, _2opt_time / iterations)
    print("BNB:")
    print(bnb_distance/iterations, bnb_time/iterations)
    print("Brute Force:")
    print(brute_distance/iterations, brute_time/iterations)
    print("Genetic:")
    print(genetic_distance / iterations, genetic_time / iterations)


# Handles user input, allowing the user to benchmark with different values.
sample_size = int(input("Enter the number of points: "))
bound_x = int(input("Enter the maximum value for X: "))
bound_y = int(input("Enter the maximum value for Y: "))
iterations = int(input("Enter the number of iterations: "))

# Calls the solver function.
solve_tsps(sample_size, bound_x, bound_y, iterations)

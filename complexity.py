import matplotlib.pyplot as plt
import numpy as np



"""
Understanding Big O Notation and Algorithm Complexity in Python
This file includes:
1. Code examples for common Big O complexities.
2. Iteration counting for validation.
3. Visualization of complexities.
"""

# O(1) Example

def get_first_element(lst):
    """
    Access the first element of a list (constant time operation).
    """
    return lst[0]

# O(log n) Example

def binary_search(arr, target):
    """
    Perform binary search on a sorted array.
    Iteration count demonstrates logarithmic complexity.
    """
    left, right = 0, len(arr) - 1
    iterations = 0
    while left <= right:
        iterations += 1
        mid = (left + right) // 2
        if arr[mid] == target:
            print(f"Iterations: {iterations}")
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    print(f"Iterations: {iterations}")
    return -1

# O(n) Example

def find_max(lst):
    """
    Find the maximum value in a list.
    Iteration count demonstrates linear complexity.
    """
    max_val = lst[0]
    iterations = 0
    for num in lst:
        iterations += 1
        if num > max_val:
            max_val = num
    print(f"Iterations: {iterations}")
    return max_val

#O(n log n) - Quasilinear Time Common in efficient sorting algorithms like mergesort. 
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    return merge(left, right)

def merge(left, right):
    sorted_arr = []
    iterations = 0
    while left and right:
        iterations += 1
        if left[0] < right[0]:
            sorted_arr.append(left.pop(0))
        else:
            sorted_arr.append(right.pop(0))
    sorted_arr.extend(left or right)
    print(f"Iterations in merge: {iterations}")
    return sorted_arr

#O(n^2) - Quadratic Time Performance grows quadratically with the size of the input, often seen in nested loops. 

def bubble_sort(arr):
    n = len(arr)
    iterations = 0
    for i in range(n):
        for j in range(0, n-i-1):
            iterations += 1
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    print(f"Iterations: {iterations}")

#O(2^n) - Exponential Time Performance doubles with each additional input, common in recursive algorithms without memoization. 

from itertools import permutations

def traveling_salesman(cities):
    all_routes = permutations(cities)
    shortest_distance = float('inf')
    iterations = 0
    for route in all_routes:
        iterations += 1
        distance = calculate_distance(route)
        if distance < shortest_distance:
            shortest_distance = distance
    print(f"Iterations: {iterations}")
    return shortest_distance

#O(n!) - Factorial Time Seen in brute-force algorithms for solving problems like the traveling salesman problem. 

from itertools import permutations

def calculate_distance(route):
    # Assuming you have a distance matrix or a method to calculate the distance between two cities
    # For simplicity, we'll assume that a simple distance function is used (e.g., Euclidean distance)
    total_distance = 0
    for i in range(len(route) - 1):
        city1 = route[i]
        city2 = route[i + 1]
        total_distance += get_distance(city1, city2)
    # To complete the cycle, return to the starting city
    total_distance += get_distance(route[-1], route[0])
    return total_distance

def get_distance(city1, city2):
    # This function should return the distance between two cities
    # For now, we are assuming a dummy distance, you should replace it with your actual distance calculation
    # E.g., Euclidean distance, or from a distance matrix
    return abs(city1 - city2)  # Replace with actual distance logic

def traveling_salesman(cities):
    all_routes = permutations(cities)
    shortest_distance = float('inf')
    iterations = 0
    for route in all_routes:
        iterations += 1
        distance = calculate_distance(route)
        if distance < shortest_distance:
            shortest_distance = distance
    print(f"Iterations: {iterations}")
    return shortest_distance


# Example input data
lst = [3, 5, 1, 2, 4]
arr = [1, 3, 5, 7, 9, 11]
target = 5
cities = [1, 2, 3]

# O(1) Example - get_first_element
first_element = get_first_element(lst)
print(f"First element: {first_element}")

# O(log n) Example - binary_search
index = binary_search(arr, target)
print(f"Target index: {index}")

# O(n) Example - find_max
max_value = find_max(lst)
print(f"Max value: {max_value}")

# O(n log n) Example - merge_sort
sorted_arr = merge_sort(lst)
print(f"Sorted array: {sorted_arr}")

# O(n^2) Example - bubble_sort
bubble_sort(lst)

# O(2^n) Example - traveling_salesman
shortest_distance = traveling_salesman(cities)
print(f"Shortest distance: {shortest_distance}")


#Visualizing Big O Complexity


n = np.linspace(1, 10, 100)

plt.plot(n, np.ones_like(n), label="O(1)")
plt.plot(n, np.log2(n), label="O(log n)")
plt.plot(n, n, label="O(n)")
plt.plot(n, n * np.log2(n), label="O(n log n)")
plt.plot(n, n**2, label="O(n^2)")
plt.plot(n, 2**n, label="O(2^n)")

plt.ylim(0, 100)
plt.xlabel("Input size (n)")
plt.ylabel("Operations")
plt.title("Big O Complexity Growth")
plt.legend()
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt

# O(1) Solution
def sum_o1(n):
    """Calculate the sum using a mathematical formula (constant time)."""
    iterations = 1  # Only one operation
    result = n * (n + 1) // 2
    print(f"O(1) iterations: {iterations}")
    return result, iterations

# O(n) Solution
def sum_on(n):
    """Calculate the sum by iterating through numbers 1 to n."""
    iterations = 0
    total = 0
    for i in range(1, n + 1):
        iterations += 1
        total += i
    print(f"O(n) iterations: {iterations}")
    return total, iterations

# O(n log n) Solution
def sum_onlogn_helper(arr, iterations):
    """Helper function to sum an array using divide and conquer."""
    if len(arr) <= 1:
        return sum(arr), iterations
    mid = len(arr) // 2
    iterations += 1
    left_sum, iterations = sum_onlogn_helper(arr[:mid], iterations)
    right_sum, iterations = sum_onlogn_helper(arr[mid:], iterations)
    return left_sum + right_sum, iterations

def sum_onlogn(n):
    """Calculate the sum using a divide-and-conquer approach."""
    arr = list(range(1, n + 1))
    total, iterations = sum_onlogn_helper(arr, 0)
    print(f"O(n log n) iterations: {iterations}")
    return total, iterations

# O(n^2) Solution
def sum_on2(n):
    """Calculate the sum using nested loops."""
    iterations = 0
    total = 0
    for i in range(1, n + 1):
        for j in range(1, i + 1):
            iterations += 1
            total += 1
    print(f"O(n^2) iterations: {iterations}")
    return total, iterations

# Compare the methods
n_values = [10, 50, 100, 200, 500]
o1_iterations = []
on_iterations = []
onlogn_iterations = []
on2_iterations = []

for n in n_values:
    _, iter_o1 = sum_o1(n)
    _, iter_on = sum_on(n)
    _, iter_onlogn = sum_onlogn(n)
    _, iter_on2 = sum_on2(n)

    o1_iterations.append(iter_o1)
    on_iterations.append(iter_on)
    onlogn_iterations.append(iter_onlogn)
    on2_iterations.append(iter_on2)

# Plot the results
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# O(1) Plot
axs[0, 0].plot(n_values, o1_iterations, label="O(1)", marker="o")
axs[0, 0].set_title("O(1) Complexity")
axs[0, 0].set_xlabel("Input size (n)")
axs[0, 0].set_ylabel("Iterations")
axs[0, 0].grid()

# O(n) Plot
axs[0, 1].plot(n_values, on_iterations, label="O(n)", marker="o", color="orange")
axs[0, 1].set_title("O(n) Complexity")
axs[0, 1].set_xlabel("Input size (n)")
axs[0, 1].set_ylabel("Iterations")
axs[0, 1].grid()

# O(n log n) Plot
axs[1, 0].plot(n_values, onlogn_iterations, label="O(n log n)", marker="o", color="green")
axs[1, 0].set_title("O(n log n) Complexity")
axs[1, 0].set_xlabel("Input size (n)")
axs[1, 0].set_ylabel("Iterations")
axs[1, 0].grid()

# O(n^2) Plot
axs[1, 1].plot(n_values, on2_iterations, label="O(n^2)", marker="o", color="red")
axs[1, 1].set_title("O(n^2) Complexity")
axs[1, 1].set_xlabel("Input size (n)")
axs[1, 1].set_ylabel("Iterations")
axs[1, 1].grid()

# Adjust layout
plt.tight_layout()
plt.show()
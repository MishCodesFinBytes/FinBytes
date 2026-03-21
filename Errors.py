'''
Explanation:
SyntaxError: This occurs when the Python code doesn't adhere to the syntax rules (e.g., a typo in return).
NameError: Raised when you reference a variable before it's defined.
ValueError: Happens when an operation is performed on an invalid value (e.g., summing integers with a string).
TypeError: Occurs when an operation involves incompatible types (e.g., adding an integer and a string).
IndexError: Raised when you try to access an index that is out of range in a list.
ZeroDivisionError: Happens when you try to divide by zero.
FileNotFoundError: Raised when trying to open a file that doesn't exist.
KeyError: Occurs when you try to access a dictionary with a key that doesn’t exist.
The fixes are provided along with each error type to show how to resolve the issue and ensure smooth execution.
'''



# -----------------------------
# 1. Syntax Error Example
# -----------------------------
# A SyntaxError occurs when the Python interpreter encounters a statement that it doesn't understand.
# In this case, we have a typo in the keyword 'return' (written as 'retun').

# SyntaxError Example:
# Uncommenting the following line will cause a SyntaxError
# def calculate_interest(principal, rate, time):
#     retun principal * rate * time

# SyntaxError Fixed:
def calculate_interest(principal, rate, time):
    # Correcting 'retun' to 'return'
    return principal * rate * time

# -----------------------------
# 2. Name Error Example
# -----------------------------
# A NameError occurs when you try to use a variable that has not been defined.
# In this case, the variable 'cost' is being used but hasn't been assigned a value.

# NameError Example:
portfolio_value = 50000
# Uncommenting the following line will cause a NameError because 'cost' is not defined
# roi = (portfolio_value - cost) / cost * 100

# NameError Fixed:
portfolio_value = 50000
cost = 40000  # Now defining 'cost' with a value
roi = (portfolio_value - cost) / cost * 100  # Now this works correctly

# -----------------------------
# 3. Value Error Example
# -----------------------------
# A ValueError occurs when you try to perform an operation on a value that is not compatible.
# In this case, adding an integer and a string in the 'prices' list will cause an error.

# ValueError Example:
prices = [100, 110, 120, "130", 140]  # '130' is a string, not an integer
# Uncommenting the following line will cause a ValueError because 'sum' cannot handle the string
# average_price = sum(prices) / len(prices)

# ValueError Fixed:
prices = [100, 110, 120, 130, 140]  # Ensuring all items are integers
average_price = sum(prices) / len(prices)  # Now it works correctly

# -----------------------------
# 4. Type Error Example
# -----------------------------
# A TypeError occurs when an operation is performed on incompatible types.
# Here, we are trying to add an integer to a string, which is not allowed.

# TypeError Example:
# Uncommenting the following line will cause a TypeError because you can't add int and str
# result = 5 + "hello"

# TypeError Fixed:
result = 5 + 10  # Correcting the operation to add integers
print(result)  # Output: 15

# -----------------------------
# 5. Index Error Example
# -----------------------------
# An IndexError occurs when you try to access an index that is out of range for a list.

# IndexError Example:
my_list = [1, 2, 3]
# Uncommenting the following line will cause an IndexError because index 5 does not exist
# print(my_list[5])

# IndexError Fixed:
my_list = [1, 2, 3]
print(my_list[2])  # Output: 3 (Valid index)

# -----------------------------
# 6. ZeroDivisionError Example
# -----------------------------
# A ZeroDivisionError occurs when dividing by zero.

# ZeroDivisionError Example:
# Uncommenting the following line will cause a ZeroDivisionError because division by zero is not allowed
# result = 10 / 0

# ZeroDivisionError Fixed:
result = 10 / 2  # This works because we are dividing by a non-zero number
print(result)  # Output: 5.0

# -----------------------------
# 7. FileNotFoundError Example
# -----------------------------
# A FileNotFoundError occurs when you try to open a file that doesn't exist.

# FileNotFoundError Example:
# Uncommenting the following line will cause a FileNotFoundError if the file does not exist
# with open("non_existent_file.txt", "r") as file:
#     content = file.read()

# FileNotFoundError Fixed:
# Assuming 'existent_file.txt' exists, this will work fine
with open("existent_file.txt", "r") as file:
    content = file.read()
    print(content)

# -----------------------------
# 8. Key Error Example
# -----------------------------
# A KeyError occurs when trying to access a key that does not exist in a dictionary.

# KeyError Example:
my_dict = {"name": "Alice", "age": 25}
# Uncommenting the following line will cause a KeyError because "address" is not a key in the dictionary
# print(my_dict["address"])

# KeyError Fixed:
my_dict = {"name": "Alice", "age": 25, "address": "123 Street"}
print(my_dict["address"])  # Output: 123 Street

# -----------------------------
# Conclusion
# -----------------------------
# These are some common errors you may encounter:
# - SyntaxError: Issues with incorrect code syntax.
# - NameError: Trying to use undefined variables.
# - ValueError: Performing invalid operations on incompatible values.
# - TypeError: Incompatible types in an operation.
# - IndexError: Accessing an out-of-range index.
# - ZeroDivisionError: Dividing by zero.
# - FileNotFoundError: Trying to open a non-existent file.
# - KeyError: Accessing a dictionary with a non-existent key.

# -----------------------------------------------
# 1. Using a List of Dictionaries
# -----------------------------------------------
# Simple dataset representation using dictionaries
dataset_dicts = [
    {"id": 1, "name": "Alice", "age": 25, "city": "London"},
    {"id": 2, "name": "Bob", "age": 30, "city": "New York"},
    {"id": 3, "name": "Charlie", "age": 35, "city": "Paris"}
]

# Accessing data from list of dictionaries
print("List of Dictionaries:")
for record in dataset_dicts:
    print(record["name"], record["city"])
print()

# -----------------------------------------------
# 2. Using pandas
# -----------------------------------------------
# Tabular data representation using pandas DataFrame
import pandas as pd

data_pandas = {
    "id": [1, 2, 3],
    "name": ["Alice", "Bob", "Charlie"],
    "age": [25, 30, 35],
    "city": ["London", "New York", "Paris"]
}

df = pd.DataFrame(data_pandas)

# Accessing data from pandas DataFrame
print("pandas DataFrame:")
print(df.head())          # First few rows
print(df["name"])         # Access a column
print(df[df["age"] > 25]) # Filter rows
print()

# -----------------------------------------------
# 3. Using numpy
# -----------------------------------------------
# Numerical dataset representation using numpy
import numpy as np

data_numpy = np.array([
    [1, 25, "London"],
    [2, 30, "New York"],
    [3, 35, "Paris"]
])

# Accessing data from numpy array
print("numpy Array:")
print(data_numpy[0])          # First record
print(data_numpy[:, 1])       # Second column (age)
print()

# -----------------------------------------------
# 4. Using PyTorch
# -----------------------------------------------
# Dataset for machine learning using PyTorch
import torch

data_pytorch = torch.tensor([[1.0, 25.0], [2.0, 30.0], [3.0, 35.0]])

# Accessing data from PyTorch tensor
print("PyTorch Tensor:")
print(data_pytorch[0])          # First record
print()

# -----------------------------------------------
# 5. Using TensorFlow
# -----------------------------------------------
# Dataset for machine learning using TensorFlow
import tensorflow as tf

data_tf = tf.data.Dataset.from_tensor_slices({
    "id": [1, 2, 3],
    "age": [25, 30, 35],
    "city": ["London", "New York", "Paris"]
})

# Accessing data from TensorFlow Dataset
print("TensorFlow Dataset:")
for record in data_tf:
    print(record)
print()

# -----------------------------------------------
# 6. Using a Custom Class
# -----------------------------------------------
# Custom class for dataset representation
class Dataset:
    def __init__(self, data):
        self.data = data

    def get_record(self, index):
        return self.data[index]

data_class = [
    {"id": 1, "name": "Alice", "age": 25},
    {"id": 2, "name": "Bob", "age": 30}
]
dataset = Dataset(data_class)

# Accessing data from custom class
print("Custom Class:")
print(dataset.get_record(0))
print()

# -----------------------------------------------
# 7. Using dataclass
# -----------------------------------------------
# Structured dataset representation using dataclass
from dataclasses import dataclass

@dataclass
class Person:
    id: int
    name: str
    age: int
    city: str

dataset_dataclass = [
    Person(1, "Alice", 25, "London"),
    Person(2, "Bob", 30, "New York"),
    Person(3, "Charlie", 35, "Paris")
]

# Accessing data from dataclass
print("dataclass:")
for person in dataset_dataclass:
    print(f"{person.name} lives in {person.city} and is {person.age} years old.")
print()

# -----------------------------------------------
# 8. Using Nested dataclass
# -----------------------------------------------
# More complex dataset with nested dataclasses
@dataclass
class Address:
    city: str
    country: str

@dataclass
class PersonWithAddress:
    id: int
    name: str
    age: int
    address: Address

dataset_nested_dataclass = [
    PersonWithAddress(1, "Alice", 25, Address("London", "UK")),
    PersonWithAddress(2, "Bob", 30, Address("New York", "USA")),
    PersonWithAddress(3, "Charlie", 35, Address("Paris", "France"))
]

# Accessing nested dataclass data
print("Nested dataclass:")
for person in dataset_nested_dataclass:
    print(f"{person.name} lives in {person.address.city}, {person.address.country}.")
print()

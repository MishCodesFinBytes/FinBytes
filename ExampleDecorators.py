### Built-in Function Decorators

# Example of @staticmethod
# This decorator is used to define a static method in a class.
# Static methods do not depend on instance or class variables.
def static_method_example():
    class MyClass:
        @staticmethod
        def static_method():
            return "Static method"

# Example of @classmethod
# This decorator is used to define a class method.
# Class methods receive the class itself as the first argument (commonly named 'cls').
def class_method_example():
    class MyClass:
        @classmethod
        def class_method(cls):
            return f"Called on {cls.__name__}"

# Example of @property
# This decorator is used to create getter methods for attributes.
# It allows a method to be accessed like an attribute without parentheses.
def property_example():
    class MyClass:
        def __init__(self, value):
            self._value = value

        @property
        def value(self):
            return self._value

# Example of @functools.lru_cache
# This decorator caches the results of a function for faster repeated calls.
from functools import lru_cache

def lru_cache_example():
    @lru_cache
    def expensive_function(x):
        return x * x

# Example of @functools.wraps
# This decorator is used when creating custom decorators.
# It preserves the metadata (like function name and docstring) of the original function.
from functools import wraps

def wraps_example():
    def my_decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            print("Before function call")
            return func(*args, **kwargs)
        return wrapper

### Class-Related Decorators

# Example of @dataclass
# This decorator automatically generates methods like __init__, __repr__, and __eq__.
from dataclasses import dataclass

def dataclass_example():
    @dataclass
    class MyClass:
        name: str
        age: int

# Example of @abc.abstractmethod
# This decorator is used in abstract base classes (ABCs).
# It enforces the implementation of methods in derived classes.
from abc import ABC, abstractmethod

def abstract_method_example():
    class MyAbstractClass(ABC):
        @abstractmethod
        def abstract_method(self):
            pass

    class ConcreteClass(MyAbstractClass):
        def abstract_method(self):
            return "Implementation of abstract method"

### Debugging and Testing Decorators

# Example of @unittest.skip
# This decorator is used to skip a specific test in the unittest framework.
import unittest

def unittest_skip_example():
    class MyTest(unittest.TestCase):
        @unittest.skip("Skip this test")
        def test_example(self):
            pass

# Example of @unittest.expectedFailure
# This decorator marks a test that is expected to fail.
def unittest_expected_failure_example():
    @unittest.expectedFailure
    def test_fail(self):
        self.assertEqual(1, 2)

### Function-Execution Control Decorators

# Example of @contextlib.contextmanager
# This decorator simplifies the creation of context managers.
from contextlib import contextmanager

def context_manager_example():
    @contextmanager
    def my_context():
        print("Enter")
        yield
        print("Exit")

### Miscellaneous Decorators

# @staticmethod and @classmethod (Already explained above)

# Example of functools.partial
# This is not strictly a decorator, but it is used to partially apply arguments to a function.
def functools_partial_example():
    from functools import partial

    def multiply(x, y):
        return x * y

    # Create a new function 'double' that always multiplies by 2
    double = partial(multiply, 2)
    print(double(5))  # Output: 10

### Common Third-Party Decorators

# Example of @pytest.mark.parametrize
# This decorator is used in pytest to run a test with multiple sets of parameters.
import pytest

def pytest_parametrize_example():
    @pytest.mark.parametrize("input,expected", [(2, 4), (3, 9)])
    def test_square(input, expected):
        assert input ** 2 == expected

# Example of @flask.route
# This decorator is used in the Flask web framework to define HTTP routes.
from flask import Flask

def flask_route_example():
    app = Flask(__name__)

    @app.route('/')
    def home():
        return "Hello, World!"

# Example of @click.command
# This decorator is used in the Click library to define command-line interfaces.
import click

def click_command_example():
    @click.command()
    def cli():
        click.echo("Hello, CLI!")

### Custom Decorators

# Example of a custom decorator
# Custom decorators are user-defined and can modify the behavior of functions.
def custom_decorator_example():
    def my_decorator(func):
        def wrapper(*args, **kwargs):
            print("Before function call")
            result = func(*args, **kwargs)
            print("After function call")
            return result
        return wrapper

    @my_decorator
    def say_hello():
        print("Hello!")

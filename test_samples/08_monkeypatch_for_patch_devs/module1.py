"""
module1.py — Consumer module used in:
"Monkeypatch for a unittest.mock.patch Developer"
"""

from module2 import Class2
import os


class Class1:
    def __init__(self):
        self.helper = Class2()

    def compute(self, a, b):
        return Class2.multiply(a, b) + 10

    def compute_instance(self, value):
        return self.helper.instance_double(value) + 5

    def get_env(self):
        return os.environ.get("API_KEY", "missing")

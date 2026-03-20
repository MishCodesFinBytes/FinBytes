"""
module1.py — Consumer module used in:
"pytest Monkeypatch: A Practical Field Guide (with Inline Return Patterns)"
"""

from module2 import Class2


class Class1:
    def __init__(self):
        self.helper = Class2()

    def compute_library(self, a, b):
        return Class2.multiply(a, b) + Class2.add(a, b)

    def compute_instance(self, value):
        return self.helper.instance_double(value) + 5

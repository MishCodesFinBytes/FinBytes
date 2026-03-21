"""
module2.py — Dependency module used in:
"pytest Monkeypatch: A Practical Field Guide (with Inline Return Patterns)"
"""


class Class2:
    def multiply(x, y):
        return x * y

    def add(x, y):
        return x + y

    def instance_double(self, value):
        return value * 2

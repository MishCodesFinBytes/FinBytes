"""
module2.py — Dependency module used in:
"Monkeypatch for a unittest.mock.patch Developer"
"""


class Class2:
    def multiply(x, y):
        return x * y

    def add(x, y):
        return x + y

    def instance_double(self, value):
        return value * 2

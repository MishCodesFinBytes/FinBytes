"""
service.py — Production module used in:
"Mastering Python Patch Targets: Top-Level Functions, Instance Methods,
 and Static Methods"
"""


# Top-level function
def compute(x):
    return x * 2


class CalculatorA:
    def compute(self, x):
        return x + 1

    @staticmethod
    def compute_static(x):
        return x * 10

    @classmethod
    def compute_class(cls, x):
        return x * 100


class CalculatorB:
    def compute(self, x):
        return x + 10

"""
services.py — Dependency module used in:
"Monkeypatching the Correct Target in Pytest"
"""


def helper():
    return "real helper"


def top_level_function():
    return "real top level"


class A:
    def method(self):
        return "real A.method"

    @staticmethod
    def static_method():
        return "real A.static"

    def call_helper(self):
        return helper()


class B:
    def method(self):
        return "real B.method"

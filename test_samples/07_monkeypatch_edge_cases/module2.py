"""
module2.py — Dependency module used in:
"Monkeypatch Edge Cases That Break Senior Engineers"
"""


class Class2:
    def do_work(self):
        return "real work"

    @staticmethod
    def static_op():
        return "real static"

    @classmethod
    def class_op(cls, x):
        return x * 2

    @property
    def label(self):
        return "real label"


class ExternalClient:
    def connect(self):
        return "real connection"


async def async_fetch():
    return "real async result"


CONFIG = {"timeout": 30, "retries": 3}

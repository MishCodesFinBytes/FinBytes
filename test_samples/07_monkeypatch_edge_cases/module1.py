"""
module1.py — Consumer module used in:
"Monkeypatch Edge Cases That Break Senior Engineers"
"""

from module2 import Class2, ExternalClient
import module2


class Service:
    def __init__(self):
        # Dependency bound at construction time — patch BEFORE instantiation
        self.client = ExternalClient()

    def run(self):
        return Class2().do_work()

    def use_client(self):
        return self.client.connect()

    def use_config(self):
        return module2.CONFIG["timeout"]

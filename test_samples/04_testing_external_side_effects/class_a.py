"""
class_a.py — Production class used in:
"Testing External Side Effects: Files, Emails, and APIs — A Unified Strategy"
"""

from email_lib import send_email


class ClassA:
    def process(self):
        send_email(
            to=["user1@test.com"],
            cc=[],
            sender="noreply@test.com",
            subject="Report 1",
            body="First report",
        )
        send_email(
            to=["user2@test.com"],
            cc=["manager@test.com"],
            sender="noreply@test.com",
            subject="Report 2",
            body="Second report",
        )

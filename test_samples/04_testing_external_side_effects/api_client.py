"""
api_client.py — Production API client used in:
"Testing External Side Effects: Files, Emails, and APIs — A Unified Strategy"
"""

import requests


class APIClient:
    def fetch_data(self, url):
        response = requests.get(url)
        return response.json()

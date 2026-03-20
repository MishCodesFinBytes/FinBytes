"""
processor.py — Consumer module used in:
"Monkeypatching the Correct Target in Pytest"

Note the two different import styles:
  - top_level_function and A imported directly (from services import ...)
  - B accessed via module reference (import services; services.B())

This difference determines the correct patch target for each.
"""

from services import top_level_function, A
import services


def run_all():
    results = []
    results.append(top_level_function())   # direct import → patch processor.top_level_function
    results.append(A().method())            # direct import → patch processor.A.method
    results.append(A.static_method())       # direct import → patch processor.A.static_method
    results.append(services.B().method())   # module ref   → patch services.B.method
    results.append(A().call_helper())       # helper lives in services → patch services.helper
    return results

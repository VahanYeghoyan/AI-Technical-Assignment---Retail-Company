"""Test package.

Exists so `from tests.test_resilience import FakeBQClient` resolves — the agent
tests reuse the BigQuery fakes rather than defining a second, drifting copy.
"""

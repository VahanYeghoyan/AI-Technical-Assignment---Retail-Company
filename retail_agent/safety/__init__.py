"""Deterministic safety layers.

Nothing in this package asks the LLM for permission. Every rule here is enforced
in code, before a query runs or after results come back, so that a jailbroken or
simply confused model cannot widen its own access.
"""

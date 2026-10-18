"""S25 canonical corpus: read-only inventory, version catalog, canonical build and verification.

Every writer in this package writes only below a fresh S25 root (see `guard.py`) and never modifies
an existing artifact. Nothing here deletes or moves files.
"""

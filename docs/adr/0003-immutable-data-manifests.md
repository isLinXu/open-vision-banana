# ADR-0003: Require Immutable Dataset and Experiment Manifests

- Status: Accepted
- Date: 2026-07-25

## Context

Multi-task datasets reuse source images across labels, making random sample splits vulnerable to train/eval leakage. Mutable directories and undocumented preprocessing also prevent independent reproduction.

## Decision

Every training and evaluation run records a canonical config hash, Git state, runtime versions, resolved data roots, and dataset fingerprints. Production data versions use per-sample manifests with source IDs and content hashes. Split assignment is group-based, and benchmark val/test hashes must have zero intersection with training data.

## Consequences

- Data changes create a new version and fingerprint instead of overwriting history.
- Full content hashing costs I/O; metadata hashing remains available for fast local preflight but is insufficient for release evidence.
- Results without their manifest are exploratory and cannot enter release tables.

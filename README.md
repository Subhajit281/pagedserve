# PagedServe

A from-scratch continuous-batching LLM inference engine
with a paged KV-cache.

## Target Hardware

- GPU: NVIDIA RTX 3050 Laptop GPU
- VRAM: 4 GB
- CPU: Intel Core i5-12500H
- RAM: 16 GB
- Runtime: WSL2 Ubuntu 24.04

## Primary Model

Qwen2.5-0.5B-Instruct

## Core Goals

- Continuous / iteration-level batching
- Paged KV-cache
- Preemption by recomputation
- Streaming inference API
- Reproducible benchmarking

## Status

Phase 0 — Foundations

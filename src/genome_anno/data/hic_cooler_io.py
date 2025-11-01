from __future__ import annotations
import cooler

def open_cool(path: str):
    return cooler.Cooler(path)

from __future__ import annotations
import pysam

def open_bam(path: str) -> pysam.AlignmentFile:
    return pysam.AlignmentFile(path, "rb")

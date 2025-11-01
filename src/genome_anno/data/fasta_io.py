from __future__ import annotations
from typing import Iterator
from Bio import SeqIO

def read_fasta(path: str) -> Iterator[SeqIO.SeqRecord]:
    return SeqIO.parse(path, "fasta")

def to_one_hot(seq: str) -> list[list[int]]:
    # A,C,G,T -> 4-канальный one-hot
    m = {"A":0, "C":1, "G":2, "T":3}
    L = [[0,0,0,0] for _ in range(len(seq))]
    for i, ch in enumerate(seq.upper()):
        if ch in m:
            L[i][m[ch]] = 1
    return L

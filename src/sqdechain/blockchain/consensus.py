from __future__ import annotations

from dataclasses import dataclass, field
from typing import List
import hashlib
import time

from sqdechain.blockchain.racs import Validator


@dataclass
class Transaction:
    tx_id: str
    user_id: str
    kpi_hash: str
    payload_size: int = 512


@dataclass
class Block:
    block_id: str
    shard_id: int
    transactions: List[Transaction]
    proposer: str
    previous_hash: str = "0" * 64
    timestamp: float = field(default_factory=time.time)
    block_hash: str = ""

    def finalize_hash(self) -> str:
        body = f"{self.block_id}|{self.shard_id}|{self.proposer}|{self.previous_hash}|" + "|".join(tx.kpi_hash for tx in self.transactions)
        self.block_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
        return self.block_hash


def max_byzantine_faults(n: int) -> int:
    return (n - 1) // 3


def quorum_size(n: int) -> int:
    return 2 * max_byzantine_faults(n) + 1


@dataclass
class ConsensusResult:
    committed: bool
    prepare_votes: int
    commit_votes: int
    quorum: int
    latency_ms: float
    block_hash: str


class ClusterPioneerBFT:
    """Deterministic local simulation of shard-level BFT finality."""

    def __init__(self, shard_id: int, committee: List[Validator], base_latency_ms: float = 250.0):
        self.shard_id = shard_id
        self.committee = committee
        self.base_latency_ms = base_latency_ms

    def validate_block(self, block: Block, malicious_validators: int = 0) -> ConsensusResult:
        n = len(self.committee)
        q = quorum_size(n)
        honest = max(n - malicious_validators, 0)
        prepare_votes = honest
        commit_votes = honest if prepare_votes >= q else 0
        committed = commit_votes >= q
        block_hash = block.finalize_hash() if committed else ""
        latency_ms = self.base_latency_ms + 12.5 * n + 20.0 * malicious_validators
        return ConsensusResult(
            committed=committed,
            prepare_votes=prepare_votes,
            commit_votes=commit_votes,
            quorum=q,
            latency_ms=float(latency_ms),
            block_hash=block_hash,
        )

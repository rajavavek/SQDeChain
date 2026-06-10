from sqdechain.blockchain.racs import make_validators, select_committee
from sqdechain.blockchain.consensus import max_byzantine_faults, quorum_size, ClusterPioneerBFT, Block, Transaction


def test_quorum():
    assert max_byzantine_faults(11) == 3
    assert quorum_size(11) == 7


def test_committee_and_consensus():
    vals = make_validators(28, seed=1)
    committee = select_committee(vals, committee_size=11, max_per_region=4)
    assert len(committee) == 11
    tx = Transaction("tx1", "u1", "a" * 64)
    block = Block("b1", 0, [tx], committee[0].validator_id)
    res = ClusterPioneerBFT(0, committee).validate_block(block, malicious_validators=3)
    assert res.committed
    assert len(res.block_hash) == 64

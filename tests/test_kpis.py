from sqdechain.kpis import service_quality_trust_index, destination_image_reliability_score, tokenised_loyalty_propensity, allocate_tokens, gini, hash_kpi_bundle


def test_sqti_range():
    score = service_quality_trust_index([0.2, 0.8, 1.0], [1, 2, 3])
    assert 0 <= score <= 1


def test_dirs_range():
    score = destination_image_reliability_score([0.8, 0.9], [0.7, 0.8], [1.0, 0.9])
    assert 0 <= score <= 1


def test_tlp_tokens_hash():
    tlp = tokenised_loyalty_propensity(0.7, 0.6, 0.8, 0.9)
    assert 0 <= tlp <= 1
    assert allocate_tokens(tlp, 100, 50) <= 50
    h1 = hash_kpi_bundle({"a": 1, "b": 2})
    h2 = hash_kpi_bundle({"b": 2, "a": 1})
    assert h1 == h2


def test_gini():
    assert gini([1, 1, 1]) == 0
    assert gini([0, 0, 10]) > 0

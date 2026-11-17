import pytest

from anomaly.data import train_subset


def test_fraction_size_seeded_and_distinct():
    a = train_subset(280, fraction=0.25, seed=0)
    assert len(a) == 70 and len(set(a)) == 70 and a == sorted(a)
    assert a == train_subset(280, fraction=0.25, seed=0)
    assert a != train_subset(280, fraction=0.25, seed=1)
    assert 0 <= min(a) and max(a) < 280


def test_n_shot_and_full_set():
    assert len(train_subset(280, n_shot=10)) == 10
    assert train_subset(5, fraction=1.0) == [0, 1, 2, 3, 4]


def test_too_small_or_too_large_raises():
    with pytest.raises(ValueError):
        train_subset(280, n_shot=1)
    with pytest.raises(ValueError):
        train_subset(5, n_shot=6)

import pytest
import tensorflow as tf


def test_splitting_to_buckets():
    """
    Accompanying code to SO answer
    https://stackoverflow.com/a/70905839/10561443
    """

    dataset = tf.data.Dataset.from_tensor_slices([f"value-{i}" for i in range(10000)])

    def to_bucket(sample):
        return tf.strings.to_hash_bucket_fast(sample, 5)

    def filter_train_fn(sample):
        return tf.math.not_equal(to_bucket(sample), 0)

    def filter_val_fn(sample):
        return tf.math.logical_not(filter_train_fn(sample))

    train_ds = dataset.filter(filter_train_fn)
    val_ds = dataset.filter(filter_val_fn)

    len_train_ds = len(list(train_ds.as_numpy_iterator()))
    len_val_ds = len(list(val_ds.as_numpy_iterator()))

    assert len_train_ds == pytest.approx(8000, rel=1e-2)
    assert len_val_ds == pytest.approx(2000, rel=1e-2)

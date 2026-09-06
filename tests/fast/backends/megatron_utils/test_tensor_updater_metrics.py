from orbit.backends.megatron_utils.update_weight.update_weight_from_tensor import UpdateWeightFromTensor


def test_pop_metrics_without_recorded_metrics_returns_empty():
    updater = UpdateWeightFromTensor.__new__(UpdateWeightFromTensor)

    assert updater.pop_metrics() == {}
    assert updater.pop_metrics() == {}
    assert "update_weight_metrics" not in vars(updater)


def test_pop_metrics_drains_existing_values_exactly_once():
    updater = UpdateWeightFromTensor.__new__(UpdateWeightFromTensor)
    metrics = {"update_weight/encode_ms": 0.0, "update_weight/e2e_ms": 12.5}
    updater.update_weight_metrics = metrics

    assert updater.pop_metrics() is metrics
    assert metrics == {"update_weight/encode_ms": 0.0, "update_weight/e2e_ms": 12.5}
    assert "update_weight_metrics" not in vars(updater)
    assert updater.pop_metrics() == {}

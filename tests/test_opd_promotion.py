from orbit.utils.opd_teacher_spec import should_promote_teacher


def test_non_self_sources_never_promote():
    assert not should_promote_teacher("adapter", 1, 0)
    assert not should_promote_teacher("base", 1, 0)
    assert not should_promote_teacher("load", 1, 0)


def test_no_interval_never_promotes():
    assert not should_promote_teacher("self_ema", None, 0)


def test_promotes_at_startup_and_on_interval():
    # rollout_id 0 = startup promotion (fills the empty engine slot before
    # the first scored rollout).
    assert should_promote_teacher("self_ema", 3, 0)
    assert not should_promote_teacher("self_ema", 3, 1)
    assert not should_promote_teacher("self_ema", 3, 2)
    assert should_promote_teacher("self_ema", 3, 3)
    assert should_promote_teacher("self_lag", 1, 7)

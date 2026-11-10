from phone_trace import counter_delta, to_trace_ns


def test_counter_delta_interpolates_between_samples() -> None:
    """
    Given: a cumulative energy counter sampled at t = 0, 10, 20 with values 0, 100, 300.
    When: the increase over [5, 15] and over [10, 20] is computed.
    Then: [5, 15] gives 150 (50 from each side, interpolated) and [10, 20] gives the exact sample difference 200.
    """
    series = [(0, 0.0), (10, 100.0), (20, 300.0)]
    assert counter_delta(series, 5, 15) == 150.0
    assert counter_delta(series, 10, 20) == 200.0


def test_counter_delta_is_none_when_samples_do_not_cover_window() -> None:
    """
    Given: a counter sampled only from t = 10 to t = 20, and a counter with one sample.
    When: a window starting before 10 or ending after 20 is asked for, and any window of the single-sample counter.
    Then: the result is None instead of an extrapolated guess.
    """
    series = [(10, 0.0), (20, 50.0)]
    assert counter_delta(series, 5, 15) is None
    assert counter_delta(series, 12, 25) is None
    assert counter_delta([(10, 1.0)], 10, 10) is None


def test_to_trace_ns_applies_phone_offset_and_boottime_offset() -> None:
    """
    Given: laptop time 100 s, a phone clock 0.5 s ahead, and REALTIME - BOOTTIME of 40 s on the phone.
    When: the laptop time is converted to trace time.
    Then: the trace timestamp is (100 + 0.5 - 40) s = 60.5 s in nanoseconds.
    """
    assert to_trace_ns(100.0, 0.5, 40_000_000_000) == 60_500_000_000

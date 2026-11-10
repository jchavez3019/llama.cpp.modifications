from run_matrix import configs, run_order, server_args


def test_run_order_rotates_start_config_per_repetition_and_covers_every_run() -> None:
    """
    Given: two models, configurations A, B, C, and three repetitions.
    When: the run order is built.
    Then: each repetition runs every configuration once, the start configuration rotates (A, then B, then C), models run one after another, and no run is missing or repeated.
    """
    order = run_order(["m1", "m2"], ["A", "B", "C"], 3)
    m1 = [(rep, name) for model, rep, name in order if model == "m1"]
    assert m1 == [(0, "A"), (0, "B"), (0, "C"), (1, "B"), (1, "C"), (1, "A"), (2, "C"), (2, "A"), (2, "B")]
    assert [model for model, _, _ in order] == ["m1"] * 9 + ["m2"] * 9
    assert len(set(order)) == len(order) == 18


def test_cache_ram_default_applies_to_a_b_e_but_not_to_c_d_f() -> None:
    """
    Given: a phone setup with --cache-ram-default 1024 and --d-cache-ram 32.
    When: the server arguments of every configuration are built.
    Then: A, B, E get 1024, C and F keep 0 (no RAM cache), D keeps 32, and without a default A passes no --cache-ram flag at all.
    """
    def cache_ram(name: str, default: int | None) -> str | None:
        """Returns the --cache-ram value of one configuration's server arguments, or None if the flag is absent."""
        args = server_args("m.gguf", configs(32, default)[name], 4096, 8090, [])
        return args[args.index("--cache-ram") + 1] if "--cache-ram" in args else None

    assert [cache_ram(n, 1024) for n in "ABCDEF"] == ["1024", "1024", "0", "32", "1024", "0"]
    assert cache_ram("A", None) is None

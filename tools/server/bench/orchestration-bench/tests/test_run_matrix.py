import sys
import time
from pathlib import Path

from run_matrix import LocalServer, configs, run_order, server_args


def test_local_server_is_running_until_it_exits_or_is_stopped(tmp_path: Path) -> None:
    """
    Given: one local process that exits at once (a crash during loading) and one that keeps running.
    When: is_running is checked, and the running one is stopped.
    Then: the exited process reports False, the running one reports True until it is stopped, then False.
    """
    crashed = LocalServer(sys.executable, ["-c", "raise SystemExit(1)"], tmp_path / "crashed.log")
    deadline = time.monotonic() + 10
    while crashed.is_running() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not crashed.is_running()
    crashed.stop(tmp_path / "crashed.log")

    running = LocalServer(sys.executable, ["-c", "import time; time.sleep(60)"], tmp_path / "running.log")
    assert running.is_running()
    running.stop(tmp_path / "running.log")
    assert not running.is_running()


def test_tree_configs_add_cache_tree_and_f_tree_uses_the_cache_size_of_a() -> None:
    """
    Given: a phone setup with --cache-ram-default 1024 and --d-cache-ram 32.
    When: the server arguments of the tree configurations are built.
    Then: A-tree, D-tree, and F-tree pass --cache-tree, A-tree and F-tree get A's 1024 MiB, D-tree keeps 32, and the stock configurations do not pass --cache-tree.
    """
    def args_of(name: str) -> list[str]:
        """Returns the server arguments of one configuration."""
        return server_args("m.gguf", configs(32, 1024)[name], 4096, 8090, [])

    for name, ram in (("A-tree", "1024"), ("D-tree", "32"), ("F-tree", "1024")):
        args = args_of(name)
        assert "--cache-tree" in args and args[args.index("--cache-ram") + 1] == ram
    assert not any("--cache-tree" in args_of(n) for n in "ABCDEF")


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

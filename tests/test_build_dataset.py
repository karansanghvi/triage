from datetime import datetime, timedelta, timezone

from eval.build_dataset import PACIFIC, Limits, Pacer, simulate_finish


def test_cache_hits_do_not_count_against_the_daily_token_budget():
    pacer = Pacer(Limits(tpd=200_000), history=[])
    for _ in range(10):
        pacer.record(api_calls=0, tokens=19_000)      # cache hits: 190K tokens Groq never charged for
    assert pacer.daily_budget_left(4_000)
    pacer.record(api_calls=1, tokens=190_000)         # a real call does count
    assert not pacer.daily_budget_left(20_000)


def test_cache_hits_do_not_count_against_the_daily_request_budget():
    pacer = Pacer(Limits(rpd=2), history=[])
    pacer.record(api_calls=0, tokens=100)
    pacer.record(api_calls=0, tokens=100)
    assert pacer.daily_budget_left(1)
    pacer.record(api_calls=2, tokens=100)
    assert not pacer.daily_budget_left(1)


def test_cache_hits_do_not_use_up_the_per_minute_budget():
    pacer = Pacer(Limits(tpm=8_000), history=[])
    pacer.record(api_calls=0, tokens=7_500)
    assert pacer.wait_for_minute_slot(4_000) == 0      # returns immediately, no sleep


def test_history_from_earlier_runs_still_counts():
    now = datetime.now(timezone.utc)
    pacer = Pacer(Limits(tpd=200_000), history=[(now - timedelta(hours=1), 1, 198_000)])
    assert not pacer.daily_budget_left(4_000)


# ---------- finish simulation ----------

PROMPT = {"messages": [{"role": "user", "content": "x" * 400}]}   # ~100 input tokens


def test_simulation_without_limits_is_calls_times_seconds():
    start = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)
    end = simulate_finish([PROMPT] * 10, lambda p: 500, [], Limits(), start, seconds_per_call=2, max_tokens=4000)
    assert end == start + timedelta(seconds=20)


def test_simulation_waits_for_old_usage_to_leave_the_24h_window():
    start = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)
    used_at = start - timedelta(hours=2)
    window = [(used_at, 198_000)]          # daily budget almost gone two hours ago
    end = simulate_finish([PROMPT], lambda p: 500, window, Limits(tpd=200_000), start, 1, 4000)
    # can't start until that usage is 24h old (22h from now), plus the 30s safety margin
    assert end >= used_at + timedelta(hours=24, seconds=30)
    assert end < used_at + timedelta(hours=24, minutes=1)


def test_simulation_spreads_heavy_work_over_several_days():
    start = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)
    prompts = [PROMPT] * 300                 # 300 x 1,500 tokens = 450K, at 200K/day
    end = simulate_finish(prompts, lambda p: 1_500, [], Limits(tpd=200_000), start, 1, 4000)
    assert timedelta(days=2) < end - start < timedelta(days=3)


def test_simulation_respects_midnight_pacific_reset():
    start = datetime(2026, 10, 20, 19, 0, tzinfo=timezone.utc)   # noon in Pacific time
    end = simulate_finish([PROMPT] * 3, lambda p: 100, [], Limits(rpd=2, daily_reset="midnight_pacific"),
                          start, 1, 4000)
    assert end.astimezone(PACIFIC).date() > start.astimezone(PACIFIC).date()


"""
Time spent per reviewer per paper.

The app calls `accrue` on every tick (a fragment that reruns every `TICK` seconds
while the paper is open in a browser tab) and on every interaction. Time counts while
the paper is open, with three limits:

  - after `IDLE_LIMIT` without any interaction, counting stops (a tab left open
    overnight adds nothing); the next interaction starts it again;
  - while the reviewer has paused the timer, nothing counts; any interaction also
    ends a pause, so a forgotten pause cannot lose time;
  - a single gap between ticks counts for at most `GAP_CAP` (a laptop asleep with the
    tab open is not reading time).

Everything here is a pure function of timestamps (seconds since the epoch), so it is
tested with a fake clock.
"""

from __future__ import annotations

TICK = 30              # seconds between timer ticks while a paper is open
IDLE_LIMIT = 15 * 60   # stop counting after this long without interaction
GAP_CAP = 90           # never credit more than this for one gap between ticks


def accrue(last_tick: float | None, now: float, last_interaction: float,
           paused: bool, idle_limit: float = IDLE_LIMIT, gap_cap: float = GAP_CAP) -> float:
    """Seconds to credit for the interval (last_tick, now]."""
    if paused or last_tick is None or now <= last_tick:
        return 0.0
    end = min(now, last_interaction + idle_limit)
    if end <= last_tick:
        return 0.0
    return min(end - last_tick, gap_cap)


def is_idle(now: float, last_interaction: float, idle_limit: float = IDLE_LIMIT) -> bool:
    return now - last_interaction > idle_limit


def format_duration(seconds: float) -> str:
    minutes = int(seconds // 60)
    if minutes < 60:
        return f"{minutes} min"
    return f"{minutes // 60} h {minutes % 60:02d} min"

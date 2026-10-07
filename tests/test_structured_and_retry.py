import random

import pytest

from guarded_agent.retry import RetryPolicy
from guarded_agent.structured import StructuredOutputError, extract_json, parse_structured
from tests.conftest import Answer


@pytest.mark.parametrize(
    "text",
    [
        '{"status": "done", "detail": "x"}',
        'Here you go:\n```json\n{"status": "done", "detail": "x"}\n```',
        'Result: {"status": "done", "detail": "x"} -- end',
    ],
)
def test_extract_json_tolerates_wrapping(text: str) -> None:
    assert extract_json(text) == {"status": "done", "detail": "x"}


def test_parse_structured_reports_field_errors() -> None:
    with pytest.raises(StructuredOutputError, match="detail"):
        parse_structured('{"status": "done"}', Answer)
    with pytest.raises(StructuredOutputError, match="not valid JSON"):
        parse_structured("no json here", Answer)


def test_backoff_grows_exponentially_and_caps() -> None:
    policy = RetryPolicy(base_delay_s=1.0, max_delay_s=5.0, jitter=0.0)
    rng = random.Random(0)
    assert [policy.delay_s(n, rng) for n in range(1, 5)] == [1.0, 2.0, 4.0, 5.0]


def test_backoff_jitter_stays_within_bounds() -> None:
    policy = RetryPolicy(base_delay_s=1.0, jitter=0.2)
    rng = random.Random(42)
    delays = [policy.delay_s(1, rng) for _ in range(200)]
    assert min(delays) >= 0.8
    assert max(delays) <= 1.2
    assert len(set(delays)) > 1

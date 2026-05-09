from __future__ import annotations

from sae_muc.models import FakeBackend, Generation


def test_generate_shape():
    backend = FakeBackend()
    out = backend.generate(["q1", "q2", "q3"], temperature=1.0, max_new_tokens=32, n=4)
    assert len(out) == 3
    for per_prompt in out:
        assert len(per_prompt) == 4
        for g in per_prompt:
            assert isinstance(g, Generation)
            assert isinstance(g.text, str)
            assert g.text != ""


def test_generate_is_deterministic():
    a = FakeBackend().generate(["q"], temperature=1.0, max_new_tokens=32, n=5)
    b = FakeBackend().generate(["q"], temperature=1.0, max_new_tokens=32, n=5)
    assert [g.text for g in a[0]] == [g.text for g in b[0]]


def test_prompt_affects_output():
    backend = FakeBackend()
    a = backend.generate(["q_one"], temperature=1.0, max_new_tokens=32, n=10)
    b = backend.generate(["q_two"], temperature=1.0, max_new_tokens=32, n=10)
    # Not every entry has to differ, but the streams should not be identical.
    assert [g.text for g in a[0]] != [g.text for g in b[0]]


def test_temperature_affects_output():
    backend = FakeBackend()
    cold = backend.generate(["q"], temperature=0.1, max_new_tokens=32, n=10)
    hot = backend.generate(["q"], temperature=1.0, max_new_tokens=32, n=10)
    assert [g.text for g in cold[0]] != [g.text for g in hot[0]]

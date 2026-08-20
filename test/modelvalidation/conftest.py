"""Shared fakes for modelvalidation tests.

Everything here is deterministic and free of pricing, so the pipeline can be
exercised end-to-end in milliseconds. The real engines appear only in the
study-level tests.
"""

from __future__ import annotations

import pytest

from quantark.modelvalidation.candidate import CandidateResult, LadderRung
from quantark.modelvalidation.reference import BatchResult
from quantark.modelvalidation.study import (
    CaseSpec,
    CertificationStudy,
    GateBounds,
    HedgeContractScale,
    SamplingPolicy,
)


def make_scale() -> HedgeContractScale:
    return HedgeContractScale(200.0, 100.0, 50_000_000.0)


def raw_from_contracts(contracts: float, quantity: str = "delta") -> float:
    """Raw engine value worth ``contracts`` economic units."""
    return contracts / make_scale().to_economic(quantity, 1.0)


class SteadyReference:
    """A benchmark whose batches agree tightly around a per-case mean."""

    def __init__(self, means_c=None, tag="fake.reference", jitter_c=0.001):
        # case name -> economic-unit mean for every quantity
        self.means_c = means_c or {}
        self.tag = tag
        self.jitter_c = jitter_c
        self.calls = 0

    def identity(self, case):
        return {
            "builder": self.tag,
            "case": case.name,
            "mean_c": self.means_c.get(case.name, 0.0),
            "jitter_c": self.jitter_c,
        }

    def run_batch(self, case, batch_index):
        self.calls += 1
        mean_c = self.means_c.get(case.name, 0.0)
        # Alternating tiny jitter: a well-defined, tiny standard error.
        offset = self.jitter_c if batch_index % 2 else -self.jitter_c
        values = {
            quantity: raw_from_contracts(mean_c + offset, quantity)
            for quantity in ("pv", "delta", "gamma")
        }
        return BatchResult(index=batch_index, seed=SEED + batch_index, values=values)


class OffsetCandidate:
    """A deterministic engine sitting a fixed economic offset from the benchmark."""

    def __init__(self, name="fake.candidate", offset_c=0.0, means_c=None, envelope_c=0.0):
        self._name = name
        self.offset_c = offset_c
        self.means_c = means_c or {}
        self.envelope_c = envelope_c
        self.calls = 0

    def name(self):
        return self._name

    def params(self):
        return {"offset_c": self.offset_c, "envelope_c": self.envelope_c}

    def evaluate(self, case):
        self.calls += 1
        mean_c = self.means_c.get(case.name, 0.0)
        values = {
            quantity: raw_from_contracts(mean_c + self.offset_c, quantity)
            for quantity in ("pv", "delta", "gamma")
        }
        target = LadderRung(axis="grid", level="target", values=values)
        medium = LadderRung(
            axis="grid",
            level="medium",
            values={
                quantity: raw_from_contracts(
                    mean_c + self.offset_c + self.envelope_c, quantity
                )
                for quantity in ("pv", "delta", "gamma")
            },
        )
        return CandidateResult(values=values, ladders=(target, medium))


class ExplodingCandidate:
    """A candidate that raises, to exercise the ERROR path."""

    def __init__(self, name="fake.exploding", exc=None):
        self._name = name
        self.exc = exc or RuntimeError("engine blew up")

    def name(self):
        return self._name

    def params(self):
        return {}

    def evaluate(self, case):
        raise self.exc


SEED = 100

CASE_MEANS_C = {"ordinary": 1.0, "near_ko": 2.0}


def make_study(**overrides) -> CertificationStudy:
    """A two-case study over fakes; overrides replace any field."""
    means = overrides.pop("means_c", CASE_MEANS_C)
    kwargs = dict(
        name="fake-study",
        schema=1,
        cases=(CaseSpec(name="ordinary"), CaseSpec(name="near_ko")),
        quantities=("pv", "delta", "gamma"),
        bounds=GateBounds(cell=0.5, mean_signed_bias=0.1),
        scale=make_scale(),
        reference=SteadyReference(means_c=means),
        candidates=(OffsetCandidate(means_c=means),),
        sampling=SamplingPolicy(
            paths_per_batch=1024, min_batches=2, max_batches=8, seed=SEED
        ),
        source_text="study: fake-study\n",
    )
    kwargs.update(overrides)
    return CertificationStudy(**kwargs)


@pytest.fixture
def study():
    return make_study()

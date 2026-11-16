import pytest

from genai_suite import Orchestrator
from genai_suite.llm import LocalExtractiveBackend
from genai_suite.roi import estimate_roi, resolution_mix


@pytest.fixture(scope="module")
def orch():
    return Orchestrator.from_data(backend=LocalExtractiveBackend())


def test_classifier_fits_training_data(orch):
    label, conf = orch.classifier.predict("my reimbursement for the hotel was rejected")
    assert label == "expense" and conf > 0.5


@pytest.mark.parametrize("query,source", [
    ("how many days of annual leave carry forward", "leave_policy.md"),
    ("vendor invoice missing purchase order", "procurement_policy.md"),
    ("refund timeline for card payments", "refund_sop.md"),
])
def test_retrieval_finds_right_policy(orch, query, source):
    hits = orch.store.search(query, k=1)
    assert hits[0][0].metadata["source"] == source


def test_end_to_end_answer_is_cited_and_grounded(orch):
    state = orch.run("Can I still submit my expense claim, it is 40 days old? email me at a@b.com")
    assert state.pii == {"EMAIL": 1}
    assert "[EMAIL]" in state.redacted and "a@b.com" not in state.final
    assert "[1]" in state.draft and "30 days" in state.draft
    assert state.grounding >= 0.7 and state.resolution == "auto_resolved"
    assert [s.agent for s in state.trace][:2] == ["intake", "intake"]


def test_critic_retry_with_query_expansion(orch):
    state = orch.run("how many quotes do we need to buy monitors worth 4 lakh")
    assert state.attempts >= 1
    assert "quotations" in state.draft


def test_high_priority_is_escalated(orch):
    assert orch.run("URGENT I lost my laptop at the airport").resolution == "escalated"


def test_roi_math():
    roi = estimate_roi({"auto_resolved": 0.5, "agent_assist": 0.5, "escalated": 0.0})
    assert roi["minutes_per_request_after"] == pytest.approx(2.75)
    assert roi["hours_saved_per_month"] > 0 and roi["payback_months"] > 0
    assert resolution_mix([]) == {"auto_resolved": 0, "agent_assist": 0, "escalated": 0}

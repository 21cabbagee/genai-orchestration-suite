"""Value model: turns observed resolution rates from a batch of runs into an ROI estimate.

The inputs are assumptions you agree with the business owner; the resolution mix comes from
actually running the pipeline on a sample of their requests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .agents import TaskState


@dataclass
class ROIAssumptions:
    monthly_volume: int = 4000            # requests per month
    baseline_minutes: float = 12.0        # manual handling time per request
    assist_minutes: float = 5.0           # handling time when an agent edits a draft
    escalated_minutes: float = 11.0       # escalations still need most of the manual effort
    hourly_cost: float = 900.0            # fully loaded cost per support hour (INR)
    build_cost: float = 600000.0          # one-off build and integration (INR)
    run_cost_monthly: float = 45000.0     # hosting + model usage (INR)


def resolution_mix(runs: list[TaskState]) -> dict[str, float]:
    total = len(runs) or 1
    keys = ("auto_resolved", "agent_assist", "escalated")
    return {k: sum(r.resolution == k for r in runs) / total for k in keys}


def estimate_roi(mix: dict[str, float], a: ROIAssumptions = ROIAssumptions()) -> dict:
    minutes_after = (mix.get("auto_resolved", 0) * 0.5
                     + mix.get("agent_assist", 0) * a.assist_minutes
                     + mix.get("escalated", 0) * a.escalated_minutes)
    saved_hours = a.monthly_volume * (a.baseline_minutes - minutes_after) / 60
    gross = saved_hours * a.hourly_cost
    net = gross - a.run_cost_monthly
    return {
        "assumptions": asdict(a),
        "mix": {k: round(v, 3) for k, v in mix.items()},
        "minutes_per_request_before": a.baseline_minutes,
        "minutes_per_request_after": round(minutes_after, 2),
        "hours_saved_per_month": round(saved_hours, 1),
        "gross_saving_per_month": round(gross),
        "net_saving_per_month": round(net),
        "payback_months": round(a.build_cost / net, 1) if net > 0 else None,
        "first_year_roi_pct": round((net * 12 - a.build_cost) / a.build_cost * 100, 1),
    }

"""Run the Python pipeline on sample requests and export everything the GitHub Pages demo needs.

    python scripts/build_demo.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from genai_suite import Orchestrator  # noqa: E402
from genai_suite.agents.specialists import CATEGORY_QUEUE, CATEGORY_SOURCE  # noqa: E402
from genai_suite.llm import LocalExtractiveBackend  # noqa: E402
from genai_suite.roi import ROIAssumptions, estimate_roi, resolution_mix  # noqa: E402

SAMPLES = [
    "My expense claim from the Delhi trip is 40 days old, can I still submit it? Reach me at ravi.k@acme.in",
    "How many quotes do we need to buy 20 monitors worth 4 lakh?",
    "URGENT: I lost my laptop at the airport this morning",
    "Customer raised a chargeback but is also asking for a refund on ORD-55120",
    "How many days of annual leave can I carry forward into next year?",
    "Customer asked us to delete all the personal data we hold about them",
    "Vendor invoice INV-2231 is on hold, what is missing?",
    "My account is locked after the password change, how do I get back in?",
    "Who needs to approve my expense claim of INR 32,000?",
    "Can I work from home three days a week?",
]


def main() -> None:
    orch = Orchestrator.from_data(backend=LocalExtractiveBackend())
    runs = [orch.run(s) for s in SAMPLES]
    out = ROOT / "docs" / "data"
    out.mkdir(parents=True, exist_ok=True)
    model = {
        "retrieval": orch.store.export(),
        "classifier": orch.classifier.export(),
        "category_source": CATEGORY_SOURCE,
        "category_queue": CATEGORY_QUEUE,
    }
    (out / "model.json").write_text(json.dumps(model, separators=(",", ":")))
    (out / "runs.json").write_text(json.dumps([r.model_dump() for r in runs], indent=1))
    mix = resolution_mix(runs)
    (out / "roi.json").write_text(json.dumps(estimate_roi(mix, ROIAssumptions()), indent=1))
    print(f"Exported {len(runs)} runs; resolution mix {mix}")


if __name__ == "__main__":
    main()

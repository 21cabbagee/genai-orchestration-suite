"""CLI: python -m genai_suite "How many days of leave can I carry forward?" """

import json
import sys

from .orchestrator import Orchestrator


def main() -> None:
    request = " ".join(sys.argv[1:]) or "My expense claim is 40 days old, can I still submit it?"
    orch = Orchestrator.from_data()
    state = orch.run(request)
    print(f"\nBackend: {orch.backend.name}\n")
    for step in state.trace:
        print(f"  [{step.agent:>9}] {step.action:<13} {json.dumps(step.detail, ensure_ascii=False)}")
    print(f"\nResolution: {state.resolution}\n\n{state.final}\n")


if __name__ == "__main__":
    main()

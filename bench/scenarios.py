"""Handwritten cross-domain fixtures. These are demonstrations, not production accuracy estimates."""

from __future__ import annotations

import json
from pathlib import Path

from jselect import Index, select

SCENARIOS = [
    {
        "name": "support",
        "task": "Why are people abandoning signup, and what helps them get through it?",
        "repeated": "I entered my email during signup but no verification link arrived. "
        "I tried again and gave up.",
        "repeated_facet": "verification",
        "other": [
            (
                "card",
                "The free trial wanted my credit card before I could register. I closed the page instead.",
            ),
            (
                "sso",
                "Registration says my company uses single sign-on, but our IT team has never configured it.",
            ),
            (
                "workaround",
                "I finally finished signup by opening the email link in the browser where I started.",
            ),
        ],
        "noise": "Please reschedule my meeting with the design team for Friday afternoon.",
    },
    {
        "name": "code",
        "task": "Find reasons this worker process keeps retaining memory after jobs finish.",
        "repeated": "# Completed worker jobs stay reachable from a global list.\ncompleted_jobs.append(job)",
        "repeated_facet": "retained_jobs",
        "other": [
            ("cache", "# Worker request cache has no size limit or eviction.\nrequest_cache[key] = response"),
            (
                "listener",
                "# A listener is added per job and never removed.\nemitter.on('result', lambda: job.payload)",
            ),
            (
                "released",
                "# Worker cleanup releases large buffers when a job finishes.\n"
                "job.payload = None\nbuffers.clear()",
            ),
        ],
        "noise": "# Styles for the navigation bar.\nNAVIGATION_COLOR = '#003366'",
    },
    {
        "name": "contracts",
        "task": "What determines whether we can end this vendor subscription "
        "without paying for another year?",
        "repeated": "The subscription renews for twelve months unless written notice "
        "is received sixty days before renewal.",
        "repeated_facet": "notice",
        "other": [
            (
                "exception",
                "A customer may terminate immediately without future fees after "
                "an uncured material breach lasting thirty days.",
            ),
            (
                "fees",
                "Convenience termination during an active annual term "
                "does not cancel outstanding fees for that term.",
            ),
            (
                "amendment",
                "This signed amendment replaces automatic renewal "
                "with month-to-month service after the initial year.",
            ),
        ],
        "noise": "Employees taking annual leave must submit vacation requests to their line manager.",
    },
]


def run():
    reports = []
    for case in SCENARIOS:
        # IDs/labels are evaluation metadata; the model only sees each passage's text.
        data = [
            {"id": f"{case['repeated_facet']}-{i}", "text": case["repeated"] + f"\nReference {100 + i}."}
            for i in range(24)
        ]
        data += [{"id": facet + "-0", "text": text} for facet, text in case["other"]]
        data += [{"id": f"noise-{i}", "text": case["noise"] + f"\nReference {200 + i}."} for i in range(48)]
        expected = {case["repeated_facet"], *(facet for facet, _ in case["other"])}
        with Index.build(data) as index:
            balanced = select(index, task=case["task"], tokens=1000, max_items=4, mode="semantic")
            ranked = select(index, task=case["task"], tokens=1000, max_items=4, mode="semantic", diversity=0)

        def facets(result):
            return {item.sources[0]["record_id"].rsplit("-", 1)[0] for item in result.items}

        reports.append(
            {
                "name": case["name"],
                "task": case["task"],
                "records": len(data),
                "expected_facets": sorted(expected),
                "balanced_facets": sorted(facets(balanced)),
                "ranked_facets": sorted(facets(ranked)),
                "balanced_coverage": len(expected & facets(balanced)) / len(expected),
                "ranked_coverage": len(expected & facets(ranked)) / len(expected),
                "balanced": balanced.to_dict(),
                "ranked": ranked.to_dict(),
            }
        )
        print(
            json.dumps(
                {
                    k: reports[-1][k]
                    for k in (
                        "name",
                        "balanced_facets",
                        "ranked_facets",
                        "balanced_coverage",
                        "ranked_coverage",
                    )
                }
            ),
            flush=True,
        )
    out = Path(__file__).parent / "out"
    out.mkdir(exist_ok=True)
    (out / "scenarios.json").write_text(json.dumps(reports, indent=2))


if __name__ == "__main__":
    run()

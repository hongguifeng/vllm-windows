import os
import re

IDS = [
    "arcane-drift-detection-baselines",
    "bandit-structured-nosec-directives",
    "effect-sse-httpapi-streaming",
    "go-critic-doc-link-checker",
    "koota-entity-snapshot-rollback",
    "kysely-window-grouping-helpers",
    "langchain-request-coalescing",
    "meriyah-explicit-resource-declarations",
    "true-myth-iterable-collection-combinators",
    "vitest-duration-sharding",
]
ROOT = "/home/hong/deep-swe/tasks"


def field(text, key):
    match = re.search(rf'^{key} = "(.*)"', text, re.M)
    return match.group(1) if match else "?"


for task in IDS:
    path = os.path.join(ROOT, task)
    toml = open(f"{path}/task.toml", errors="replace").read()
    patch = os.path.join(path, "solution", "solution.patch")
    files = adds = dels = 0
    if os.path.exists(patch):
        diff = open(patch, errors="replace").read()
        adds = sum(1 for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++"))
        dels = sum(1 for l in diff.splitlines() if l.startswith("-") and not l.startswith("---"))
        files = len(re.findall(r"^diff --git", diff, re.M))
    print(
        f"{task:<45} lang={field(toml,'language'):<10} "
        f"gold_files={files:<2} +{adds}/-{dels}  repo={field(toml,'repository_url')}"
    )

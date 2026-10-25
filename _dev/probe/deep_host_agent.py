"""Host-side mini-swe-agent for DeepSWE tasks under Harbor.

Runs the LLM loop on the host (no in-sandbox agent install) and commits the
agent's edits afterwards, because DeepSWE's collect hook grades
`git diff <base_commit> HEAD`.
"""

import shlex

from host_mini_swe import HostMiniSweAgent


class CommitHostMiniSwe(HostMiniSweAgent):
    @staticmethod
    def name() -> str:
        return "host-mini-swe-commit"

    async def run(self, instruction, environment, context):
        await super().run(instruction, environment, context)
        command = (
            "repo=$(git -C /app rev-parse --show-toplevel 2>/dev/null "
            "|| git -C /testbed rev-parse --show-toplevel 2>/dev/null || echo /app); "
            'cd "$repo" && git add -A && '
            'git -c user.email=agent@local -c user.name=agent commit -m "agent changes" '
            "--allow-empty >/dev/null 2>&1 || true; "
            'git log --oneline -1'
        )
        result = await environment.exec(command=command, timeout_sec=180)
        self.logger.info(f"commit after agent: {shlex.quote((result.stdout or '').strip())}")

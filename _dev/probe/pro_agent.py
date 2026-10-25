"""LockedMiniSwe with the `mcp` package added to the in-sandbox install.

The stock install pins mini-swe-agent + litellm + orjson + fastapi, but the
Responses API client used when `reasoning_effort` is set imports `mcp` at
runtime, which the pin does not include. Patch the install command in place so
the rest of Harbor's locked protocol (offline agent, patch capture) is intact.
"""

from locked_mini_swe import LockedMiniSwe


class ProLockedMiniSwe(LockedMiniSwe):
    @staticmethod
    def name() -> str:
        return "mini-swe-agent-locked-mcp"

    async def exec_as_agent(self, environment, command, **kwargs):
        if "uv tool install" in command and "--with mcp" not in command:
            command = command.replace("--with litellm", "--with litellm --with mcp")
        return await super().exec_as_agent(environment, command, **kwargs)

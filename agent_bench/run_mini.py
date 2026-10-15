#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Run mini-SWE-agent against a local vLLM endpoint on a local SWE-bench subset."""

import argparse
import concurrent.futures
import json
import logging
import time
from pathlib import Path

import yaml
from minisweagent.config import builtin_config_dir
from minisweagent.run.benchmarks.swebench import process_instance
from minisweagent.run.benchmarks.utils.batch_progress import RunBatchProgressManager
from minisweagent.utils.log import add_file_handler, logger


def merge(base: dict, extra: dict) -> dict:
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            merge(base[k], v)
        else:
            base[k] = v
    return base


def build_config(
    api_base,
    max_tokens,
    temperature,
    step_limit,
    enable_thinking,
    reasoning_effort,
    model_name,
    tool_call_format="",
):
    cfg = yaml.safe_load(
        (builtin_config_dir / "benchmarks" / "swebench.yaml").read_text()
    )
    chat_kwargs = {"enable_thinking": enable_thinking}
    if tool_call_format:
        chat_kwargs["tool_call_format"] = tool_call_format
    if enable_thinking:
        chat_kwargs["reasoning_effort"] = reasoning_effort
    return merge(
        cfg,
        {
            "agent": {"step_limit": step_limit, "cost_limit": 0},
            "environment": {"pull_timeout": 3600},
            "model": {
                "model_name": model_name,
                "cost_tracking": "ignore_errors",
                "model_kwargs": {
                    "api_base": api_base,
                    "api_key": "EMPTY",
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "timeout": 600,
                    "drop_params": True,
                    "parallel_tool_calls": True,
                    "extra_body": {"chat_template_kwargs": chat_kwargs},
                },
            },
        },
    )


def install_strip_thinking_patch():
    """Drop reasoning_content from history so past thinking never re-enters the prompt.

    Only use this when an endpoint's chat template re-renders every previous
    thinking block unconditionally; measure first with diagnose_thinking.py.
    """
    from minisweagent.models import litellm_model as lm

    original = lm.LitellmModel._prepare_messages_for_api

    def patched(self, messages):
        prepared = original(self, messages)
        for m in prepared:
            if m.get("role") == "assistant":
                m.pop("reasoning_content", None)
                m.pop("provider_specific_fields", None)
        return prepared

    lm.LitellmModel._prepare_messages_for_api = patched


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subset", required=True)
    p.add_argument(
        "--model-name",
        default="openai/Qwen3.8-Flash-Next",
        help="litellm model string; must include the openai/ prefix",
    )
    p.add_argument(
        "--tool-call-format",
        default="",
        help="passed into chat_template_kwargs when the chat template supports it",
    )
    p.add_argument("--output", required=True)
    p.add_argument("--api-base", required=True)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--max-tokens", type=int, default=1536)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--step-limit", type=int, default=60)
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument(
        "--reasoning-effort", default="medium", choices=["low", "medium", "xhigh"]
    )
    p.add_argument(
        "--strip-thinking-history",
        action="store_true",
        help="do not send previous thinking back to the server",
    )
    p.add_argument("--only", default="", help="comma separated instance ids")
    p.add_argument("--redo-existing", action="store_true")
    a = p.parse_args()

    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    add_file_handler(out / "minisweagent.log")

    with open(a.subset) as f:
        instances = json.load(f)
    if a.only:
        ids = set(a.only.split(","))
        instances = [i for i in instances if i["instance_id"] in ids]
    preds_path = out / "preds.json"
    if not a.redo_existing and preds_path.exists():
        done = set(json.loads(preds_path.read_text()).keys())
        instances = [i for i in instances if i["instance_id"] not in done]
        logger.info(f"Skipping {len(done)} existing instances")
    if a.strip_thinking_history:
        install_strip_thinking_patch()
        logger.info("history thinking will be STRIPPED before sending")
    effort = a.reasoning_effort if a.enable_thinking else "off"
    logger.info(
        f"Running {len(instances)} instances against {a.api_base} "
        f"(thinking={a.enable_thinking}, effort={effort}, "
        f"workers={a.workers}, max_tokens={a.max_tokens}, "
        f"step_limit={a.step_limit})"
    )

    config = build_config(
        a.api_base,
        a.max_tokens,
        a.temperature,
        a.step_limit,
        a.enable_thinking,
        a.reasoning_effort,
        a.model_name,
        a.tool_call_format,
    )
    pm = RunBatchProgressManager(
        len(instances), out / f"exit_statuses_{int(time.time())}.yaml"
    )
    done = 0

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as ex:
        futures = {
            ex.submit(process_instance, inst, out, config, pm): inst["instance_id"]
            for inst in instances
        }
        for fut in concurrent.futures.as_completed(futures):
            iid = futures[fut]
            try:
                fut.result()
            except Exception as e:
                logger.error(f"Uncaught error for {iid}: {e}", exc_info=True)
                pm.on_uncaught_exception(iid, e)
            done += 1
            logger.info(
                f"[{done}/{len(futures)} done in {time.time() - t0:.0f}s] last: {iid}"
            )
    pm.print_report()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

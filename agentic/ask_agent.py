from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from agentic.tool_executor import ToolExecutor


@dataclass(frozen=True)
class AgentConfig:
    max_steps: int = 6
    budget_chars: int = 24000
    confidence_threshold: float = 0.75
    temperature: float = 0.0
    timeout: int = 600
    mode: str = "general"  # auto | general | pr_review
    pr_base: str = "main"
    pr_head: str = "HEAD"
    pr_repo_root: str = "."


class AskAgent:
    """LLM-driven tool loop over ToolExecutor."""

    def __init__(
        self,
        llm: Any,
        executor: ToolExecutor,
        config: AgentConfig | None = None,
        on_event=None,
    ) -> None:
        self.llm = llm
        self.executor = executor
        self.config = config or AgentConfig()
        self.on_event = on_event

    def run(self, question: str) -> dict[str, Any]:
        tools = self.executor.available_tools()
        evidence: list[dict[str, Any]] = []
        messages: list[dict[str, Any]] = []
        total_cost = 0
        warnings: list[str] = []
        mode = self._resolve_mode(question)
        self._emit("mode_selected", {"mode": mode})

        for step in range(1, self.config.max_steps + 1):
            self._emit("step_start", {"step": step})
            prompt = self._planning_prompt(question, tools, evidence, step, mode)
            raw = self.llm.generate(
                prompt=prompt,
                temperature=self.config.temperature,
                timeout=self.config.timeout,
            )
            action = _parse_json(raw)
            messages.append({"step": step, "raw": raw, "parsed": action})
            self._emit("planner_action", {"step": step, "action": action})

            if action.get("action") == "final_answer":
                confidence = float(action.get("confidence", 0.0) or 0.0)
                answer = str(action.get("answer", "")).strip()
                if answer and confidence >= self.config.confidence_threshold:
                    self._emit(
                        "final_answer",
                        {"step": step, "confidence": confidence, "accepted": True},
                    )
                    return {
                        "status": "completed",
                        "mode": mode,
                        "question": question,
                        "answer": answer,
                        "confidence": confidence,
                        "steps_used": step,
                        "total_cost": total_cost,
                        "evidence": evidence,
                        "messages": messages,
                        "warnings": warnings,
                    }
                warnings.append(
                    f"Step {step}: final_answer confidence {confidence:.2f} below threshold "
                    f"{self.config.confidence_threshold:.2f}; continuing."
                )
                self._emit(
                    "final_answer",
                    {"step": step, "confidence": confidence, "accepted": False},
                )
                continue

            if action.get("action") != "tool_call":
                warnings.append(_invalid_action_warning(step=step, raw=raw, parsed=action))
                self._emit(
                    "planner_invalid_action",
                    {
                        "step": step,
                        "raw": raw,
                        "parsed": action,
                    },
                )
                continue

            tool_name = str(action.get("tool", "")).strip()
            tool_args = action.get("args", {})
            if not isinstance(tool_args, dict):
                warnings.append(f"Step {step}: tool args must be object/dict.")
                continue

            try:
                self._emit("tool_execute_start", {"step": step, "tool": tool_name, "args": tool_args})
                execution = self.executor.execute(tool_name, tool_args)
                self._emit(
                    "tool_execute_done",
                    {"step": step, "tool": tool_name, "cost": execution.get("cost", 0)},
                )
            except Exception as exc:  # pragma: no cover - runtime/error dependent
                warnings.append(f"Step {step}: tool execution failed for '{tool_name}': {exc}")
                self._emit("tool_execute_error", {"step": step, "tool": tool_name, "error": str(exc)})
                continue

            total_cost += int(execution.get("cost", 0))
            summary = _summarize_result(execution.get("result"), max_chars=1800)
            evidence.append(
                {
                    "step": step,
                    "tool": tool_name,
                    "args": tool_args,
                    "cost": execution.get("cost", 0),
                    "summary": summary,
                    "result": execution.get("result"),
                }
            )

            # Hard budget guard over accumulated evidence summaries.
            current_size = sum(len(item.get("summary", "")) for item in evidence)
            while current_size > self.config.budget_chars and evidence:
                dropped = evidence.pop(0)
                current_size -= len(dropped.get("summary", ""))
                warnings.append(
                    f"Dropped oldest evidence step {dropped.get('step')} to honor budget."
                )

        # Final fallback answer synthesis if agent loop did not terminate with high confidence.
        fallback_prompt = self._fallback_answer_prompt(question, evidence, warnings, mode)
        self._emit("fallback_start", {"reason": "max_steps_or_low_confidence"})
        fallback_answer = self.llm.generate(
            prompt=fallback_prompt,
            temperature=self.config.temperature,
            timeout=self.config.timeout,
        )
        self._emit("fallback_done", {})
        return {
            "status": "max_steps_reached",
            "mode": mode,
            "question": question,
            "answer": fallback_answer.strip(),
            "confidence": None,
            "steps_used": self.config.max_steps,
            "total_cost": total_cost,
            "evidence": evidence,
            "messages": messages,
            "warnings": warnings,
        }

    def _planning_prompt(
        self,
        question: str,
        tools: list[dict[str, Any]],
        evidence: list[dict[str, Any]],
        step: int,
        mode: str,
    ) -> str:
        mode_directive = self._mode_directive(mode)
        repo_question_guidance = (
            "Repository-level question guidance:\n"
            "- If the user asks to explain a repository, architecture, modules, components, or requests a diagram/mermaid graph,\n"
            "  do not stop after only one shallow repo-level tool result unless the evidence already includes concrete module/component relationships.\n"
            "- Prefer gathering both:\n"
            "  1) repo-level structure via inspect_repo\n"
            "  2) at least one deeper follow-up such as module_context, symbol_context, module_dependencies, or find_symbol_like\n"
            "- For architecture or diagram requests, prefer another tool_call over final_answer when evidence contains only README/top-level structure.\n"
            "- Use final_answer only when you can name the main modules/components and describe how they relate.\n"
        )
        json_contract_guidance = (
            "JSON contract reminders:\n"
            "- The action field must be exactly one of: tool_call, final_answer\n"
            "- Never put a tool name inside action\n"
            '- Valid tool call example: {"action":"tool_call","tool":"impact_context","args":{"symbol":"X","depth":2},"reason":"Need impact details"}\n'
            '- Invalid example: {"action":"impact_context","args":{"symbol":"X","depth":2},"reason":"..."}\n'
            '- Valid final answer example: {"action":"final_answer","answer":"...","confidence":0.84}\n'
        )
        return (
            "You are a code intelligence planning agent.\n"
            "Given a question, choose one next tool call or provide final answer.\n"
            "Rules:\n"
            "1) Return strict JSON only. Do not return prose, markdown, or explanation outside the JSON object.\n"
            "2) If more data is needed, return:\n"
            '{"action":"tool_call","tool":"<tool_name>","args":{...},"reason":"..."}\n'
            "3) If enough evidence exists, return:\n"
            '{"action":"final_answer","answer":"...","confidence":0.0}\n'
            "4) Keep confidence conservative.\n"
            "5) When evidence is shallow, incomplete, or only repo-top-level, prefer tool_call over final_answer.\n"
            "6) If you want to answer directly, you must still use action=final_answer JSON.\n\n"
            f"Mode:\n{mode}\n"
            f"Mode directive:\n{mode_directive}\n\n"
            f"{json_contract_guidance}\n"
            f"{repo_question_guidance}\n"
            f"Step: {step}\n"
            f"Question:\n{question}\n\n"
            f"Available tools:\n{json.dumps(tools, indent=2)}\n\n"
            f"Evidence so far:\n{json.dumps(evidence, indent=2)}\n"
        )

    def _emit(self, event: str, payload: dict[str, Any]) -> None:
        if self.on_event is None:
            return
        try:
            self.on_event(event, payload)
        except Exception:
            return

    @staticmethod
    def _fallback_answer_prompt(
        question: str,
        evidence: list[dict[str, Any]],
        warnings: list[str],
        mode: str,
    ) -> str:
        if mode == "pr_review":
            return (
                "You are a senior PR risk reviewer.\n"
                "Use only the provided evidence. Do not invent numbers.\n"
                "Output exactly this markdown structure:\n"
                "PR Risk Assessment\n"
                "------------------\n\n"
                "Risk: <LOW|MEDIUM|HIGH>\n\n"
                "Reasoning:\n"
                "• <bullet with evidence number>\n"
                "• <bullet with evidence number>\n"
                "• <bullet with evidence number>\n\n"
                "High-risk symbols:\n"
                "1. <symbol>\n"
                "2. <symbol>\n"
                "3. <symbol>\n\n"
                "Recommendation:\n"
                "Merge is likely safe if the following modules are tested:\n"
                "- <module>\n"
                "- <module>\n"
                "- <module>\n\n"
                f"Question:\n{question}\n\n"
                f"Evidence:\n{json.dumps(evidence, indent=2)}\n\n"
                f"Warnings:\n{json.dumps(warnings, indent=2)}\n"
            )
        return (
            "You are a code intelligence assistant. Produce the best possible answer "
            "from available evidence. If uncertain, clearly state uncertainty.\n\n"
            f"Question:\n{question}\n\n"
            f"Evidence:\n{json.dumps(evidence, indent=2)}\n\n"
            f"Warnings:\n{json.dumps(warnings, indent=2)}\n\n"
            "Final answer:"
        )

    def _resolve_mode(self, question: str) -> str:
        mode = self.config.mode.lower().strip()
        if mode in {"general", "pr_review"}:
            return mode
        lowered = question.lower()
        pr_patterns = (
            r"\bpr\b",
            r"\bpull request\b",
            r"\bmerge\b",
            r"\bdiff\b",
            r"\bsafe to merge\b",
            r"\brisk\b",
        )
        if any(re.search(pattern, lowered) for pattern in pr_patterns):
            return "pr_review"
        return "general"

    def _mode_directive(self, mode: str) -> str:
        if mode != "pr_review":
            return "General analysis mode."
        return (
            "You are in PR review mode.\n"
            "Preferred first tool call: pr_diff_context.\n"
            "Use these default args unless the user overrides them in the question:\n"
            f'{{"base":"{self.config.pr_base}","head":"{self.config.pr_head}",'
            f'"repo_root":"{self.config.pr_repo_root}","depth":3,"format":"summary",'
            '"top_symbol_contexts":5,"top_module_contexts":3}\n'
            "After pr_diff_context, optionally call symbol_context or stack_trace for top risky symbols.\n"
            "When you have enough evidence, return final_answer in markdown PR reviewer style with:"
            " Risk, Reasoning bullets, High-risk symbols, Recommendation test modules."
        )


def _parse_json(text: str) -> dict[str, Any]:
    body = text.strip()
    if not body:
        return {}
    try:
        parsed = json.loads(body)
        if isinstance(parsed, dict):
            return parsed
        return {}
    except json.JSONDecodeError:
        start = body.find("{")
        end = body.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        snippet = body[start : end + 1]
        try:
            parsed = json.loads(snippet)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return {}
    return {}


def _summarize_result(value: Any, max_chars: int = 1800) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False)
    except TypeError:
        text = str(value)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _invalid_action_warning(step: int, raw: str, parsed: dict[str, Any], max_chars: int = 240) -> str:
    raw_one_line = " ".join(raw.split())
    if len(raw_one_line) > max_chars:
        raw_one_line = raw_one_line[: max_chars - 3] + "..."
    parsed_keys = sorted(parsed.keys()) if isinstance(parsed, dict) else []
    if parsed_keys:
        return (
            f"Step {step}: planner returned invalid action payload "
            f"(keys={parsed_keys}); expected action=tool_call/final_answer. raw={raw_one_line}"
        )
    return (
        f"Step {step}: planner did not return valid JSON with an action field; "
        f"expected action=tool_call/final_answer. raw={raw_one_line}"
    )

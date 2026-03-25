from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
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
        steps_used = 0
        stop_reason = "max_steps_reached"
        llm_usage_totals: dict[str, int | None] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "calls": 0,
        }
        consecutive_quality_rejections = 0
        consecutive_must_tool_violations = 0
        last_quality_rejection: dict[str, Any] | None = None
        seen_tool_calls: set[str] = set()
        warnings: list[str] = []
        mode = self._resolve_mode(question)
        question_class = self._question_class(question=question, mode=mode)
        self._emit("mode_selected", {"mode": mode})

        for step in range(1, self.config.max_steps + 1):
            steps_used = step
            self._emit("step_start", {"step": step})
            prompt = self._planning_prompt(
                question,
                tools,
                evidence,
                step,
                mode,
                last_quality_rejection=last_quality_rejection,
                recent_tool_calls=_recent_tool_calls(evidence, limit=8),
            )
            self._emit(
                "planner_prompt_stats",
                {
                    "step": step,
                    "prompt_chars": len(prompt),
                    "prompt_tokens_est": _estimate_tokens_from_chars(len(prompt)),
                },
            )
            raw = self.llm.generate(
                prompt=prompt,
                temperature=self.config.temperature,
                timeout=self.config.timeout,
            )
            usage = _llm_usage_snapshot(self.llm)
            if usage is not None:
                self._emit("planner_llm_usage", {"step": step, "usage": usage})
                _accumulate_usage(llm_usage_totals, usage)
            action = _parse_json(raw)
            if not _is_valid_action(action):
                self._emit(
                    "planner_repair_start",
                    {"step": step, "raw": raw, "parsed": action},
                )
                repair_prompt = self._repair_action_prompt(raw=raw)
                self._emit(
                    "planner_repair_prompt_stats",
                    {
                        "step": step,
                        "prompt_chars": len(repair_prompt),
                        "prompt_tokens_est": _estimate_tokens_from_chars(len(repair_prompt)),
                    },
                )
                repaired_raw = self.llm.generate(
                    prompt=repair_prompt,
                    temperature=0.0,
                    timeout=self.config.timeout,
                )
                repair_usage = _llm_usage_snapshot(self.llm)
                if repair_usage is not None:
                    self._emit(
                        "planner_repair_llm_usage",
                        {"step": step, "usage": repair_usage},
                    )
                    _accumulate_usage(llm_usage_totals, repair_usage)
                repaired_action = _parse_json(repaired_raw)
                messages.append(
                    {
                        "step": step,
                        "repair_of": "planner_action",
                        "raw": repaired_raw,
                        "parsed": repaired_action,
                        "usage": repair_usage,
                    }
                )
                self._emit(
                    "planner_repair_done",
                    {"step": step, "raw": repaired_raw, "parsed": repaired_action},
                )
                if _is_valid_action(repaired_action):
                    raw = repaired_raw
                    action = repaired_action
            messages.append({"step": step, "raw": raw, "parsed": action})
            self._emit("planner_action", {"step": step, "action": action})

            if action.get("action") == "final_answer":
                confidence = float(action.get("confidence", 0.0) or 0.0)
                answer = str(action.get("answer", "")).strip()
                if last_quality_rejection is not None:
                    consecutive_must_tool_violations += 1
                    warnings.append(
                        f"Step {step}: final_answer rejected; must perform a tool_call after prior quality rejection."
                    )
                    self._emit(
                        "final_answer",
                        {
                            "step": step,
                            "confidence": confidence,
                            "accepted": False,
                            "reason": "must_tool_call_after_quality_rejection",
                        },
                    )
                    if consecutive_must_tool_violations >= 2:
                        warnings.append(
                            "Stopping early: planner repeatedly ignored required tool_call after quality rejection."
                        )
                        stop_reason = "stopped_early_quality_loop"
                        break
                    continue
                consecutive_must_tool_violations = 0
                quality_ok, quality_diag = _evaluate_final_answer_quality(
                    question_class=question_class,
                    answer=answer,
                    evidence=evidence,
                )
                if quality_diag:
                    self._emit(
                        "final_answer_quality",
                        {"step": step, "question_class": question_class, "quality": quality_diag},
                    )
                if answer and confidence >= self.config.confidence_threshold and quality_ok:
                    if mode == "pr_review":
                        answer = _format_pr_answer_sections(answer=answer, evidence=evidence)
                    self._emit(
                        "final_answer",
                        {"step": step, "confidence": confidence, "accepted": True},
                    )
                    self._emit("llm_usage_totals", {"usage": llm_usage_totals})
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
                        "llm_usage_totals": llm_usage_totals,
                        "warnings": warnings,
                    }
                if answer and confidence >= self.config.confidence_threshold:
                    consecutive_quality_rejections += 1
                    last_quality_rejection = quality_diag
                    warnings.append(
                        f"Step {step}: final_answer confidence {confidence:.2f} passed but quality gate failed ({json.dumps(quality_diag)}); continuing."
                    )
                    self._emit(
                        "final_answer",
                        {
                            "step": step,
                            "confidence": confidence,
                            "accepted": False,
                            "reason": "insufficient_specificity",
                        },
                    )
                    recovery = _quality_recovery_tool_call(
                        question_class=question_class,
                        evidence=evidence,
                    )
                    if recovery is not None:
                        recovery_tool, recovery_args = recovery
                        recovery_fingerprint = _tool_call_fingerprint(recovery_tool, recovery_args)
                        if recovery_fingerprint not in seen_tool_calls:
                            try:
                                self._emit(
                                    "quality_recovery_tool",
                                    {"step": step, "tool": recovery_tool, "args": recovery_args},
                                )
                                self._emit(
                                    "tool_execute_start",
                                    {"step": step, "tool": recovery_tool, "args": recovery_args},
                                )
                                recovery_execution = self.executor.execute(recovery_tool, recovery_args)
                                self._emit(
                                    "tool_execute_done",
                                    {
                                        "step": step,
                                        "tool": recovery_tool,
                                        "cost": recovery_execution.get("cost", 0),
                                    },
                                )
                                total_cost += int(recovery_execution.get("cost", 0))
                                recovery_summary = _summarize_result(
                                    recovery_execution.get("result"), max_chars=1800
                                )
                                evidence.append(
                                    {
                                        "step": step,
                                        "tool": recovery_tool,
                                        "args": recovery_args,
                                        "cost": recovery_execution.get("cost", 0),
                                        "summary": recovery_summary,
                                        "result": recovery_execution.get("result"),
                                    }
                                )
                                seen_tool_calls.add(recovery_fingerprint)
                                last_quality_rejection = None
                                consecutive_quality_rejections = 0
                                consecutive_must_tool_violations = 0
                                current_size = sum(len(item.get("summary", "")) for item in evidence)
                                while current_size > self.config.budget_chars and evidence:
                                    dropped = evidence.pop(0)
                                    current_size -= len(dropped.get("summary", ""))
                                    warnings.append(
                                        f"Dropped oldest evidence step {dropped.get('step')} to honor budget."
                                    )
                            except Exception as exc:  # pragma: no cover - runtime/error dependent
                                warnings.append(
                                    f"Step {step}: quality recovery tool execution failed for '{recovery_tool}': {exc}"
                                )
                    if consecutive_quality_rejections >= 2:
                        warnings.append(
                            "Stopping early: repeated final_answer quality rejections without new tool evidence."
                        )
                        stop_reason = "stopped_early_quality_loop"
                        break
                    continue
                consecutive_quality_rejections = 0
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
            fingerprint = _tool_call_fingerprint(tool_name, tool_args)
            if fingerprint in seen_tool_calls:
                warnings.append(
                    f"Step {step}: duplicate tool call skipped for {tool_name} with same args."
                )
                self._emit(
                    "planner_duplicate_tool_call",
                    {"step": step, "tool": tool_name, "args": tool_args},
                )
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
            if mode == "pr_review" and tool_name == "pr_diff_context":
                pr_llm_review = self._run_pr_llm_review(
                    pr_args=tool_args,
                    pr_summary=execution.get("result") or {},
                )
                if pr_llm_review is not None:
                    evidence.append(
                        {
                            "step": step,
                            "tool": "pr_llm_review",
                            "args": {
                                "base": tool_args.get("base"),
                                "head": tool_args.get("head"),
                                "repo_root": tool_args.get("repo_root"),
                            },
                            "cost": 6,
                            "summary": _summarize_result(pr_llm_review, max_chars=1800),
                            "result": pr_llm_review,
                        }
                    )
            seen_tool_calls.add(fingerprint)
            last_quality_rejection = None
            consecutive_quality_rejections = 0
            consecutive_must_tool_violations = 0

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
        self._emit("fallback_start", {"reason": stop_reason})
        self._emit(
            "fallback_prompt_stats",
            {
                "prompt_chars": len(fallback_prompt),
                "prompt_tokens_est": _estimate_tokens_from_chars(len(fallback_prompt)),
            },
        )
        fallback_answer = self.llm.generate(
            prompt=fallback_prompt,
            temperature=self.config.temperature,
            timeout=self.config.timeout,
        )
        if mode == "pr_review":
            fallback_answer = _format_pr_answer_sections(answer=fallback_answer, evidence=evidence)
        fallback_usage = _llm_usage_snapshot(self.llm)
        if fallback_usage is not None:
            self._emit("fallback_llm_usage", {"usage": fallback_usage})
            _accumulate_usage(llm_usage_totals, fallback_usage)
        self._emit("llm_usage_totals", {"usage": llm_usage_totals})
        self._emit("fallback_done", {})
        return {
            "status": stop_reason,
            "mode": mode,
            "question": question,
            "answer": fallback_answer.strip(),
            "confidence": None,
            "steps_used": steps_used,
            "total_cost": total_cost,
            "evidence": evidence,
            "messages": messages,
            "llm_usage_totals": llm_usage_totals,
            "warnings": warnings,
        }

    def _planning_prompt(
        self,
        question: str,
        tools: list[dict[str, Any]],
        evidence: list[dict[str, Any]],
        step: int,
        mode: str,
        *,
        last_quality_rejection: dict[str, Any] | None = None,
        recent_tool_calls: list[dict[str, Any]] | None = None,
    ) -> str:
        mode_directive = self._mode_directive(mode)
        question_class = self._question_class(question=question, mode=mode)
        question_class_guidance = self._question_class_guidance(question_class)
        repo_question_guidance = (
            "Repository-level question guidance:\n"
            "- If the user asks to explain a repository, architecture, modules, components, or requests a diagram/mermaid graph,\n"
            "  do not stop after only one shallow repo-level tool result unless the evidence already includes concrete module/component relationships.\n"
            "- Prefer gathering both:\n"
            "  1) repo-level structure via inspect_repo\n"
            "  2) at least one deeper follow-up such as module_context, symbol_context, module_dependencies_ranked, module_dependencies, or find_symbol_like\n"
            "- For architecture or diagram requests, prefer another tool_call over final_answer when evidence contains only README/top-level structure.\n"
            "- Use final_answer only when you can name the main modules/components and describe how they relate.\n"
        )
        json_contract_guidance = (
            "JSON contract reminders:\n"
            "- The action field must be exactly one of: tool_call, final_answer\n"
            "- Never put a tool name inside action\n"
            '- Valid tool call example: {"action":"tool_call","tool":"impact_context","args":{"symbol":"X","depth":2},"reason":"Need impact details"}\n'
            '- Invalid example: {"action":"impact_context","args":{"symbol":"X","depth":2},"reason":"..."}\n'
            '- Invalid example: {"action":"tool_call","tool":"symbol_context","args":"GraphBuilder"}\n'
            '- Valid final answer example: {"action":"final_answer","answer":"...","confidence":0.84}\n'
            '- Invalid example: {"action":"final_answer","answer":"..."}  # confidence missing\n'
            "- Return one JSON object only. No markdown fences. No commentary.\n"
        )
        evidence_sufficiency_guidance = (
            "Evidence sufficiency checklist for broad architecture/repo explain:\n"
            "- Include structure evidence (inspect_repo or directory_context).\n"
            "- Include relationship evidence (module_dependencies/module_context/directory_context/find_symbol_like).\n"
            "- If evidence only lists top-level directories/files without relationships, do one more tool_call.\n"
        )
        retry_guidance = (
            "Retry behavior guidance:\n"
            "- If final_answer was rejected for quality in prior step, next action must be tool_call.\n"
            "- Avoid repeating the exact same tool_call with identical args unless you have new evidence.\n"
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
            f"Question class:\n{question_class}\n"
            f"Question-class guidance:\n{question_class_guidance}\n\n"
            f"{json_contract_guidance}\n"
            f"{repo_question_guidance}\n"
            f"{evidence_sufficiency_guidance}\n"
            f"{retry_guidance}\n"
            f"Recent tool calls:\n{json.dumps(recent_tool_calls or [], indent=2)}\n\n"
            f"Last quality rejection feedback:\n{json.dumps(last_quality_rejection or {}, indent=2)}\n\n"
            f"Step: {step}\n"
            f"Question:\n{question}\n\n"
            f"Available tools:\n{json.dumps(tools, indent=2)}\n\n"
            f"Evidence so far:\n{json.dumps(_compact_evidence(evidence), indent=2)}\n"
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
                f"Evidence:\n{json.dumps(_compact_evidence(evidence), indent=2)}\n\n"
                f"Warnings:\n{json.dumps(warnings, indent=2)}\n"
            )
        return (
            "You are a code intelligence assistant. Produce the best possible answer "
            "from available evidence. If uncertain, clearly state uncertainty.\n\n"
            f"Question:\n{question}\n\n"
            f"Evidence:\n{json.dumps(_compact_evidence(evidence), indent=2)}\n\n"
            f"Warnings:\n{json.dumps(warnings, indent=2)}\n\n"
            "Final answer:"
        )

    def _run_pr_llm_review(
        self,
        *,
        pr_args: dict[str, Any],
        pr_summary: dict[str, Any],
    ) -> dict[str, Any] | None:
        base = str(pr_args.get("base", self.config.pr_base))
        head = str(pr_args.get("head", self.config.pr_head))
        repo_root = str(pr_args.get("repo_root", self.config.pr_repo_root))
        depth = int(pr_args.get("depth", 3))

        try:
            full_ctx = self.executor.execute(
                "pr_diff_context",
                {
                    "base": base,
                    "head": head,
                    "repo_root": repo_root,
                    "depth": depth,
                    "format": "full",
                    "top_symbol_contexts": 8,
                    "top_module_contexts": 4,
                },
            ).get("result", {})
        except Exception as exc:  # pragma: no cover - runtime/error dependent
            return {
                "status": "error",
                "stage": "full_context",
                "error": str(exc),
            }

        changed_hunks = _git_changed_hunks(repo_root=repo_root, base=base, head=head, max_chars=16000)
        stage1_prompt = (
            "You are reviewing a PR for obvious defects.\n"
            "Given changed hunks and touched symbols, return strict JSON only:\n"
            '{"local_findings":[{"severity":"low|medium|high","kind":"obvious_bug|type_mismatch|logic_risk|style","file":"...","line":0,"why":"...",'
            '"evidence_class":"confirmed_by_diff|inferred_from_context|generic_risk"}],'
            '"propagation_candidates":[{"symbol_id":"...","reason":"signature_or_contract_change|behavior_change|serialization_change"}],'
            '"notes":"..."}\n\n'
            "Classification rules:\n"
            "- confirmed_by_diff: directly visible in changed lines (signature/type/default/return mismatch).\n"
            "- inferred_from_context: inferred from nearby code or usage context.\n"
            "- generic_risk: general caution not tied to concrete diff evidence.\n\n"
            f"Changed hunks:\n{changed_hunks}\n\n"
            f"Touched symbols:\n{json.dumps(full_ctx.get('touched_symbols', [])[:40], indent=2)}\n"
        )
        stage1_raw = self.llm.generate(prompt=stage1_prompt, temperature=0.0, timeout=self.config.timeout)
        stage1 = _parse_json(stage1_raw)

        symbol_impacts = full_ctx.get("symbol_impacts", [])
        top_symbols = []
        if isinstance(symbol_impacts, list):
            top_symbols = symbol_impacts[:8]
        changed_function_excerpts: list[dict[str, Any]] = []
        for row in top_symbols[:6]:
            symbol = row.get("symbol", {}) if isinstance(row, dict) else {}
            symbol_id = symbol.get("symbol_id") if isinstance(symbol, dict) else None
            if not isinstance(symbol_id, str):
                continue
            try:
                excerpt = self.executor.execute(
                    "get_source_excerpt", {"symbol": symbol_id, "max_lines": 60}
                ).get("result")
            except Exception:
                excerpt = None
            changed_function_excerpts.append(
                {
                    "symbol_id": symbol_id,
                    "risk_score": row.get("risk_score"),
                    "summary": row.get("summary"),
                    "source_excerpt": excerpt,
                }
            )

        stage2_prompt = (
            "You are reviewing changed functions for concrete defects and propagation risk.\n"
            "Return strict JSON only:\n"
            '{"function_findings":[{"symbol_id":"...","severity":"low|medium|high","defect":"...","confidence":"low|medium|high",'
            '"evidence_class":"confirmed_by_diff|inferred_from_context|generic_risk"}],'
            '"propagation_needed":[{"symbol_id":"...","why":"..."}]}\n\n'
            "Classification rules:\n"
            "- confirmed_by_diff: defect is explicit in changed function excerpt.\n"
            "- inferred_from_context: likely defect from interactions/call patterns.\n"
            "- generic_risk: generic warning without concrete evidence.\n\n"
            f"Changed function excerpts:\n{json.dumps(changed_function_excerpts, indent=2)}\n"
        )
        stage2_raw = self.llm.generate(prompt=stage2_prompt, temperature=0.0, timeout=self.config.timeout)
        stage2 = _parse_json(stage2_raw)

        propagation_candidates = []
        for source in (stage1.get("propagation_candidates"), stage2.get("propagation_needed")):
            if isinstance(source, list):
                for item in source:
                    if isinstance(item, dict) and isinstance(item.get("symbol_id"), str):
                        propagation_candidates.append(item)
        seen_symbols: set[str] = set()
        propagation_symbols: list[str] = []
        for item in propagation_candidates:
            symbol_id = str(item.get("symbol_id"))
            if symbol_id in seen_symbols:
                continue
            seen_symbols.add(symbol_id)
            propagation_symbols.append(symbol_id)
            if len(propagation_symbols) >= 3:
                break

        stage3_context: list[dict[str, Any]] = []
        for symbol_id in propagation_symbols:
            callers = self.executor.execute(
                "dependents", {"symbol": symbol_id, "kind": "calls"}
            ).get("result", [])[:5]
            callees = self.executor.execute(
                "dependencies", {"symbol": symbol_id, "kind": "calls"}
            ).get("result", [])[:5]

            caller_excerpts = []
            for caller in callers:
                caller_id = caller.get("symbol_id") if isinstance(caller, dict) else None
                if not isinstance(caller_id, str):
                    continue
                excerpt = self.executor.execute(
                    "get_source_excerpt", {"symbol": caller_id, "max_lines": 30}
                ).get("result")
                caller_excerpts.append({"symbol_id": caller_id, "source_excerpt": excerpt})

            callee_excerpts = []
            for callee in callees:
                callee_id = callee.get("symbol_id") if isinstance(callee, dict) else None
                if not isinstance(callee_id, str):
                    continue
                excerpt = self.executor.execute(
                    "get_source_excerpt", {"symbol": callee_id, "max_lines": 30}
                ).get("result")
                callee_excerpts.append({"symbol_id": callee_id, "source_excerpt": excerpt})

            stage3_context.append(
                {
                    "symbol_id": symbol_id,
                    "callers": caller_excerpts,
                    "callees": callee_excerpts,
                }
            )

        stage3 = {}
        if stage3_context:
            stage3_prompt = (
                "You are doing integration breakage review.\n"
                "Return strict JSON only:\n"
                '{"integration_findings":[{"symbol_id":"...","severity":"low|medium|high","kind":"signature_mismatch|behavioral_break|compat_risk","evidence":"...","confidence":"low|medium|high",'
                '"evidence_class":"confirmed_by_diff|inferred_from_context|generic_risk"}]}\n\n'
                "Classification rules:\n"
                "- confirmed_by_diff: caller/callee snippets show direct mismatch or breakage.\n"
                "- inferred_from_context: likely integration defect inferred from context.\n"
                "- generic_risk: broad integration caution.\n\n"
                f"Propagation contexts:\n{json.dumps(stage3_context, indent=2)}\n"
            )
            stage3_raw = self.llm.generate(prompt=stage3_prompt, temperature=0.0, timeout=self.config.timeout)
            stage3 = _parse_json(stage3_raw)

        return {
            "status": "ok",
            "base": base,
            "head": head,
            "repo_root": str(Path(repo_root).resolve()),
            "summary": pr_summary.get("summary", {}),
            "stage1": stage1,
            "stage2": stage2,
            "stage3": stage3,
        }

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

    @staticmethod
    def _question_class(question: str, mode: str) -> str:
        if mode == "pr_review":
            return "pr_or_diff_review"
        lowered = question.lower()
        if any(
            token in lowered
            for token in (
                "pull request",
                "pr ",
                "pr:",
                "diff",
                "safe to merge",
                "risk",
            )
        ):
            return "pr_or_diff_review"
        if any(
            token in lowered
            for token in (
                "architecture",
                "repo",
                "repository",
                "module",
                "diagram",
                "mermaid",
                "explain this repo",
            )
        ):
            return "broad_architecture"
        return "symbol_focused"

    @staticmethod
    def _question_class_guidance(question_class: str) -> str:
        if question_class == "pr_or_diff_review":
            return (
                "- Prefer pr_diff_context first.\n"
                "- Use pr_llm_review evidence (if available) to identify obvious defects and integration mismatches.\n"
                "- Then optionally use symbol_context or stack_trace for top risky symbols.\n"
                "- In final_answer, separate: confirmed defects, likely defects, and impact-only risks.\n"
                "- Use evidence_class tags from pr_llm_review: confirmed_by_diff, inferred_from_context, generic_risk.\n"
                "- Prioritize confirmed_by_diff over inferred_from_context; keep generic_risk as low-priority caveats.\n"
                "- Use final_answer after diff impact and risk evidence is present."
            )
        if question_class == "broad_architecture":
            return (
                "- Prefer inspect_repo early.\n"
                "- Then use at least one deeper relationship tool: module_or_directory_context, directory_context(s), module_context(s), "
                "module_dependencies_ranked, module_dependencies, or find_symbol_like.\n"
                "- Prefer fewer deeper calls over many shallow module calls.\n"
                "- Avoid large module_contexts lists by default; start with 1-3 high-signal modules/directories.\n"
                "- Only return final_answer when you can name concrete modules/directories and at least 1-2 explicit dependency relationships from evidence.\n"
                "- For final_answer, use markdown sections with these exact headings:\n"
                "  Flow\n"
                "  Key modules\n"
                "  Dependencies\n"
                "  Uncertainty\n"
                "- For diagram/mermaid requests, avoid final_answer if only top-level structure is known."
            )
        return (
            "- Prefer symbol discovery and targeted context (symbol_lookup/find_symbol_like/symbol_context).\n"
            "- Use call_chain/impact when question asks about breakage or propagation."
        )

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
            "Do not add sections titled 'Primary changed symbols' or 'Container-only touched symbols (downweighted)'; they are appended automatically.\n"
            "When you have enough evidence, return final_answer in markdown PR reviewer style with:"
            " Risk, Reasoning bullets, High-risk symbols, Recommendation test modules."
        )

    @staticmethod
    def _repair_action_prompt(raw: str) -> str:
        return (
            "Repair the following planner output into one strict JSON object only.\n"
            "Allowed action values: tool_call or final_answer.\n"
            "Rules:\n"
            '- For tool calls, output: {"action":"tool_call","tool":"<tool_name>","args":{...},"reason":"..."}\n'
            '- For final answer, output: {"action":"final_answer","answer":"...","confidence":0.0}\n'
            "- args must be a JSON object.\n"
            "- confidence is required for final_answer.\n"
            "- Do not wrap in markdown fences.\n\n"
            "Planner output to repair:\n"
            f"{raw}"
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


def _is_valid_action(action: dict[str, Any]) -> bool:
    if not isinstance(action, dict):
        return False
    action_name = action.get("action")
    if action_name == "tool_call":
        tool = action.get("tool")
        args = action.get("args")
        return isinstance(tool, str) and bool(tool.strip()) and isinstance(args, dict)
    if action_name == "final_answer":
        answer = action.get("answer")
        confidence = action.get("confidence")
        return isinstance(answer, str) and "confidence" in action and confidence is not None
    return False


def _estimate_tokens_from_chars(chars: int) -> int:
    if chars <= 0:
        return 0
    # Rough heuristic for English/code mixed text: ~4 chars/token.
    return (chars + 3) // 4


def _llm_usage_snapshot(llm: Any) -> dict[str, Any] | None:
    usage = getattr(llm, "last_usage", None)
    if isinstance(usage, dict):
        return usage
    return None


def _accumulate_usage(totals: dict[str, int | None], usage: dict[str, Any]) -> None:
    totals["calls"] = int(totals.get("calls") or 0) + 1
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(key)
        if value is None:
            continue
        totals[key] = int(totals.get(key) or 0) + int(value)


def _evaluate_final_answer_quality(
    *,
    question_class: str,
    answer: str,
    evidence: list[dict[str, Any]],
) -> tuple[bool, dict[str, Any]]:
    if not answer.strip():
        return False, {"reason": "empty_answer"}
    if question_class != "broad_architecture":
        return True, {}

    terms = _architecture_terms_from_evidence(evidence)
    if not terms:
        return True, {"reason": "no_terms_from_evidence"}

    required_headers = ("Flow", "Key modules", "Dependencies", "Uncertainty")
    headers_found = {
        header: bool(
            re.search(
                rf"^\s*(?:#+\s*)?{re.escape(header)}\s*:?\s*$",
                answer,
                flags=re.IGNORECASE | re.MULTILINE,
            )
        )
        for header in required_headers
    }
    header_hits = sum(1 for value in headers_found.values() if value)

    hits = 0
    lowered_answer = answer.lower()
    for term in terms:
        if term.lower() in lowered_answer:
            hits += 1
    dep_relations = _count_dependency_relations(answer)
    min_term_hits = 4
    min_dep_relations = 2
    min_header_hits = len(required_headers)

    ok = (
        header_hits >= min_header_hits
        and hits >= min_term_hits
        and dep_relations >= min_dep_relations
    )
    return ok, {
        "required_headers": list(required_headers),
        "headers_found": headers_found,
        "header_hits": header_hits,
        "required_header_hits": min_header_hits,
        "module_term_hits": hits,
        "required_module_term_hits": min_term_hits,
        "dependency_relations": dep_relations,
        "required_dependency_relations": min_dep_relations,
        "evidence_term_count": len(terms),
    }


def _architecture_terms_from_evidence(evidence: list[dict[str, Any]]) -> set[str]:
    terms: set[str] = set()
    for item in evidence:
        tool = str(item.get("tool") or "")
        result = item.get("result") or {}
        if tool == "inspect_repo" and isinstance(result, dict):
            for row in result.get("top_level_entries", []) or []:
                name = row.get("name") if isinstance(row, dict) else None
                if isinstance(name, str):
                    terms.add(name)
            for row in result.get("top_modules", []) or []:
                name = row.get("module") if isinstance(row, dict) else None
                if isinstance(name, str):
                    terms.add(name)
        elif tool in {"directory_context", "module_or_directory_context"} and isinstance(result, dict):
            focus = result.get("focus") or {}
            if isinstance(focus, dict):
                for key in ("directory", "module", "name"):
                    value = focus.get(key)
                    if isinstance(value, str):
                        terms.add(value)
        elif tool == "directory_contexts" and isinstance(result, dict):
            facts = result.get("facts") or {}
            contexts = facts.get("contexts") if isinstance(facts, dict) else None
            if isinstance(contexts, list):
                for row in contexts:
                    directory = row.get("directory") if isinstance(row, dict) else None
                    if isinstance(directory, str):
                        terms.add(directory)
        elif tool in {"module_context", "module_contexts"} and isinstance(result, dict):
            focus = result.get("focus") or {}
            if isinstance(focus, dict):
                module = focus.get("module")
                if isinstance(module, str):
                    terms.add(module)
            facts = result.get("facts") or {}
            contexts = facts.get("contexts") if isinstance(facts, dict) else None
            if isinstance(contexts, list):
                for row in contexts:
                    module = row.get("module") if isinstance(row, dict) else None
                    if isinstance(module, str):
                        terms.add(module)
        elif tool in {"module_dependencies", "module_dependents", "module_dependencies_ranked"} and isinstance(result, list):
            for edge in result[:200]:
                if not isinstance(edge, dict):
                    continue
                source = edge.get("source_module")
                target = edge.get("target_module")
                if isinstance(source, str):
                    terms.add(source)
                if isinstance(target, str):
                    terms.add(target)
    return {term for term in terms if isinstance(term, str) and term.strip()}


def _count_dependency_relations(answer: str) -> int:
    patterns = (
        r"\bdepends on\b",
        r"\brequires\b",
        r"\bimports\b",
        r"\buses\b",
        r"->",
    )
    total = 0
    for pattern in patterns:
        total += len(re.findall(pattern, answer, flags=re.IGNORECASE))
    return total


def _tool_call_fingerprint(tool_name: str, tool_args: dict[str, Any]) -> str:
    try:
        normalized_args = json.dumps(tool_args, sort_keys=True, ensure_ascii=False)
    except TypeError:
        normalized_args = str(tool_args)
    return f"{tool_name}:{normalized_args}"


def _recent_tool_calls(evidence: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in evidence[-limit:]:
        rows.append(
            {
                "step": item.get("step"),
                "tool": item.get("tool"),
                "args": item.get("args"),
            }
        )
    return rows


def _git_changed_hunks(
    *,
    repo_root: str,
    base: str,
    head: str,
    max_chars: int = 16000,
) -> str:
    cmd = [
        "git",
        "-C",
        str(Path(repo_root).resolve()),
        "diff",
        "--unified=2",
        "--no-color",
        f"{base}...{head}",
    ]
    try:
        text = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    except Exception as exc:  # pragma: no cover - runtime/error dependent
        return f"<diff_unavailable: {exc}>"
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _quality_recovery_tool_call(
    *,
    question_class: str,
    evidence: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]] | None:
    if question_class != "broad_architecture":
        return None

    candidate = _primary_architecture_target(evidence)
    if candidate:
        return (
            "module_dependencies_ranked",
            {"module": candidate, "max_edges": 10, "include_ancillary": False},
        )

    return ("inspect_repo", {"max_entries": 10, "readme_max_lines": 10, "top_modules": 5})


def _primary_architecture_target(evidence: list[dict[str, Any]]) -> str | None:
    for item in reversed(evidence):
        tool = str(item.get("tool") or "")
        args = item.get("args") or {}
        result = item.get("result") or {}
        if tool in {"module_dependencies_ranked", "module_dependencies", "module_context"}:
            module = args.get("module") if isinstance(args, dict) else None
            if isinstance(module, str) and module.strip():
                return module
        if tool in {"module_or_directory_context"}:
            name = args.get("name") if isinstance(args, dict) else None
            if isinstance(name, str) and name.strip():
                return name
        if tool == "inspect_repo" and isinstance(result, dict):
            for row in result.get("top_level_modules", []) or []:
                module = row.get("module") if isinstance(row, dict) else None
                if isinstance(module, str) and module.strip():
                    return module
    return None


def _compact_evidence(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for item in evidence:
        compact.append(
            {
                "step": item.get("step"),
                "tool": item.get("tool"),
                "args": item.get("args"),
                "cost": item.get("cost"),
                "summary": item.get("summary"),
            }
        )
    return compact


def _format_pr_answer_sections(answer: str, evidence: list[dict[str, Any]]) -> str:
    text = (answer or "").strip()
    primary, container = _pr_symbol_partitions_from_evidence(evidence)
    if not primary and not container:
        return text

    lowered = text.lower()
    lines: list[str] = [text] if text else []

    if "primary changed symbols" not in lowered:
        lines.append("")
        lines.append("Primary changed symbols")
        for row in primary[:8]:
            lines.append(f"- `{row}`")

    if "container-only touched symbols" not in lowered:
        lines.append("")
        lines.append("Container-only touched symbols (downweighted)")
        if container:
            for row in container[:8]:
                lines.append(f"- `{row}`")
        else:
            lines.append("- `(none)`")

    return "\n".join(lines).strip()


def _pr_symbol_partitions_from_evidence(evidence: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    for item in reversed(evidence):
        if str(item.get("tool") or "") != "pr_diff_context":
            continue
        result = item.get("result")
        if not isinstance(result, dict):
            continue
        primary_rows = result.get("touched_symbols")
        container_rows = result.get("container_symbols")
        primary = _symbol_rows_to_labels(primary_rows)
        container = _symbol_rows_to_labels(container_rows)
        return primary, container
    return [], []


def _symbol_rows_to_labels(rows: Any) -> list[str]:
    if not isinstance(rows, list):
        return []
    labels: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = row.get("name")
        file_path = row.get("file")
        line = row.get("start_line")
        if isinstance(name, str) and isinstance(file_path, str):
            if isinstance(line, int):
                labels.append(f"{name} ({file_path}:{line})")
            else:
                labels.append(f"{name} ({file_path})")
    # preserve order, de-dup
    deduped: list[str] = []
    seen: set[str] = set()
    for label in labels:
        if label in seen:
            continue
        seen.add(label)
        deduped.append(label)
    return deduped

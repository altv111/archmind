from __future__ import annotations

import re
from typing import Any

from query.query_engine import QueryEngine


class QueryPlanner:
    def __init__(self, llm: Any = None):
        self.llm = llm

    def plan(self, question: str) -> dict:
        intent = self.detect_intent(question)

        if intent == "impact_analysis":
            return self.plan_impact(question)

        if intent == "symbol_explanation":
            return self.plan_symbol_explain(question)

        if intent == "architecture_analysis":
            return self.plan_architecture(question)

        raise ValueError("Unknown query type")

    def detect_intent(self, question: str) -> str:
        if self.llm and hasattr(self.llm, "detect_intent"):
            intent = self.llm.detect_intent(question)
            if intent in {"impact_analysis", "symbol_explanation", "architecture_analysis"}:
                return intent

        lowered = question.lower()
        if any(token in lowered for token in ("impact", "break", "affected", "blast radius")):
            return "impact_analysis"
        if any(token in lowered for token in ("architecture", "module", "layer", "dependency between modules")):
            return "architecture_analysis"
        return "symbol_explanation"

    def extract_symbol(self, question: str) -> str:
        if self.llm and hasattr(self.llm, "extract_symbol"):
            symbol = self.llm.extract_symbol(question)
            if symbol:
                return symbol

        candidates = re.findall(r"\b[A-Z][A-Za-z0-9_]*\b", question)
        if candidates:
            return candidates[-1]

        words = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", question)
        if words:
            return words[-1]
        raise ValueError("Could not extract symbol from question")

    def extract_module(self, question: str) -> str:
        if self.llm and hasattr(self.llm, "extract_module"):
            module = self.llm.extract_module(question)
            if module:
                return module

        module_match = re.search(r"\bmodule\s+([a-zA-Z_][a-zA-Z0-9_]*)\b", question, flags=re.IGNORECASE)
        if module_match:
            return module_match.group(1)

        candidates = re.findall(r"\b[a-z_][a-z0-9_]*\b", question)
        if candidates:
            return candidates[-1]
        raise ValueError("Could not extract module from question")

    def plan_impact(self, question: str) -> dict:
        symbol = self.extract_symbol(question)

        return {
            "intent": "impact_analysis",
            "focus_symbol": symbol,
            "plan": [
                {"tool": "symbol_lookup", "symbol": symbol},
                {"tool": "callers", "symbol": symbol, "depth": 3},
                {"tool": "callees", "symbol": symbol, "depth": 1},
                {"tool": "module_dependencies", "symbol": symbol},
                {"tool": "containment", "symbol": symbol},
            ],
        }

    def plan_architecture(self, question: str) -> dict:
        module = self.extract_module(question)

        return {
            "intent": "architecture_analysis",
            "focus_module": module,
            "plan": [
                {"tool": "module_dependencies", "module": module},
                {"tool": "module_dependents", "module": module},
            ],
        }

    def plan_symbol_explain(self, question: str) -> dict:
        symbol = self.extract_symbol(question)

        return {
            "intent": "symbol_explanation",
            "focus_symbol": symbol,
            "plan": [
                {"tool": "symbol_lookup", "symbol": symbol},
                {"tool": "containment", "symbol": symbol},
                {"tool": "callees", "symbol": symbol, "depth": 1},
                {"tool": "callers", "symbol": symbol, "depth": 1},
            ],
        }


class QueryExecutor:
    def __init__(self, query_engine: QueryEngine):
        self.engine = query_engine

    def execute(self, plan: dict) -> dict:
        results: dict[str, Any] = {}

        for step in plan.get("plan", []):
            tool = step["tool"]

            if tool == "symbol_lookup":
                results["symbol"] = self.engine.resolve_symbols(step["symbol"])
            elif tool == "callers":
                symbol = step["symbol"]
                depth = int(step.get("depth", 1))
                results["callers"] = self._walk_callers(symbol, depth)
            elif tool == "callees":
                symbol = step["symbol"]
                depth = int(step.get("depth", 1))
                results["callees"] = self._walk_callees(symbol, depth)
            elif tool == "module_dependencies":
                module = step.get("module")
                if module is None and step.get("symbol"):
                    module = self.engine.module_of_symbol(step["symbol"])
                if module:
                    results["module_dependencies"] = self.engine.module_dependencies_of(module)
                    results["focus_module"] = module
                else:
                    results["module_dependencies"] = []
            elif tool == "module_dependents":
                module = step.get("module")
                if module:
                    results["module_dependents"] = self.engine.module_dependents_of(module)
                else:
                    results["module_dependents"] = []
            elif tool == "containment":
                results["containment"] = self.engine.children_of(step["symbol"])
            else:
                raise ValueError(f"Unsupported tool in query plan: {tool}")

        return results

    def _walk_callers(self, symbol: str, depth: int) -> list:
        resolved = self.engine.resolve_symbol(symbol)
        if resolved is None:
            return []
        frontier = [resolved.symbol_id]
        seen: set[str] = {resolved.symbol_id}
        out: list = []
        for _ in range(depth):
            next_frontier: list[str] = []
            for symbol_id in frontier:
                for caller in self.engine.callers_of(symbol_id):
                    if caller.symbol_id in seen:
                        continue
                    seen.add(caller.symbol_id)
                    out.append(caller)
                    next_frontier.append(caller.symbol_id)
            if not next_frontier:
                break
            frontier = next_frontier
        return out

    def _walk_callees(self, symbol: str, depth: int) -> list:
        resolved = self.engine.resolve_symbol(symbol)
        if resolved is None:
            return []
        frontier = [resolved.symbol_id]
        seen: set[str] = {resolved.symbol_id}
        out: list = []
        for _ in range(depth):
            next_frontier: list[str] = []
            for symbol_id in frontier:
                for callee in self.engine.callees_of(symbol_id):
                    if callee.symbol_id in seen:
                        continue
                    seen.add(callee.symbol_id)
                    out.append(callee)
                    next_frontier.append(callee.symbol_id)
            if not next_frontier:
                break
            frontier = next_frontier
        return out


class QueryOrchestrator:
    def __init__(
        self,
        query_engine: QueryEngine,
        context_builder,
        llm: Any = None,
        planner: QueryPlanner | None = None,
        executor: QueryExecutor | None = None,
    ) -> None:
        self.query_engine = query_engine
        self.context_builder = context_builder
        self.llm = llm
        self.planner = planner or QueryPlanner(llm=llm)
        self.executor = executor or QueryExecutor(query_engine)

    def run(self, question: str) -> dict:
        plan = self.planner.plan(question)
        results = self.executor.execute(plan)
        context = self._build_context(plan)

        payload = {
            "question": question,
            "plan": plan,
            "results": results,
            "context": context,
        }

        if self.llm and hasattr(self.llm, "answer"):
            payload["llm_answer"] = self.llm.answer(question, context)
        return payload

    def _build_context(self, plan: dict) -> dict:
        intent = plan.get("intent")
        if intent == "impact_analysis":
            symbol = plan["focus_symbol"]
            return {
                "impact_context": self.context_builder.impact_context(symbol, depth=3),
                "call_chain": self.context_builder.call_chain(symbol, depth=2, direction="both"),
            }
        if intent == "symbol_explanation":
            symbol = plan["focus_symbol"]
            return {
                "symbol_context": self.context_builder.symbol_context(symbol),
                "call_chain": self.context_builder.call_chain(symbol, depth=1, direction="both"),
            }
        if intent == "architecture_analysis":
            module = plan["focus_module"]
            return {
                "module_context": self.context_builder.module_context(module),
            }
        return {}

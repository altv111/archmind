from graph.code_graph import CodeGraph
from ingestion.symbol_extractor import Symbol


class QueryEngine:

    def __init__(self, graph: CodeGraph):
        self.graph = graph

    def who_calls(self, symbol_query: str):
        incoming = self.graph.incoming_dependencies_of(symbol_query)
        return [
            self.graph.symbol_lookup(dep.source_symbol)[0]
            for dep in incoming
            if dep.kind == "calls"
        ]

    def what_does(self, symbol_query: str):
        outgoing = self.graph.outgoing_dependencies_of(symbol_query)
        return [
            self.graph.symbol_lookup(dep.target_symbol)[0]
            for dep in outgoing
            if dep.kind == "calls"
        ]

    def impact_of(self, symbol_query: str):
        visited = set()
        stack = [symbol_query]
        impacted = []

        while stack:
            current = stack.pop()
            for dep in self.graph.incoming_dependencies_of(current):
                if dep.kind != "calls":
                    continue
                caller = dep.source_symbol
                if caller in visited:
                    continue
                visited.add(caller)
                impacted.append(caller)
                stack.append(caller)

        return impacted

    def module_dependencies(self):
        return getattr(self.graph, "module_edges", [])
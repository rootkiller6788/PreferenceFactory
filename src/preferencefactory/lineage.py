"""偏好数据的血缘 DAG。

**为什么偏好数据也需要血缘**：一条 `chosen > rejected` 记录进了训练集之后，它的
来源就消失了 —— 谁生成的候选、哪个裁判判的、当时这条记录的一致率是多少，全都
看不见。等下游发现「模型在某个题型上特别怪」时，没有任何东西能把现象指回原因。

血缘记的是**产生关系**（谁产出了谁），不是流程顺序。节点是对象（prompt / 候选 /
裁判 / 记录），边是关系（生成 / 判定 / 聚合 / 过滤）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

__all__ = ["Lineage", "Node", "Edge", "DOT_COLORS"]

DOT_COLORS: dict[str, str] = {
    "prompt": "#cfe3ff",
    "candidate": "#d9f2d9",
    "judge": "#ffe0cc",
    "record": "#e8dcff",
    "dataset": "#ffd9d9",
    "root": "#f0f0f0",
}


@dataclass(frozen=True)
class Node:
    id: str
    kind: str
    label: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "label": self.label or self.id, "meta": dict(self.meta)}


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    kind: str
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"src": self.src, "dst": self.dst, "kind": self.kind, "label": self.label}


class Lineage:
    """有向无环图。**加边时拒绝成环** —— 成环的血缘图是没法解释的
    （「这条记录来自它自己」），而且它会让 DOT 渲染器静默丢边。
    """

    def __init__(self) -> None:
        self._nodes: dict[str, Node] = {}
        self._edges: list[Edge] = []
        self._out: dict[str, set[str]] = {}

    # ---- 写

    def add_node(self, node: Node) -> None:
        if node.id in self._nodes:
            raise ValueError(f"血缘节点重复：{node.id}")
        self._nodes[node.id] = node
        self._out.setdefault(node.id, set())

    def add_edge(self, edge: Edge) -> None:
        if edge.src not in self._nodes:
            raise ValueError(f"边的起点不存在：{edge.src}")
        if edge.dst not in self._nodes:
            raise ValueError(f"边的终点不存在：{edge.dst}")
        if edge.dst == edge.src:
            raise ValueError(f"自环：{edge.src}")
        if self._reaches(edge.dst, edge.src):
            raise ValueError(f"加这条边会成环：{edge.src} -> {edge.dst}")
        self._edges.append(edge)
        self._out[edge.src].add(edge.dst)

    def _reaches(self, src: str, dst: str, _seen: set[str] | None = None) -> bool:
        seen = _seen if _seen is not None else set()
        if src in seen:
            return False
        seen.add(src)
        for nxt in self._out.get(src, ()):  # type: ignore[arg-type]
            if nxt == dst or self._reaches(nxt, dst, seen):
                return True
        return False

    # ---- 读

    @property
    def nodes(self) -> tuple[Node, ...]:
        return tuple(self._nodes.values())

    @property
    def edges(self) -> tuple[Edge, ...]:
        return tuple(self._edges)

    def node(self, nid: str) -> Node | None:
        return self._nodes.get(nid)

    def parents_of(self, nid: str) -> tuple[str, ...]:
        return tuple(e.src for e in self._edges if e.dst == nid)

    def children_of(self, nid: str) -> tuple[str, ...]:
        return tuple(e.dst for e in self._edges if e.src == nid)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Lineage":
        g = cls()
        for n in raw.get("nodes", ()):
            g.add_node(Node(str(n["id"]), str(n["kind"]), str(n.get("label", "")), dict(n.get("meta") or {})))
        for e in raw.get("edges", ()):
            g.add_edge(Edge(str(e["src"]), str(e["dst"]), str(e["kind"]), str(e.get("label", ""))))
        return g

    # ---- DOT

    def to_dot(self) -> str:
        lines = ["digraph lineage {", '  rankdir="LR";', '  node [shape=box, style="rounded,filled", fontname="Helvetica"];']
        for n in self.nodes:
            color = DOT_COLORS.get(n.kind, "#ffffff")
            label = (n.label or n.id).replace('"', "'")
            lines.append(f'  "{n.id}" [label="{label}", fillcolor="{color}"];')
        for e in self.edges:
            label = e.label or e.kind
            lines.append(f'  "{e.src}" -> "{e.dst}" [label="{label}"];')
        lines.append("}")
        return "\n".join(lines) + "\n"

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
        # 邻接索引：parents_of / children_of 原本每次扫全表（O(E)），
        # 建索引后是 O(度数)。用 list 而不是 set，保留边的插入顺序与重复边。
        self._in_edges: dict[str, list[str]] = {}
        self._out_edges: dict[str, list[str]] = {}

    # ---- 写

    def add_node(self, node: Node) -> None:
        if node.id in self._nodes:
            raise ValueError(f"血缘节点重复：{node.id}")
        self._nodes[node.id] = node
        self._out.setdefault(node.id, set())

    def add_edge(self, edge: Edge) -> None:
        self._append_edge(edge, check_cycle=True)

    def _append_edge(self, edge: Edge, *, check_cycle: bool) -> None:
        if edge.src not in self._nodes:
            raise ValueError(f"边的起点不存在：{edge.src}")
        if edge.dst not in self._nodes:
            raise ValueError(f"边的终点不存在：{edge.dst}")
        if edge.dst == edge.src:
            raise ValueError(f"自环：{edge.src}")
        if check_cycle and self._reaches(edge.dst, edge.src):
            raise ValueError(f"加这条边会成环：{edge.src} -> {edge.dst}")
        self._edges.append(edge)
        self._out[edge.src].add(edge.dst)
        self._out_edges.setdefault(edge.src, []).append(edge.dst)
        self._in_edges.setdefault(edge.dst, []).append(edge.src)

    def _validate_acyclic(self) -> None:
        """Kahn 拓扑排序，一次判定整张图是否有环。

        `from_dict` 读入的边是**已经构建好的**，逐条做 DFS 环检测是 O(E·(V+E))；
        全部读完再做一次拓扑排序是 O(V+E)，且错误语义不变（仍然抛 ValueError）。
        """
        indeg = {nid: 0 for nid in self._nodes}
        for e in self._edges:
            indeg[e.dst] += 1
        queue = [nid for nid, d in indeg.items() if d == 0]
        seen = 0
        while queue:
            cur = queue.pop()
            seen += 1
            # 用带重复边的索引：indeg 是按 _edges 的每条边记的，必须逐条减。
            for nxt in self._out_edges.get(cur, ()):
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    queue.append(nxt)
        if seen != len(self._nodes):
            raise ValueError("血缘图成环：from_dict 校验失败")

    def _reaches(self, src: str, dst: str) -> bool:
        """从 src 出发能否到达 dst。**显式栈而不是递归** —— 血缘链一旦超过
        Python 的递归上限（默认 1000），递归版本会抛 RecursionError 而不是给出
        「会成环」这个本该给出的结论。"""
        seen: set[str] = set()
        stack = [src]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            for nxt in self._out.get(cur, ()):  # type: ignore[arg-type]
                if nxt == dst:
                    return True
                if nxt not in seen:
                    stack.append(nxt)
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
        return tuple(self._in_edges.get(nid, ()))

    def children_of(self, nid: str) -> tuple[str, ...]:
        return tuple(self._out_edges.get(nid, ()))

    def to_dict(self) -> dict[str, Any]:
        # 直接迭代内部存储：`nodes` / `edges` 属性每次都会各复制一份 tuple，
        # 而这里只读一次，没必要为 678 个节点 / 2,693 条边付两份中间副本。
        return {
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "edges": [e.to_dict() for e in self._edges],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Lineage":
        g = cls()
        for n in raw.get("nodes", ()):
            g.add_node(Node(str(n["id"]), str(n["kind"]), str(n.get("label", "")), dict(n.get("meta") or {})))
        for e in raw.get("edges", ()):
            # 先全部插入，最后统一做一次拓扑校验；不要每条边都 DFS 一遍。
            g._append_edge(
                Edge(str(e["src"]), str(e["dst"]), str(e["kind"]), str(e.get("label", ""))),
                check_cycle=False,
            )
        g._validate_acyclic()
        return g

    # ---- DOT

    def to_dot(self) -> str:
        lines = ["digraph lineage {", '  rankdir="LR";', '  node [shape=box, style="rounded,filled", fontname="Helvetica"];']
        for n in self._nodes.values():
            color = DOT_COLORS.get(n.kind, "#ffffff")
            label = (n.label or n.id).replace('"', "'")
            lines.append(f'  "{n.id}" [label="{label}", fillcolor="{color}"];')
        for e in self._edges:
            label = e.label or e.kind
            lines.append(f'  "{e.src}" -> "{e.dst}" [label="{label}"];')
        lines.append("}")
        return "\n".join(lines) + "\n"

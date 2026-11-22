"""血缘 DAG 的环检测与邻接索引。

两条容易退化的性质：① 成环的边必须被拒绝，深链也不能退化成 RecursionError；
② `from_dict` 的批量校验与逐条 `add_edge` 必须给出同样的结论。
"""
from __future__ import annotations

import pytest

from preferencefactory.lineage import Edge, Lineage, Node


def _node(nid: str) -> Node:
    return Node(nid, "test")


def test_add_edge_rejects_cycle() -> None:
    g = Lineage()
    for nid in ("a", "b"):
        g.add_node(_node(nid))
    g.add_edge(Edge("a", "b", "x"))
    with pytest.raises(ValueError, match="成环"):
        g.add_edge(Edge("b", "a", "x"))


def test_deep_chain_cycle_does_not_hit_recursion_limit() -> None:
    n = 3000
    g = Lineage()
    for i in range(n):
        g.add_node(_node(str(i)))
    for i in range(n - 1):
        g.add_edge(Edge(str(i), str(i + 1), "x"))
    # 递归实现会在这里抛 RecursionError（默认上限 1000），而不是给出成环结论。
    with pytest.raises(ValueError, match="成环"):
        g.add_edge(Edge(str(n - 1), "0", "x"))


def test_from_dict_rejects_cycle_and_dangling() -> None:
    cycle = {
        "nodes": [{"id": "a", "kind": "x"}, {"id": "b", "kind": "x"}],
        "edges": [{"src": "a", "dst": "b", "kind": "x"}, {"src": "b", "dst": "a", "kind": "x"}],
    }
    with pytest.raises(ValueError, match="成环"):
        Lineage.from_dict(cycle)

    dangling = {"nodes": [{"id": "a", "kind": "x"}], "edges": [{"src": "a", "dst": "zz", "kind": "x"}]}
    with pytest.raises(ValueError, match="不存在"):
        Lineage.from_dict(dangling)


def test_from_dict_roundtrip_preserves_adjacency() -> None:
    g = Lineage()
    for nid in ("a", "b", "c"):
        g.add_node(_node(nid))
    g.add_edge(Edge("a", "b", "x"))
    g.add_edge(Edge("a", "c", "y"))
    g.add_edge(Edge("b", "c", "z"))

    again = Lineage.from_dict(g.to_dict())
    assert [n.id for n in again.nodes] == [n.id for n in g.nodes]
    assert [e.to_dict() for e in again.edges] == [e.to_dict() for e in g.edges]
    assert again.children_of("a") == ("b", "c")
    assert again.parents_of("c") == ("a", "b")

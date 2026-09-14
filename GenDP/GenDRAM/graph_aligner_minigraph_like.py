from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class MinimizerHit:
    query_pos: int
    node_id: str
    node_pos: int
    kmer: str


@dataclass
class GraphNode:
    node_id: str
    seq: str
    out_edges: List[str]


class SequenceGraph:
    def __init__(self) -> None:
        self.nodes: Dict[str, GraphNode] = {}
        self.in_edges: Dict[str, List[str]] = defaultdict(list)

    def add_node(self, node_id: str, seq: str) -> None:
        if node_id in self.nodes:
            self.nodes[node_id].seq = seq
            return
        self.nodes[node_id] = GraphNode(node_id=node_id, seq=seq, out_edges=[])

    def add_edge(self, from_node: str, to_node: str) -> None:
        if from_node not in self.nodes or to_node not in self.nodes:
            raise ValueError(f"Unknown node in edge: {from_node}->{to_node}")
        self.nodes[from_node].out_edges.append(to_node)
        self.in_edges[to_node].append(from_node)

    def topological_order(self) -> List[str]:
        indeg = {node_id: 0 for node_id in self.nodes}
        for node_id, node in self.nodes.items():
            for nxt in node.out_edges:
                indeg[nxt] += 1

        q = deque([n for n, d in indeg.items() if d == 0])
        order: List[str] = []
        while q:
            cur = q.popleft()
            order.append(cur)
            for nxt in self.nodes[cur].out_edges:
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    q.append(nxt)

        if len(order) != len(self.nodes):
            raise ValueError("Graph contains cycle; this aligner expects DAG-like graph")
        return order

    def shortest_node_hops(self) -> Dict[str, Dict[str, int]]:
        hops: Dict[str, Dict[str, int]] = {n: {} for n in self.nodes}
        for src in self.nodes:
            dist = {src: 0}
            q = deque([src])
            while q:
                cur = q.popleft()
                for nxt in self.nodes[cur].out_edges:
                    if nxt not in dist:
                        dist[nxt] = dist[cur] + 1
                        q.append(nxt)
            hops[src] = dist
        return hops

    def path_between(self, src: str, dst: str) -> List[str]:
        if src == dst:
            return [src]
        q = deque([src])
        parent = {src: ""}
        while q:
            cur = q.popleft()
            for nxt in self.nodes[cur].out_edges:
                if nxt in parent:
                    continue
                parent[nxt] = cur
                if nxt == dst:
                    q.clear()
                    break
                q.append(nxt)

        if dst not in parent:
            return []

        path = [dst]
        cur = dst
        while cur != src:
            cur = parent[cur]
            path.append(cur)
        path.reverse()
        return path


class MinigraphLikeAligner:
    """
    A lightweight sequence-to-graph aligner inspired by minigraph's stages:
    minimizer seeding -> sparse chaining -> banded base-level DP on selected path.

    This implementation is simplified for research prototyping and simulator input.
    """

    def __init__(self, k: int = 15, w: int = 10, max_occ: int = 200, band: int = 96):
        self.k = k
        self.w = w
        self.max_occ = max_occ
        self.band = band

        self.graph: SequenceGraph | None = None
        self._index: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        self._topo_rank: Dict[str, int] = {}
        self._hops: Dict[str, Dict[str, int]] = {}

    @staticmethod
    def _canonical_kmer(kmer: str) -> str:
        comp = kmer.translate(str.maketrans("ACGT", "TGCA"))[::-1]
        return min(kmer, comp)

    def _minimizers(self, seq: str) -> List[Tuple[str, int]]:
        if len(seq) < self.k:
            return []

        kmers: List[Tuple[str, int]] = []
        for i in range(len(seq) - self.k + 1):
            kmer = seq[i : i + self.k]
            if any(c not in "ACGT" for c in kmer):
                continue
            kmers.append((self._canonical_kmer(kmer), i))

        if not kmers:
            return []
        if len(kmers) <= self.w:
            m = min(kmers, key=lambda x: x[0])
            return [m]

        mins: List[Tuple[str, int]] = []
        for i in range(len(kmers) - self.w + 1):
            window = kmers[i : i + self.w]
            mins.append(min(window, key=lambda x: x[0]))

        dedup: List[Tuple[str, int]] = []
        seen = set()
        for item in mins:
            if item not in seen:
                dedup.append(item)
                seen.add(item)
        return dedup

    def build_index(self, graph: SequenceGraph) -> None:
        self.graph = graph
        self._index.clear()
        for node_id, node in graph.nodes.items():
            for kmer, pos in self._minimizers(node.seq):
                self._index[kmer].append((node_id, pos))

        order = graph.topological_order()
        self._topo_rank = {node_id: i for i, node_id in enumerate(order)}
        if len(graph.nodes) <= 2000:
            self._hops = graph.shortest_node_hops()
        else:
            self._hops = {}

    def _collect_anchors(self, query: str) -> List[MinimizerHit]:
        anchors: List[MinimizerHit] = []
        for kmer, qpos in self._minimizers(query):
            hits = self._index.get(kmer, [])
            if len(hits) == 0 or len(hits) > self.max_occ:
                continue
            for node_id, npos in hits:
                anchors.append(MinimizerHit(query_pos=qpos, node_id=node_id, node_pos=npos, kmer=kmer))
        anchors.sort(key=lambda x: (x.query_pos, self._topo_rank.get(x.node_id, 10**9), x.node_pos))
        return anchors

    def _chain_score(self, prev: MinimizerHit, cur: MinimizerHit) -> float:
        dq = cur.query_pos - prev.query_pos
        if dq < 0:
            return -1e9

        if prev.node_id == cur.node_id:
            dr = cur.node_pos - prev.node_pos
            if dr < 0:
                return -1e9
            gap = abs(dq - dr)
            return self.k - 0.5 * gap

        hop_dist = self._hops.get(prev.node_id, {}).get(cur.node_id)
        if hop_dist is None:
            prev_rank = self._topo_rank.get(prev.node_id, -1)
            cur_rank = self._topo_rank.get(cur.node_id, -1)
            if prev_rank < 0 or cur_rank < 0 or cur_rank < prev_rank:
                return -1e9
            hop_dist = max(1, cur_rank - prev_rank)

        ref_step = hop_dist * self.k
        gap = abs(dq - ref_step)
        return self.k - 0.8 * gap - 0.2 * hop_dist

    def _sparse_chain(self, anchors: List[MinimizerHit]) -> Tuple[float, List[MinimizerHit]]:
        if not anchors:
            return 0.0, []

        n = len(anchors)
        dp = [float(self.k)] * n
        prev_idx = [-1] * n

        for i in range(n):
            for j in range(max(0, i - 256), i):
                link = self._chain_score(anchors[j], anchors[i])
                if link < -1e8:
                    continue
                cand = dp[j] + link
                if cand > dp[i]:
                    dp[i] = cand
                    prev_idx[i] = j

        best_i = max(range(n), key=lambda i: dp[i])
        chain: List[MinimizerHit] = []
        cur = best_i
        while cur >= 0:
            chain.append(anchors[cur])
            cur = prev_idx[cur]
        chain.reverse()
        return dp[best_i], chain

    def _chain_to_path(self, chain: List[MinimizerHit]) -> List[str]:
        if not chain:
            return []

        nodes = [chain[0].node_id]
        for hit in chain[1:]:
            if hit.node_id == nodes[-1]:
                continue
            bridge = self.graph.path_between(nodes[-1], hit.node_id) if self.graph else []
            if not bridge:
                nodes.append(hit.node_id)
                continue
            nodes.extend(bridge[1:])

        dedup = [nodes[0]]
        for n in nodes[1:]:
            if n != dedup[-1]:
                dedup.append(n)
        return dedup

    def _path_sequence(self, path: List[str]) -> str:
        if not self.graph:
            return ""
        return "".join(self.graph.nodes[node_id].seq for node_id in path)

    def _banded_global_dp(self, query: str, ref: str) -> Dict[str, int]:
        m = len(query)
        n = len(ref)
        if m == 0 or n == 0:
            return {"score": 0, "dp_cells": 0, "edit_distance": max(m, n)}

        neg_inf = -10**12
        match = 2
        mismatch = -4
        gap = -3

        prev = [neg_inf] * (n + 1)
        cur = [neg_inf] * (n + 1)
        prev[0] = 0
        for j in range(1, min(n, self.band) + 1):
            prev[j] = prev[j - 1] + gap

        dp_cells = 0
        for i in range(1, m + 1):
            cur = [neg_inf] * (n + 1)
            left = max(1, i - self.band)
            right = min(n, i + self.band)
            if left == 1:
                cur[0] = prev[0] + gap
            for j in range(left, right + 1):
                dp_cells += 1
                s = match if query[i - 1] == ref[j - 1] else mismatch
                diag = prev[j - 1] + s
                up = prev[j] + gap
                le = cur[j - 1] + gap if j - 1 >= left else neg_inf
                cur[j] = max(diag, up, le)
            prev = cur

        score = prev[n] if abs(m - n) <= self.band else max(prev)

        # A compact approximation for simulator statistics.
        norm = max(1, min(m, n))
        edit_distance = int(max(0, (2 * norm - score) / 3))
        return {"score": int(score), "dp_cells": dp_cells, "edit_distance": edit_distance}

    def align(self, query: str) -> Dict[str, object]:
        if self.graph is None:
            raise ValueError("build_index(graph) must be called first")

        query = query.upper().replace("N", "A")
        anchors = self._collect_anchors(query)
        chain_score, chain = self._sparse_chain(anchors)
        path = self._chain_to_path(chain)
        ref_seq = self._path_sequence(path)

        if not path or not ref_seq:
            return {
                "query_length": len(query),
                "num_anchors": len(anchors),
                "chain_anchors": 0,
                "chain_score": 0,
                "graph_path": [],
                "dp_cells": 0,
                "alignment_score": 0,
                "edit_distance": len(query),
            }

        dp = self._banded_global_dp(query, ref_seq)
        return {
            "query_length": len(query),
            "num_anchors": len(anchors),
            "chain_anchors": len(chain),
            "chain_score": int(chain_score),
            "graph_path": path,
            "dp_cells": dp["dp_cells"],
            "alignment_score": dp["score"],
            "edit_distance": dp["edit_distance"],
        }


def parse_fasta(path: str) -> List[str]:
    seqs: List[str] = []
    cur: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur:
                    seqs.append("".join(cur).upper())
                    cur = []
                continue
            cur.append(line)
    if cur:
        seqs.append("".join(cur).upper())
    return seqs


def parse_fastq(path: str) -> List[str]:
    seqs: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        while True:
            header = ""
            for raw in f:
                header = raw.strip()
                if header:
                    break
            if not header:
                break
            if not header.startswith("@"):
                raise ValueError(f"Invalid FASTQ header line: {header[:40]}")

            seq_parts: List[str] = []
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                if line.startswith("+"):
                    break
                seq_parts.append(line)

            seq = "".join(seq_parts).upper()
            if not seq:
                continue

            # FASTQ quality may be wrapped; consume until quality length >= seq length.
            qlen = 0
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                qlen += len(line)
                if qlen >= len(seq):
                    break

            seqs.append(seq)

    return seqs


def parse_gfa(path: str) -> SequenceGraph:
    graph = SequenceGraph()
    pending_edges: List[Tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            rec = fields[0]
            if rec == "S" and len(fields) >= 3:
                graph.add_node(fields[1], fields[2].upper())
            elif rec == "L" and len(fields) >= 5:
                frm = fields[1]
                to = fields[3]
                pending_edges.append((frm, to))

    for frm, to in pending_edges:
        if frm in graph.nodes and to in graph.nodes:
            graph.add_edge(frm, to)

    if not graph.nodes:
        raise ValueError("No graph segments found in GFA")
    return graph


def build_linear_graph_from_fasta(path: str, chunk_size: int = 512) -> SequenceGraph:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    refs = parse_fasta(path)
    if not refs:
        raise ValueError(f"No sequences found in FASTA: {path}")

    graph = SequenceGraph()
    node_count = 0

    for ridx, ref in enumerate(refs):
        clean_ref = "".join(c if c in "ACGT" else "A" for c in ref)
        prev_node = ""
        for pos in range(0, len(clean_ref), chunk_size):
            chunk = clean_ref[pos : pos + chunk_size]
            if not chunk:
                continue
            node_id = f"r{ridx}_n{node_count}"
            graph.add_node(node_id, chunk)
            if prev_node:
                graph.add_edge(prev_node, node_id)
            prev_node = node_id
            node_count += 1

    if not graph.nodes:
        raise ValueError(f"No graph nodes generated from FASTA: {path}")
    return graph


def build_demo_graph() -> SequenceGraph:
    graph = SequenceGraph()
    graph.add_node("n1", "ACCGTATGGC")
    graph.add_node("n2", "TTACG")
    graph.add_node("n3", "GGAAC")
    graph.add_node("n4", "CTTAGG")

    graph.add_edge("n1", "n2")
    graph.add_edge("n1", "n3")
    graph.add_edge("n2", "n4")
    graph.add_edge("n3", "n4")
    return graph

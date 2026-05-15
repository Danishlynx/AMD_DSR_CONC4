"""Static tree topology for tree speculation (EAGLE-2 style).
Minimum viable: depth=2, topk=[2, 2] = 7 nodes (1 root + 2 children + 4 grandchildren).
"""
import torch
from dataclasses import dataclass
from typing import List


@dataclass
class TreeTopology:
    tree_size: int                    # total nodes including root
    depth: int                        # max depth
    topk_per_level: List[int]         # branching factor per depth level
    parent_indices: torch.Tensor      # [tree_size] int32, parent of each node (-1 for root)
    depth_per_node: torch.Tensor      # [tree_size] int32
    tree_attn_mask: torch.Tensor      # [tree_size, tree_size] bool — ancestor relation
    retrieve_indices: torch.Tensor    # [num_leaves, max_depth+1] int32, paths from root to leaf
    num_leaves: int


def build_static_tree(depth: int = 2, topk_per_level: List[int] = [2, 2],
                       device: str = "cuda") -> TreeTopology:
    """Build a static greedy-topk tree topology.

    For depth=2, topk=[2,2]:
      Node 0: root (depth 0)
      Node 1, 2: children of root (depth 1)
      Node 3, 4: children of node 1 (depth 2)
      Node 5, 6: children of node 2 (depth 2)
      Total: 7 nodes, 4 leaves
    """
    assert len(topk_per_level) == depth

    # Build tree structure level by level
    parent_indices = [-1]  # root has no parent
    depth_per_node = [0]   # root at depth 0
    nodes_at_depth = {0: [0]}  # depth -> list of node indices

    node_idx = 1
    for d in range(1, depth + 1):
        topk = topk_per_level[d - 1]
        nodes_at_depth[d] = []
        for parent in nodes_at_depth[d - 1]:
            for _ in range(topk):
                parent_indices.append(parent)
                depth_per_node.append(d)
                nodes_at_depth[d].append(node_idx)
                node_idx += 1

    tree_size = node_idx
    num_leaves = len(nodes_at_depth[depth])

    # Build ancestor mask: node i can attend to node j if j is an ancestor of i (or j == i)
    tree_attn_mask = torch.zeros(tree_size, tree_size, dtype=torch.bool, device=device)
    for i in range(tree_size):
        # Walk up from i to root, marking all ancestors
        node = i
        while node >= 0:
            tree_attn_mask[i, node] = True
            node = parent_indices[node]

    # Build retrieve_indices: for each leaf, the path from root to leaf
    leaves = nodes_at_depth[depth]
    retrieve_indices = torch.full((num_leaves, depth + 1), -1, dtype=torch.int32, device=device)
    for leaf_idx, leaf in enumerate(leaves):
        path = []
        node = leaf
        while node >= 0:
            path.append(node)
            node = parent_indices[node]
        path.reverse()  # root to leaf order
        for j, p in enumerate(path):
            retrieve_indices[leaf_idx, j] = p

    parent_indices_t = torch.tensor(parent_indices, dtype=torch.int32, device=device)
    depth_per_node_t = torch.tensor(depth_per_node, dtype=torch.int32, device=device)

    return TreeTopology(
        tree_size=tree_size,
        depth=depth,
        topk_per_level=topk_per_level,
        parent_indices=parent_indices_t,
        depth_per_node=depth_per_node_t,
        tree_attn_mask=tree_attn_mask,
        retrieve_indices=retrieve_indices,
        num_leaves=num_leaves,
    )


def build_tree_mask_flat(tree_topo: TreeTopology, bs: int,
                          seq_lens: torch.Tensor, device: str = "cuda"):
    """Build flat custom_mask + mask_indptr for extend_attention_fwd.

    Per SGLang convention with skip_prefix_custom_mask=True:
    - Prefix portion of mask is skipped (all-1s implicitly)
    - Only the extend×extend block (tree_size × tree_size) needs explicit mask
    - mask_indptr[i] = cumsum of mask sizes per sequence

    Returns:
        custom_mask: [total_mask_bytes] uint8
        mask_indptr: [bs+1] int64
        qo_indptr: [bs+1] int32
    """
    tree_size = tree_topo.tree_size

    # qo_indptr: each sequence contributes tree_size extend tokens
    qo_indptr = torch.arange(0, (bs + 1) * tree_size, tree_size,
                              dtype=torch.int32, device=device)

    # mask size per seq: tree_size * (seq_len + tree_size)
    # But with skip_prefix_custom_mask=True, we only need tree_size * tree_size per seq
    mask_per_seq = tree_size * tree_size
    mask_indptr = torch.arange(0, (bs + 1) * mask_per_seq, mask_per_seq,
                                dtype=torch.int64, device=device)

    # Build the flat mask: repeat the tree_attn_mask for each sequence
    tree_mask_flat = tree_topo.tree_attn_mask.view(-1).to(torch.uint8)
    custom_mask = tree_mask_flat.repeat(bs)

    return custom_mask, mask_indptr, qo_indptr

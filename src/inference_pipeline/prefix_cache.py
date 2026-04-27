"""RadixTree-based prefix cache for sharing common prefixes across requests.

Most production LLM traffic has massive prefix redundancy - system prompts,
few-shot examples, tool definitions. Instead of recomputing attention for these
shared prefixes on every request, we cache the KV states and look them up via
a radix tree keyed on token sequences.

# radix tree gives O(prefix_len) lookup vs O(n) scan on the block table
The old approach (scanning the block-level hash table) was O(n) where n is the
total number of cached prefix blocks. The radix tree makes it O(k) where k is
just the prefix length in blocks - much better when you have thousands of
cached prefixes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)

# minimum prefix length worth caching (tokens)
# shorter than this and the cache lookup overhead exceeds the compute savings
MIN_PREFIX_LENGTH = 32

# maximum number of cached prefixes before eviction kicks in
MAX_CACHED_PREFIXES = 8192


@dataclass
class RadixNode:
    """A node in the radix tree representing a segment of a token sequence.

    Each edge from parent to child represents a sequence of tokens (not just one).
    This compression means a common prefix of 1024 tokens might be just 3-4 nodes
    deep instead of 1024 nodes in a trie.
    """

    # the token segment this edge represents (from parent to this node)
    token_segment: tuple[int, ...] = ()
    # children keyed by first token of their segment
    children: dict[int, "RadixNode"] = field(default_factory=dict)
    # if this node terminates a cached prefix, store the cache block IDs
    cache_block_ids: Optional[list[int]] = None
    # reference counting for eviction
    ref_count: int = 0
    last_access_time: float = field(default_factory=time.monotonic)
    # how many tokens total from root to this node (inclusive)
    depth_tokens: int = 0

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def has_cache(self) -> bool:
        return self.cache_block_ids is not None

    def touch(self) -> None:
        self.last_access_time = time.monotonic()
        self.ref_count += 1


@dataclass
class PrefixMatch:
    """Result of a prefix lookup in the radix tree."""

    matched_length: int  # how many tokens matched
    block_ids: list[int]  # cached KV block IDs for the matched prefix
    is_exact: bool  # whether the entire query was matched (vs partial)


class RadixPrefixCache:
    """Radix tree for O(prefix_len) prefix lookup and KV-cache sharing.

    The tree structure:
    - Root represents empty prefix
    - Each edge represents a compressed sequence of tokens
    - Internal nodes can also hold cached blocks (partial prefix matches)
    - Eviction removes least-recently-used leaf nodes when capacity is hit

    Compared to the flat hash table in kv_cache.py:
    - Supports partial prefix matching (find longest cached prefix)
    - O(k) lookup instead of O(n) scan
    - Natural hierarchy: if system_prompt + few_shot is cached, and a new
      request has just system_prompt, we can still use that partial match

    The tradeoff: slightly more complex insertion/deletion logic, and each
    node has overhead. But with typical prefix lengths of 500-2000 tokens
    and thousands of unique prefixes, this wins by a mile.
    """

    def __init__(
        self,
        max_cached_prefixes: int = MAX_CACHED_PREFIXES,
        min_prefix_length: int = MIN_PREFIX_LENGTH,
        block_size_tokens: int = 16,
    ) -> None:
        self._root = RadixNode(token_segment=(), depth_tokens=0)
        self._max_cached = max_cached_prefixes
        self._min_prefix_length = min_prefix_length
        self._block_size = block_size_tokens
        self._num_cached = 0

        # metrics
        self._hits = 0
        self._misses = 0
        self._partial_hits = 0
        self._evictions = 0
        self._total_tokens_saved = 0

        logger.info(
            "radix_prefix_cache_init",
            max_cached=max_cached_prefixes,
            min_prefix_length=min_prefix_length,
            block_size=block_size_tokens,
        )

    def lookup(self, token_ids: list[int]) -> Optional[PrefixMatch]:
        """Find the longest cached prefix matching the start of token_ids.

        Walks down the radix tree following edges that match the token sequence.
        Returns the deepest node that has cached KV blocks, along with how many
        tokens were matched.

        # radix tree gives O(prefix_len) lookup vs O(n) scan on the block table
        Each step down the tree consumes len(edge.token_segment) tokens from the
        query, so total work is proportional to the matched prefix length.
        """
        if len(token_ids) < self._min_prefix_length:
            return None

        current = self._root
        position = 0
        best_match: Optional[PrefixMatch] = None

        while position < len(token_ids):
            # look for a child edge starting with the current token
            next_token = token_ids[position]
            if next_token not in current.children:
                break

            child = current.children[next_token]
            segment = child.token_segment

            # check if the edge segment matches our query
            segment_len = len(segment)
            query_remaining = len(token_ids) - position

            if segment_len > query_remaining:
                # partial edge match - we've gone as far as we can
                # check if the partial match aligns to a block boundary
                match_len = query_remaining
                matching = all(
                    token_ids[position + i] == segment[i] for i in range(match_len)
                )
                if not matching:
                    break
                # partial edge match, use best_match from parent
                break

            # verify the full segment matches
            matching = all(
                token_ids[position + i] == segment[i] for i in range(segment_len)
            )
            if not matching:
                break

            # full edge match, advance
            position += segment_len
            current = child

            # if this node has cached blocks, record as best match so far
            if current.has_cache:
                current.touch()
                best_match = PrefixMatch(
                    matched_length=position,
                    block_ids=list(current.cache_block_ids),  # type: ignore[arg-type]
                    is_exact=(position == len(token_ids)),
                )

        if best_match is not None:
            if best_match.is_exact:
                self._hits += 1
            else:
                self._partial_hits += 1
            self._total_tokens_saved += best_match.matched_length
            return best_match

        self._misses += 1
        return None

    def insert(self, token_ids: list[int], block_ids: list[int]) -> bool:
        """Insert a prefix and its cached KV block IDs into the radix tree.

        If the prefix partially overlaps with existing entries, we split edges
        to create a new branch point. This is the standard radix tree insertion
        algorithm but applied to token sequences instead of strings.

        Returns True if insertion succeeded, False if capacity reached.
        """
        if len(token_ids) < self._min_prefix_length:
            return False

        if self._num_cached >= self._max_cached:
            evicted = self._evict_lru()
            if not evicted:
                return False

        current = self._root
        position = 0

        while position < len(token_ids):
            next_token = token_ids[position]

            if next_token not in current.children:
                # no matching edge, create a new leaf
                remaining = tuple(token_ids[position:])
                new_node = RadixNode(
                    token_segment=remaining,
                    cache_block_ids=list(block_ids),
                    ref_count=1,
                    depth_tokens=len(token_ids),
                )
                new_node.touch()
                current.children[next_token] = new_node
                self._num_cached += 1
                return True

            child = current.children[next_token]
            segment = child.token_segment
            segment_len = len(segment)

            # find how far the segment matches
            match_len = 0
            remaining_query = len(token_ids) - position
            compare_len = min(segment_len, remaining_query)
            while match_len < compare_len and token_ids[position + match_len] == segment[match_len]:
                match_len += 1

            if match_len == segment_len:
                # full edge match, continue down
                position += segment_len
                if position == len(token_ids):
                    # exact match at this node - update cache
                    child.cache_block_ids = list(block_ids)
                    child.touch()
                    if not child.has_cache:
                        self._num_cached += 1
                    return True
                current = child
                continue

            # partial match - need to split the edge
            # create intermediate node at the split point
            matched_segment = segment[:match_len]
            remaining_segment = segment[match_len:]

            split_node = RadixNode(
                token_segment=matched_segment,
                depth_tokens=current.depth_tokens + match_len,
            )

            # old child becomes child of split node
            child.token_segment = remaining_segment
            split_node.children[remaining_segment[0]] = child

            # new branch for our insertion
            new_remaining = tuple(token_ids[position + match_len :])
            if new_remaining:
                new_leaf = RadixNode(
                    token_segment=new_remaining,
                    cache_block_ids=list(block_ids),
                    ref_count=1,
                    depth_tokens=len(token_ids),
                )
                new_leaf.touch()
                split_node.children[new_remaining[0]] = new_leaf
            else:
                # split point is exactly where we want to cache
                split_node.cache_block_ids = list(block_ids)
                split_node.touch()

            # replace old edge with split node
            current.children[next_token] = split_node
            self._num_cached += 1
            return True

        return False

    def invalidate(self, token_ids: list[int]) -> bool:
        """Remove a cached prefix from the tree.

        Used when the underlying KV blocks are evicted from GPU memory.
        We walk down to find the node and clear its cache, but leave the
        tree structure intact (other prefixes might branch from this path).
        """
        current = self._root
        position = 0

        while position < len(token_ids):
            next_token = token_ids[position]
            if next_token not in current.children:
                return False

            child = current.children[next_token]
            segment_len = len(child.token_segment)
            position += segment_len
            current = child

        if current.has_cache:
            current.cache_block_ids = None
            current.ref_count = 0
            self._num_cached -= 1
            return True
        return False

    def _evict_lru(self) -> bool:
        """Evict the least-recently-used cached prefix.

        Walks the entire tree to find the LRU leaf with cache data.
        This is O(n) but only runs when we hit capacity, which should be rare
        if max_cached is sized properly for the workload.
        """
        lru_node: Optional[RadixNode] = None
        lru_time = float("inf")
        lru_parent: Optional[RadixNode] = None
        lru_key: Optional[int] = None

        # BFS to find LRU leaf with cache
        stack: list[tuple[RadixNode, Optional[RadixNode], Optional[int]]] = [
            (self._root, None, None)
        ]
        while stack:
            node, parent, key = stack.pop()
            if node.has_cache and node.last_access_time < lru_time:
                lru_node = node
                lru_time = node.last_access_time
                lru_parent = parent
                lru_key = key
            for child_key, child in node.children.items():
                stack.append((child, node, child_key))

        if lru_node is None:
            return False

        # clear cache from LRU node
        lru_node.cache_block_ids = None
        lru_node.ref_count = 0
        self._num_cached -= 1
        self._evictions += 1

        # if the node is now a useless leaf (no cache, no children), prune it
        if lru_node.is_leaf and lru_parent is not None and lru_key is not None:
            del lru_parent.children[lru_key]

        return True

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._partial_hits + self._misses
        if total == 0:
            return 0.0
        return (self._hits + self._partial_hits) / total

    @property
    def stats(self) -> dict[str, int | float]:
        return {
            "num_cached_prefixes": self._num_cached,
            "max_cached_prefixes": self._max_cached,
            "hits": self._hits,
            "partial_hits": self._partial_hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
            "evictions": self._evictions,
            "total_tokens_saved": self._total_tokens_saved,
        }

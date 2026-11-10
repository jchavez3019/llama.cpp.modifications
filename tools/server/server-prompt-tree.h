#pragma once

// fork-only: radix tree prompt cache, see docs/radix_tree_kv_cache/design.md

#include "common.h"
#include "llama.h"

#include <cstdint>
#include <map>
#include <memory>
#include <vector>

// K/V of the positions [pos0, pos0 + n_tokens) of one path
// pos0 is a multiple of the block size B; n_tokens is B, or less for the short last block of a leaf
struct server_prompt_block {
    llama_pos pos0     = 0;
    int32_t   n_tokens = 0;

    // empty until the data is saved
    std::vector<uint8_t> data;
};

// a run of tokens on one path of the tree, at positions [p0, p0 + tokens.size())
// a node owns the full blocks whose last token is in its run, and a leaf also owns a short block for its tokens after its last full block
struct server_prompt_tree_node {
    llama_pos    p0 = 0;
    llama_tokens tokens;

    server_prompt_tree_node * parent = nullptr;
    std::map<llama_token, std::unique_ptr<server_prompt_tree_node>> children;

    // sorted by pos0
    std::vector<server_prompt_block> blocks;

    uint32_t hits   = 0;
    uint64_t t_last = 0;

    llama_pos p1() const { return p0 + (llama_pos) tokens.size(); }
};

struct server_prompt_tree_match {
    // nodes from the root (excluded) to the node where the match ends
    std::vector<server_prompt_tree_node *> path;

    // tokens that match the tree
    int32_t n_match = 0;

    // tokens from position 0 that the saved blocks cover, at most n_match
    int32_t n_restore = 0;

    // blocks that cover [0, n_restore), in position order; the last one can extend past n_restore
    std::vector<server_prompt_block *> blocks;
};

// index of cached prompts over RAM blocks
// pointers returned by match() and insert() are valid until the next insert() or evict()
struct server_prompt_tree {
    explicit server_prompt_tree(int32_t n_block, int32_t n_min_restore = 0);
    ~server_prompt_tree();

    /**
     * @brief Finds the longest cached prefix of `tokens` and the blocks that restore it.
     *
     * Does not change the tree. Call touch() on the result when the match is used.
     *
     * @param tokens The prompt tokens.
     * @return The matched path, the matched length, and the restorable length with its blocks.
     */
    server_prompt_tree_match match(const llama_tokens & tokens);

    /**
     * @brief Advances the clock by one request, then counts a hit on every node of a matched path and marks them as recently used.
     *
     * Call it once per request, also when nothing matched, so the clock counts requests.
     *
     * @param m A match returned by match() since the last insert() or evict().
     */
    void touch(const server_prompt_tree_match & m);

    /**
     * @brief Adds `tokens` to the tree, splitting a node where the tokens diverge inside it.
     *
     * Splits move block references only. Existing block data is not copied.
     *
     * @param tokens The tokens of a prompt whose K/V exist in a slot at positions [0, tokens.size()).
     * @return The blocks on the path of `tokens` that have no data yet and end at or before tokens.size(), in position order. The caller fills them with set_data().
     */
    std::vector<server_prompt_block *> insert(const llama_tokens & tokens);

    /**
     * @brief Stores the saved K/V of a block and updates the byte count.
     *
     * @param block A block returned by insert().
     * @param data The saved state of the block's positions.
     */
    void set_data(server_prompt_block * block, std::vector<uint8_t> && data);

    /**
     * @brief Removes leaves until the tree holds at most `n_bytes_max` bytes.
     *
     * Leaves are removed bottom-up, lowest aged hit count first, then least recently used. A parent that becomes a leaf is also removed when its run is shorter than the restore threshold.
     *
     * @param n_bytes_max The byte limit to reach.
     * @return The number of bytes freed.
     */
    size_t evict(size_t n_bytes_max);

    // data movement: the context must use a unified KV cache, because seq_cp() copies position ranges only within one stream

    /**
     * @brief Saves the K/V of `blocks` from a sequence into the tree.
     *
     * All blocks are saved with one llama_state_seq_get_data_ranges() call, then split into one buffer per block.
     *
     * @param ctx The context that holds the sequence.
     * @param seq_id The sequence that holds the positions of all blocks.
     * @param blocks Blocks returned by insert(), with no other tree call in between.
     * @return False if the save fails. Then no block gets data.
     */
    bool save_blocks(llama_context * ctx, llama_seq_id seq_id, const std::vector<server_prompt_block *> & blocks);

    /**
     * @brief Restores the positions [0, m.n_restore) of a match into a sequence.
     *
     * Each block is loaded into `seq_scratch` and then moved to `seq_id`. Positions of the last block past n_restore are removed.
     *
     * @param ctx The context that holds the sequence.
     * @param seq_id An empty sequence that receives the K/V.
     * @param seq_scratch An empty sequence, used only for the copy.
     * @param m A match returned by match(), with no insert() or evict() in between.
     * @return The number of positions restored. On failure the sequence is cleared and the return value is 0.
     */
    int32_t restore(llama_context * ctx, llama_seq_id seq_id, llama_seq_id seq_scratch, const server_prompt_tree_match & m);

    size_t n_bytes() const { return n_bytes_total; }
    size_t n_nodes() const;

    const server_prompt_tree_node & get_root() const { return root; }

private:
    int32_t n_block;
    int32_t n_min_restore;

    server_prompt_tree_node root;

    size_t   n_bytes_total = 0;
    uint64_t clock         = 0;

    /**
     * @brief Cuts a node in two at a position inside its run.
     *
     * The node keeps its first `n_keep` tokens. A new child takes the rest of the tokens, the old children, the blocks whose last token is in the rest, and the hit count and last-use time.
     *
     * @param node The node to cut.
     * @param n_keep Tokens the node keeps, in [1, tokens.size()).
     */
    void split(server_prompt_tree_node * node, int32_t n_keep);

    /**
     * @brief Makes a node's block list follow the ownership rule after its run or its children change.
     *
     * The node must own the full blocks whose last token is in its run, and also a short last block if it is a leaf. Blocks that stay keep their data. Blocks that go are freed. New blocks are added with no data. The byte count is updated.
     *
     * @param node The node to fix. The root is skipped.
     */
    void fix_blocks(server_prompt_tree_node * node);

    /**
     * @brief Deletes a node with no children, frees its data, and fixes the blocks of its parent.
     *
     * The parent can become a leaf and get back an empty short block.
     *
     * @param node The leaf to delete.
     */
    void remove_leaf(server_prompt_tree_node * node);

    /**
     * @brief Returns the hit count of a node, halved once for each full age period since its last use.
     *
     * See "Hit count and aging" in docs/radix_tree_kv_cache/design.md.
     *
     * @param node The node to rank.
     * @return hits >> ((clock - t_last) / age period), or 0 after 32 or more periods.
     */
    uint32_t aged_hits(const server_prompt_tree_node & node) const;
};

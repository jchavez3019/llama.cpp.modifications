#include "server-prompt-tree.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

// requests between halvings of a node's hit count, see "Hit count and aging" in design.md
static constexpr uint64_t SERVER_PROMPT_TREE_AGE_PERIOD = 64;

server_prompt_tree::server_prompt_tree(int32_t n_block, int32_t n_min_restore) : n_block(n_block), n_min_restore(n_min_restore) {
    if (n_block <= 0) {
        throw std::invalid_argument("prompt tree block size must be positive");
    }
}

server_prompt_tree::~server_prompt_tree() {
    // tear down without recursion: a long chat history is a deep chain of nodes
    std::vector<std::unique_ptr<server_prompt_tree_node>> stack;
    for (auto & [tok, child] : root.children) {
        stack.push_back(std::move(child));
    }
    root.children.clear();

    while (!stack.empty()) {
        std::unique_ptr<server_prompt_tree_node> node = std::move(stack.back());
        stack.pop_back();
        for (auto & [tok, child] : node->children) {
            stack.push_back(std::move(child));
        }
        node->children.clear();
    }
}

server_prompt_tree_match server_prompt_tree::match(const llama_tokens & tokens) {
    server_prompt_tree_match res;

    // walk down while the tokens match, possibly ending inside a node
    server_prompt_tree_node * node = &root;
    size_t i = 0;
    while (i < tokens.size()) {
        auto it = node->children.find(tokens[i]);
        if (it == node->children.end()) {
            break;
        }
        server_prompt_tree_node * child = it->second.get();
        res.path.push_back(child);

        size_t j = 0;
        while (j < child->tokens.size() && i < tokens.size() && child->tokens[j] == tokens[i]) {
            ++i;
            ++j;
        }
        if (j < child->tokens.size()) {
            break;
        }
        node = child;
    }
    res.n_match = (int32_t) i;

    // collect saved blocks from position 0 without gaps
    llama_pos covered = 0;
    bool gap = false;
    for (server_prompt_tree_node * n : res.path) {
        for (server_prompt_block & b : n->blocks) {
            if (b.pos0 >= res.n_match) {
                break;
            }
            if (b.pos0 != covered || b.data.empty()) {
                gap = true;
                break;
            }
            res.blocks.push_back(&b);
            covered = b.pos0 + b.n_tokens;
        }
        if (gap) {
            break;
        }
    }

    // the block holding the last matched tokens can belong to any child of the last node, because its K/V before the match end do not depend on the child
    if (!gap && covered < res.n_match && !res.path.empty()) {
        for (auto & [tok, child] : res.path.back()->children) {
            if (!child->blocks.empty() && child->blocks[0].pos0 == covered && !child->blocks[0].data.empty()) {
                res.blocks.push_back(&child->blocks[0]);
                covered += child->blocks[0].n_tokens;
                break;
            }
        }
    }
    res.n_restore = std::min<int32_t>(covered, res.n_match);

    return res;
}

void server_prompt_tree::touch(const server_prompt_tree_match & m) {
    ++clock;
    for (server_prompt_tree_node * node : m.path) {
        // age the stored count first, so old hits do not come back
        node->hits   = aged_hits(*node) + 1;
        node->t_last = clock;
    }
}

std::vector<server_prompt_block *> server_prompt_tree::insert(const llama_tokens & tokens) {
    // walk down, split where the tokens diverge inside a node, and add a leaf for the rest
    std::vector<server_prompt_tree_node *> path;
    server_prompt_tree_node * node = &root;
    size_t i = 0;
    while (i < tokens.size()) {
        auto it = node->children.find(tokens[i]);
        if (it == node->children.end()) {
            auto leaf = std::make_unique<server_prompt_tree_node>();
            leaf->p0     = (llama_pos) i;
            leaf->tokens = llama_tokens(tokens.begin() + i, tokens.end());
            leaf->parent = node;

            server_prompt_tree_node * added = leaf.get();
            node->children[tokens[i]] = std::move(leaf);
            fix_blocks(added);
            fix_blocks(node);

            path.push_back(added);
            break;
        }

        server_prompt_tree_node * child = it->second.get();
        path.push_back(child);

        size_t j = 0;
        while (j < child->tokens.size() && i + j < tokens.size() && child->tokens[j] == tokens[i + j]) {
            ++j;
        }
        i += j;

        // the prompt ends inside the node: its tokens are already in the tree
        if (i == tokens.size()) {
            break;
        }
        if (j < child->tokens.size()) {
            split(child, (int32_t) j);
        }
        node = child;
    }

    // blocks the slot can fill now: no data yet, and all their tokens are in the prompt
    std::vector<server_prompt_block *> res;
    for (server_prompt_tree_node * n : path) {
        n->t_last = clock;
        for (server_prompt_block & b : n->blocks) {
            if (b.data.empty() && b.pos0 + b.n_tokens <= (llama_pos) tokens.size()) {
                res.push_back(&b);
            }
        }
    }

    return res;
}

void server_prompt_tree::set_data(server_prompt_block * block, std::vector<uint8_t> && data) {
    n_bytes_total -= block->data.size();
    block->data = std::move(data);
    n_bytes_total += block->data.size();
}

size_t server_prompt_tree::evict(size_t n_bytes_max) {
    const size_t n_bytes_before = n_bytes_total;

    while (n_bytes_total > n_bytes_max) {
        // find the leaf with the lowest aged hit count, then the oldest use
        server_prompt_tree_node * victim = nullptr;
        std::vector<server_prompt_tree_node *> stack = { &root };
        while (!stack.empty()) {
            server_prompt_tree_node * node = stack.back();
            stack.pop_back();
            if (node != &root && node->children.empty()) {
                if (victim == nullptr || aged_hits(*node) < aged_hits(*victim) ||
                        (aged_hits(*node) == aged_hits(*victim) && node->t_last < victim->t_last)) {
                    victim = node;
                }
            }
            for (auto & [tok, child] : node->children) {
                stack.push_back(child.get());
            }
        }
        if (victim == nullptr) {
            break;
        }

        // a parent that becomes a leaf is pruned when its run is too short to be worth a restore
        server_prompt_tree_node * parent = victim->parent;
        remove_leaf(victim);
        while (parent != &root && parent->children.empty() && parent->p1() < n_min_restore) {
            server_prompt_tree_node * next = parent->parent;
            remove_leaf(parent);
            parent = next;
        }
    }

    return n_bytes_before - n_bytes_total;
}

bool server_prompt_tree::save_blocks(llama_context * ctx, llama_seq_id seq_id, const std::vector<server_prompt_block *> & blocks) {
    if (blocks.empty()) {
        return true;
    }

    const int32_t n_blocks = (int32_t) blocks.size();
    std::vector<llama_pos> p0(n_blocks);
    std::vector<llama_pos> p1(n_blocks);
    for (int32_t i = 0; i < n_blocks; ++i) {
        p0[i] = blocks[i]->pos0;
        p1[i] = blocks[i]->pos0 + blocks[i]->n_tokens;
    }

    std::vector<size_t> sizes(n_blocks);
    const size_t size = llama_state_seq_get_size_ranges(ctx, seq_id, p0.data(), p1.data(), n_blocks, sizes.data());
    if (size == 0) {
        return false;
    }

    // one buffer for all blocks, so the K/V of all blocks is read from the device in one batch
    std::unique_ptr<uint8_t[]> buf(new uint8_t[size]);
    if (llama_state_seq_get_data_ranges(ctx, buf.get(), size, seq_id, p0.data(), p1.data(), n_blocks) != size) {
        return false;
    }

    size_t offset = 0;
    for (int32_t i = 0; i < n_blocks; ++i) {
        set_data(blocks[i], std::vector<uint8_t>(buf.get() + offset, buf.get() + offset + sizes[i]));
        offset += sizes[i];
    }

    return true;
}

int32_t server_prompt_tree::restore(llama_context * ctx, llama_seq_id seq_id, llama_seq_id seq_scratch, const server_prompt_tree_match & m) {
    if (m.n_restore == 0) {
        return 0;
    }

    llama_memory_t mem = llama_get_memory(ctx);

    // set_data() clears its target sequence, so each block goes through the scratch sequence
    for (const server_prompt_block * b : m.blocks) {
        if (llama_state_seq_set_data(ctx, b->data.data(), b->data.size(), seq_scratch) == 0) {
            llama_memory_seq_rm(mem, seq_scratch, -1, -1);
            llama_memory_seq_rm(mem, seq_id, -1, -1);
            return 0;
        }
        llama_memory_seq_cp(mem, seq_scratch, seq_id, -1, -1);
        llama_memory_seq_rm(mem, seq_scratch, -1, -1);
    }

    // the last block can extend past the match
    llama_memory_seq_rm(mem, seq_id, m.n_restore, -1);

    return m.n_restore;
}

size_t server_prompt_tree::n_nodes() const {
    size_t n = 0;
    std::vector<const server_prompt_tree_node *> stack = { &root };
    while (!stack.empty()) {
        const server_prompt_tree_node * node = stack.back();
        stack.pop_back();
        for (const auto & [tok, child] : node->children) {
            stack.push_back(child.get());
            ++n;
        }
    }
    return n;
}

void server_prompt_tree::split(server_prompt_tree_node * node, int32_t n_keep) {
    auto tail = std::make_unique<server_prompt_tree_node>();
    tail->p0       = node->p0 + n_keep;
    tail->tokens   = llama_tokens(node->tokens.begin() + n_keep, node->tokens.end());
    tail->parent   = node;
    tail->children = std::move(node->children);
    tail->hits     = node->hits;
    tail->t_last   = node->t_last;
    for (auto & [tok, child] : tail->children) {
        child->parent = tail.get();
    }

    // blocks whose last token is in the tail move with it, data included
    std::vector<server_prompt_block> keep;
    for (server_prompt_block & b : node->blocks) {
        if (b.pos0 + b.n_tokens - 1 >= tail->p0) {
            tail->blocks.push_back(std::move(b));
        } else {
            keep.push_back(std::move(b));
        }
    }
    node->blocks = std::move(keep);

    node->tokens.resize(n_keep);
    node->children.clear();

    server_prompt_tree_node * added = tail.get();
    node->children[added->tokens[0]] = std::move(tail);
    fix_blocks(added);
    fix_blocks(node);
}

void server_prompt_tree::fix_blocks(server_prompt_tree_node * node) {
    if (node == &root) {
        return;
    }

    // the blocks this node must own: full blocks ending in its run, plus a short last block if it is a leaf
    std::vector<std::pair<llama_pos, int32_t>> want;
    for (llama_pos k = node->p0 / n_block; (k + 1) * n_block <= node->p1(); ++k) {
        want.emplace_back(k * n_block, n_block);
    }
    const llama_pos pos_short = (node->p1() / n_block) * n_block;
    if (node->children.empty() && pos_short < node->p1()) {
        want.emplace_back(pos_short, node->p1() - pos_short);
    }

    size_t n_bytes_old = 0;
    for (const server_prompt_block & b : node->blocks) {
        n_bytes_old += b.data.size();
    }

    // keep the data of blocks that are still wanted, drop the others
    std::vector<server_prompt_block> blocks;
    for (const auto & [pos0, n_tokens] : want) {
        server_prompt_block nb;
        nb.pos0     = pos0;
        nb.n_tokens = n_tokens;
        for (server_prompt_block & b : node->blocks) {
            if (b.pos0 == pos0 && b.n_tokens == n_tokens) {
                nb.data = std::move(b.data);
                break;
            }
        }
        blocks.push_back(std::move(nb));
    }
    n_bytes_total -= n_bytes_old;
    for (const server_prompt_block & b : blocks) {
        n_bytes_total += b.data.size();
    }
    node->blocks = std::move(blocks);
}

void server_prompt_tree::remove_leaf(server_prompt_tree_node * node) {
    for (const server_prompt_block & b : node->blocks) {
        n_bytes_total -= b.data.size();
    }
    server_prompt_tree_node * parent = node->parent;
    parent->children.erase(node->tokens[0]);
    fix_blocks(parent);
}

uint32_t server_prompt_tree::aged_hits(const server_prompt_tree_node & node) const {
    const uint64_t age = (clock - node.t_last) / SERVER_PROMPT_TREE_AGE_PERIOD;
    return age >= 32 ? 0 : node.hits >> age;
}

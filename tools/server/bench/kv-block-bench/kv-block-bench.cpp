// kv-block-bench: measures prefill vs state restore, and checks that a sequence saved as fixed-size blocks restores correctly
// see docs/radix_tree_kv_cache/roadplan/phase_1_measurements.md

#include "llama.h"
#include "server-prompt-tree.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#ifdef __GLIBC__
#include <malloc.h>
#endif

static const llama_seq_id SEQ_MAIN    = 0;
static const llama_seq_id SEQ_SCRATCH = 1;
static const llama_seq_id SEQ_PARKED  = 2;
static const uint32_t     TOKEN_SEED  = 1234;

/** @brief Command line options. */
struct bench_args {
    std::string model;
    std::string out;
    std::string device = "unknown";
    std::string block  = "auto";

    int32_t ngl       = 99;
    int32_t n_ctx     = 8192;
    int32_t n_ubatch  = 512;
    int32_t n_threads = -1;
    int32_t n_prefix  = 1500;
    int32_t reps      = 5;
    bool    verbose   = false;

    std::vector<int32_t> n_list = { 16, 64, 256, 1024, 4096 };
    std::vector<int32_t> b_list = { 16, 64, 256 };
};

/** @brief One timed measurement, written as one JSON line. */
struct bench_row {
    const char * kind;
    int32_t n;
    int32_t b;
    int32_t rep;
    double  ms;
    size_t  bytes;
    int32_t calls;
};

/** @brief State shared by all measurements. */
struct bench_ctx {
    llama_context   * ctx;
    llama_memory_t    mem;
    llama_batch_ext * batch;
    FILE            * out;
    std::string       device;
    std::string       model;
    int32_t           n_vocab;
};

static void print_usage(const char * argv0) {
    printf("\nusage: %s -m model.gguf -o out.jsonl [options]\n\n", argv0);
    printf("  --device NAME     label written to every row (default: unknown)\n");
    printf("  -ngl N            GPU layers (default: 99)\n");
    printf("  -c N              context size (default: 8192)\n");
    printf("  -ub N             physical batch size (default: 512, same as llama-server)\n");
    printf("  -t N              threads (default: llama.cpp default)\n");
    printf("  --n-list LIST     token counts, comma separated (default: 16,64,256,1024,4096)\n");
    printf("  --b-list LIST     block sizes, comma separated (default: 16,64,256)\n");
    printf("  --prefix N        tokens before the timed prefill in prefill_offset (default: 1500)\n");
    printf("  -r N              repetitions after one warm-up (default: 5)\n");
    printf("  --block MODE      auto|on|off; auto runs the block path only for full-attention models\n");
    printf("  -v                print llama.cpp logs\n\n");
}

/** @brief Parse "16,64,256" into integers. Returns false on a bad or non-positive value. */
static bool parse_int_list(const char * s, std::vector<int32_t> & out) {
    out.clear();
    std::string cur;
    for (const char * p = s; ; ++p) {
        if (*p == ',' || *p == '\0') {
            try {
                const int32_t v = std::stoi(cur);
                if (v <= 0) {
                    return false;
                }
                out.push_back(v);
            } catch (...) {
                return false;
            }
            cur.clear();
            if (*p == '\0') {
                break;
            }
        } else {
            cur += *p;
        }
    }
    return !out.empty();
}

/** @brief Parse argv into args. Returns false and prints usage on error. */
static bool parse_args(int argc, char ** argv, bench_args & args) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const bool has_val = i + 1 < argc;

        try {
            if (a == "-m" && has_val) {
                args.model = argv[++i];
            } else if (a == "-o" && has_val) {
                args.out = argv[++i];
            } else if (a == "--device" && has_val) {
                args.device = argv[++i];
            } else if (a == "-ngl" && has_val) {
                args.ngl = std::stoi(argv[++i]);
            } else if (a == "-c" && has_val) {
                args.n_ctx = std::stoi(argv[++i]);
            } else if (a == "-ub" && has_val) {
                args.n_ubatch = std::stoi(argv[++i]);
            } else if (a == "-t" && has_val) {
                args.n_threads = std::stoi(argv[++i]);
            } else if (a == "--prefix" && has_val) {
                args.n_prefix = std::stoi(argv[++i]);
            } else if (a == "-r" && has_val) {
                args.reps = std::stoi(argv[++i]);
            } else if (a == "--block" && has_val) {
                args.block = argv[++i];
            } else if (a == "--n-list" && has_val) {
                if (!parse_int_list(argv[++i], args.n_list)) {
                    return false;
                }
            } else if (a == "--b-list" && has_val) {
                if (!parse_int_list(argv[++i], args.b_list)) {
                    return false;
                }
            } else if (a == "-v") {
                args.verbose = true;
            } else {
                return false;
            }
        } catch (...) {
            return false;
        }
    }

    const bool block_ok = args.block == "auto" || args.block == "on" || args.block == "off";
    return !args.model.empty() && !args.out.empty() && block_ok && args.reps > 0 && args.n_prefix >= 0 && args.n_ubatch > 0;
}

static void log_quiet(ggml_log_level level, const char * text, void * /*user_data*/) {
    if (level == GGML_LOG_LEVEL_ERROR) {
        fputs(text, stderr);
    }
}

/** @brief Escape a string for a JSON value (quotes and backslashes only; inputs are labels and model names). */
static std::string json_str(const std::string & s) {
    std::string r = "\"";
    for (const char c : s) {
        if (c == '"' || c == '\\') {
            r += '\\';
        }
        r += c;
    }
    return r + "\"";
}

static void emit_row(const bench_ctx & bc, const bench_row & row) {
    fprintf(bc.out, "{\"device\":%s,\"model\":%s,\"kind\":\"%s\",\"n\":%d,\"b\":%d,\"rep\":%d,\"ms\":%.4f,\"bytes\":%zu,\"calls\":%d}\n",
            json_str(bc.device).c_str(), json_str(bc.model).c_str(), row.kind, row.n, row.b, row.rep, row.ms, row.bytes, row.calls);
    fflush(bc.out);
}

static void emit_check(const bench_ctx & bc, const char * kind, int32_t n, int32_t b, float max_abs_diff, bool argmax_match) {
    fprintf(bc.out, "{\"device\":%s,\"model\":%s,\"kind\":\"%s\",\"n\":%d,\"b\":%d,\"max_abs_diff\":%.6g,\"argmax_match\":%s}\n",
            json_str(bc.device).c_str(), json_str(bc.model).c_str(), kind, n, b, max_abs_diff, argmax_match ? "true" : "false");
    fflush(bc.out);
}

static double elapsed_ms(int64_t t0_us) {
    return (llama_time_us() - t0_us) / 1000.0;
}

/** @brief Decode tokens[p0, p0 + n) at positions p0.. on one sequence, with logits for the last token only. */
static bool decode_range(bench_ctx & bc, const std::vector<llama_token> & tokens, int32_t p0, int32_t n, llama_seq_id seq) {
    llama_batch_ext_clear(bc.batch);
    for (int32_t i = 0; i < n; ++i) {
        const int32_t idx = llama_batch_ext_add_token(bc.batch, seq, tokens[p0 + i]);
        if (idx < 0) {
            return false;
        }
        const llama_pos pos = p0 + i;
        llama_batch_ext_set_pos(bc.batch, idx, &pos);
    }
    llama_batch_ext_set_output_logits(bc.batch, n - 1, true);

    if (llama_process(bc.ctx, LLAMA_PROCESS_TYPE_DECODE, bc.batch) != 0) {
        return false;
    }
    llama_synchronize(bc.ctx);
    return true;
}

/** @brief Decode the probe token at position n and copy its logits. Assumes positions [0, n) of SEQ_MAIN are in the cache. */
static bool probe_logits(bench_ctx & bc, const std::vector<llama_token> & tokens, int32_t n, std::vector<float> & logits) {
    if (!decode_range(bc, tokens, n, 1, SEQ_MAIN)) {
        return false;
    }
    const float * l = llama_get_logits_ith(bc.ctx, -1);
    if (l == nullptr) {
        return false;
    }
    logits.assign(l, l + bc.n_vocab);
    return true;
}

static void compare_logits(const std::vector<float> & ref, const std::vector<float> & cur, float & max_abs_diff, bool & argmax_match) {
    max_abs_diff = 0.0f;
    for (size_t i = 0; i < ref.size(); ++i) {
        max_abs_diff = std::max(max_abs_diff, std::fabs(ref[i] - cur[i]));
    }
    const auto ref_top = std::max_element(ref.begin(), ref.end()) - ref.begin();
    const auto cur_top = std::max_element(cur.begin(), cur.end()) - cur.begin();
    argmax_match = ref_top == cur_top;
}

/** @brief Prefill n tokens from an empty cache (kind prefill_cold) and after n_prefix tokens (kind prefill_offset). */
static bool bench_prefill(bench_ctx & bc, const std::vector<llama_token> & tokens, int32_t n, int32_t n_prefix, int32_t rep) {
    llama_memory_clear(bc.mem, true);
    int64_t t0 = llama_time_us();
    if (!decode_range(bc, tokens, 0, n, SEQ_MAIN)) {
        return false;
    }
    const double ms_cold = elapsed_ms(t0);

    llama_memory_clear(bc.mem, true);
    if (n_prefix > 0 && !decode_range(bc, tokens, 0, n_prefix, SEQ_MAIN)) {
        return false;
    }
    t0 = llama_time_us();
    if (!decode_range(bc, tokens, n_prefix, n, SEQ_MAIN)) {
        return false;
    }
    const double ms_offset = elapsed_ms(t0);

    if (rep >= 0) {
        emit_row(bc, { "prefill_cold",   n, 0, rep, ms_cold,   0, 1 });
        emit_row(bc, { "prefill_offset", n, 0, rep, ms_offset, 0, 1 });
    }
    return true;
}

/** @brief Save and restore the whole sequence with today's API (kinds whole_save and whole_restore). */
static bool bench_whole(bench_ctx & bc, const std::vector<llama_token> & tokens, int32_t n, int32_t rep, std::vector<uint8_t> & buf, const std::vector<float> * ref) {
    llama_memory_clear(bc.mem, true);
    if (!decode_range(bc, tokens, 0, n, SEQ_MAIN)) {
        return false;
    }

    int64_t t0 = llama_time_us();
    const size_t size = llama_state_seq_get_size(bc.ctx, SEQ_MAIN);
    buf.resize(size);
    if (llama_state_seq_get_data(bc.ctx, buf.data(), buf.size(), SEQ_MAIN) != size) {
        return false;
    }
    const double ms_save = elapsed_ms(t0);

    llama_memory_seq_rm(bc.mem, SEQ_MAIN, -1, -1);

    t0 = llama_time_us();
    if (llama_state_seq_set_data(bc.ctx, buf.data(), buf.size(), SEQ_MAIN) == 0) {
        return false;
    }
    llama_synchronize(bc.ctx);
    const double ms_restore = elapsed_ms(t0);

    if (rep >= 0) {
        emit_row(bc, { "whole_save",    n, 0, rep, ms_save,    size, 1 });
        emit_row(bc, { "whole_restore", n, 0, rep, ms_restore, size, 1 });
    }

    if (ref != nullptr) {
        std::vector<float> cur;
        if (!probe_logits(bc, tokens, n, cur)) {
            return false;
        }
        float diff;
        bool  match;
        compare_logits(*ref, cur, diff, match);
        emit_check(bc, "check_whole", n, 0, diff, match);
    }
    return true;
}

/** @brief Restore SEQ_MAIN from the block states `bufs`, probe position n, and compare with `ref`. */
static bool check_blocks(bench_ctx & bc, const std::vector<llama_token> & tokens, int32_t n, int32_t b, const std::vector<std::vector<uint8_t>> & bufs, const std::vector<float> & ref, const char * kind) {
    llama_memory_clear(bc.mem, true);
    for (const auto & buf : bufs) {
        if (llama_state_seq_set_data(bc.ctx, buf.data(), buf.size(), SEQ_SCRATCH) == 0) {
            return false;
        }
        llama_memory_seq_cp(bc.mem, SEQ_SCRATCH, SEQ_MAIN, -1, -1);
        llama_memory_seq_rm(bc.mem, SEQ_SCRATCH, -1, -1);
    }

    const llama_pos pos_max = llama_memory_seq_pos_max(bc.mem, SEQ_MAIN);
    std::vector<float> cur;
    if (pos_max != n - 1 || !probe_logits(bc, tokens, n, cur)) {
        fprintf(stderr, "%s failed: n = %d, b = %d, pos_max = %d\n", kind, n, b, pos_max);
        emit_check(bc, kind, n, b, INFINITY, false);
        return true;
    }
    float diff;
    bool  match;
    compare_logits(ref, cur, diff, match);
    emit_check(bc, kind, n, b, diff, match);
    return true;
}

/**
 * @brief Save the sequence as n / b blocks and restore it block by block (kinds block_save, block_batch_save and block_restore).
 *
 * Save: tag the block's cells with SEQ_SCRATCH (seq_cp takes a position range), save SEQ_SCRATCH, drop the tag.
 * Batch save: one llama_state_seq_get_data_ranges() call for all blocks.
 * Restore: set_data clears its destination first, so each block goes into SEQ_SCRATCH and is then re-tagged to SEQ_MAIN.
 */
static bool bench_block(bench_ctx & bc, const std::vector<llama_token> & tokens, int32_t n, int32_t b, int32_t rep, std::vector<std::vector<uint8_t>> & bufs, std::vector<std::vector<uint8_t>> & batch_bufs, const std::vector<float> * ref) {
    const int32_t n_blocks = n / b;
    bufs.resize(n_blocks);

    llama_memory_clear(bc.mem, true);
    if (!decode_range(bc, tokens, 0, n, SEQ_MAIN)) {
        return false;
    }

    // both saves put each block in a new buffer, as the server does
    size_t bytes = 0;
    int64_t t0 = llama_time_us();
    for (int32_t k = 0; k < n_blocks; ++k) {
        llama_memory_seq_cp(bc.mem, SEQ_MAIN, SEQ_SCRATCH, k * b, (k + 1) * b);
        const size_t size = llama_state_seq_get_size(bc.ctx, SEQ_SCRATCH);
        bufs[k] = std::vector<uint8_t>(size);
        if (llama_state_seq_get_data(bc.ctx, bufs[k].data(), size, SEQ_SCRATCH) != size) {
            return false;
        }
        llama_memory_seq_rm(bc.mem, SEQ_SCRATCH, -1, -1);
        bytes += size;
    }
    const double ms_save = elapsed_ms(t0);

    std::vector<llama_pos> p0(n_blocks);
    std::vector<llama_pos> p1(n_blocks);
    for (int32_t k = 0; k < n_blocks; ++k) {
        p0[k] = k * b;
        p1[k] = (k + 1) * b;
    }
    std::vector<size_t> sizes(n_blocks);
    std::vector<uint8_t *> dst(n_blocks);
    batch_bufs.clear();
    batch_bufs.resize(n_blocks);

    t0 = llama_time_us();
    const size_t batch_bytes = llama_state_seq_get_size_ranges(bc.ctx, SEQ_MAIN, p0.data(), p1.data(), n_blocks, sizes.data());
    for (int32_t k = 0; k < n_blocks; ++k) {
        batch_bufs[k].resize(sizes[k]);
        dst[k] = batch_bufs[k].data();
    }
    if (batch_bytes == 0 || llama_state_seq_get_data_ranges(bc.ctx, dst.data(), sizes.data(), SEQ_MAIN, p0.data(), p1.data(), n_blocks) != batch_bytes) {
        return false;
    }
    const double ms_batch_save = elapsed_ms(t0);

    // a range state has the same layout as a scratch sequence state, so the sizes must match
    for (int32_t k = 0; k < n_blocks; ++k) {
        if (sizes[k] != bufs[k].size()) {
            fprintf(stderr, "batch save size mismatch: n = %d, b = %d, block %d: %zu vs %zu\n", n, b, k, sizes[k], bufs[k].size());
            return false;
        }
    }

    llama_memory_seq_rm(bc.mem, SEQ_MAIN, -1, -1);

    t0 = llama_time_us();
    for (int32_t k = 0; k < n_blocks; ++k) {
        if (llama_state_seq_set_data(bc.ctx, bufs[k].data(), bufs[k].size(), SEQ_SCRATCH) == 0) {
            return false;
        }
        llama_memory_seq_cp(bc.mem, SEQ_SCRATCH, SEQ_MAIN, -1, -1);
        llama_memory_seq_rm(bc.mem, SEQ_SCRATCH, -1, -1);
    }
    llama_synchronize(bc.ctx);
    const double ms_restore = elapsed_ms(t0);

    if (rep >= 0) {
        emit_row(bc, { "block_save",       n, b, rep, ms_save,       bytes,       n_blocks });
        emit_row(bc, { "block_batch_save", n, b, rep, ms_batch_save, batch_bytes, 1        });
        emit_row(bc, { "block_restore",    n, b, rep, ms_restore,    bytes,       n_blocks });
    }

    if (ref != nullptr) {
        const llama_pos pos_max = llama_memory_seq_pos_max(bc.mem, SEQ_MAIN);
        std::vector<float> cur;
        if (pos_max != n - 1 || !probe_logits(bc, tokens, n, cur)) {
            fprintf(stderr, "block restore check failed: n = %d, b = %d, pos_max = %d\n", n, b, pos_max);
            emit_check(bc, "check_block", n, b, INFINITY, false);
        } else {
            float diff;
            bool  match;
            compare_logits(*ref, cur, diff, match);
            emit_check(bc, "check_block", n, b, diff, match);
        }

        return check_blocks(bc, tokens, n, b, batch_bufs, *ref, "check_block_batch");
    }
    return true;
}

/** @brief Decode tokens[p_from, n) into `seq` in chunks of b, with p_from a multiple of b. Prompts that share a prefix then get the same batch shapes, so the shared K/V are bit-identical. */
static bool decode_chunked(bench_ctx & bc, const std::vector<llama_token> & tokens, int32_t n, int32_t b, llama_seq_id seq = SEQ_MAIN, int32_t p_from = 0) {
    for (int32_t p0 = p_from; p0 < n; p0 += b) {
        if (!decode_range(bc, tokens, p0, std::min(b, n - p0), seq)) {
            return false;
        }
    }
    return true;
}

/** @brief Restore the cached prefix of `prompt[0, n)` from the tree on top of the first p_keep positions that SEQ_MAIN holds, probe position n, and compare with `ref`. */
static void check_tree_case(bench_ctx & bc, server_prompt_tree & tree, const std::vector<llama_token> & prompt, int32_t n, int32_t b, const std::vector<float> & ref, const char * kind, llama_pos p_keep = 0) {
    if (p_keep == 0) {
        llama_memory_clear(bc.mem, true);
    }
    const server_prompt_tree_match m = tree.match(llama_tokens(prompt.begin(), prompt.begin() + n));
    const int32_t n_restored = tree.restore(bc.ctx, SEQ_MAIN, SEQ_SCRATCH, m, p_keep);
    llama_synchronize(bc.ctx);

    std::vector<float> cur;
    const bool ok = n_restored == n && llama_memory_seq_pos_max(bc.mem, SEQ_MAIN) == n - 1 && llama_memory_seq_pos_max(bc.mem, SEQ_SCRATCH) == -1 && probe_logits(bc, prompt, n, cur);
    if (!ok) {
        fprintf(stderr, "%s failed: n = %d, b = %d, restored = %d\n", kind, n, b, n_restored);
        emit_check(bc, kind, n, b, INFINITY, false);
        return;
    }
    float diff;
    bool  match;
    compare_logits(ref, cur, diff, match);
    emit_check(bc, kind, n, b, diff, match);
}

/** @brief Probe position n of SEQ_MAIN, compare with `ref`, and emit the result. `ok` is false if the setup already failed. */
static void check_probe(bench_ctx & bc, const std::vector<llama_token> & prompt, int32_t n, int32_t b, const std::vector<float> & ref, const char * kind, bool ok) {
    std::vector<float> cur;
    if (!ok || llama_memory_seq_pos_max(bc.mem, SEQ_MAIN) != n - 1 || !probe_logits(bc, prompt, n, cur)) {
        fprintf(stderr, "%s failed: n = %d, b = %d\n", kind, n, b);
        emit_check(bc, kind, n, b, INFINITY, false);
        return;
    }
    float diff;
    bool  match;
    compare_logits(ref, cur, diff, match);
    emit_check(bc, kind, n, b, diff, match);
}

/**
 * @brief Check attach of a parked sequence by tag only (kinds check_attach_*).
 *
 * Prompt 1 stays in SEQ_PARKED, as a parked tree leaf. Then:
 * - prefix: SEQ_MAIN attaches the first block of prompt 1 with seq_cp, which prompt 2 shares, and prefills the rest of prompt 2
 * - parked: SEQ_MAIN is removed, then attaches all of prompt 1; the cells must still be there, because SEQ_PARKED tags them
 *
 * Prompt 2's own cells take other cell indices than in the reference, so this also checks that the cell layout does not change the result.
 */
static bool check_attach(bench_ctx & bc, const std::vector<llama_token> & prompt1, int32_t n1, const std::vector<llama_token> & prompt2, int32_t n2, int32_t b, const std::vector<float> & ref1, const std::vector<float> & ref2) {
    llama_memory_clear(bc.mem, true);
    if (!decode_chunked(bc, prompt1, n1, b, SEQ_PARKED)) {
        return false;
    }

    llama_memory_seq_cp(bc.mem, SEQ_PARKED, SEQ_MAIN, 0, b);
    const bool ok_prefix = decode_chunked(bc, prompt2, n2, b, SEQ_MAIN, b);
    check_probe(bc, prompt2, n2, b, ref2, "check_attach_prefix", ok_prefix);

    llama_memory_seq_rm(bc.mem, SEQ_MAIN, -1, -1);
    const bool ok_parked = llama_memory_seq_pos_min(bc.mem, SEQ_PARKED) == 0 && llama_memory_seq_pos_max(bc.mem, SEQ_PARKED) == n1 - 1;
    llama_memory_seq_cp(bc.mem, SEQ_PARKED, SEQ_MAIN, 0, n1);
    check_probe(bc, prompt1, n1, b, ref1, "check_attach_parked", ok_parked);
    return true;
}

/**
 * @brief Check server_prompt_tree save and restore against a cold prefill (kinds check_tree_*).
 *
 * Prompt 1 has 4b tokens. Prompt 2 shares the first p tokens, with p in the middle of a block, then has its own tokens up to 3b + 5. Both are saved through the tree. Then three restores are checked:
 * - branch: all of prompt 2, which ends in a short block
 * - split:  all of prompt 1, whose node was split by prompt 2
 * - inside: the first p - 1 tokens, whose last block belongs to a child
 * - keep:   all of prompt 2 into a sequence that holds prompt 1, keeping its first block
 * Then check_attach() runs on the same prompts and references.
 */
static bool check_tree(bench_ctx & bc, const std::vector<llama_token> & tokens, int32_t b) {
    const int32_t n1 = 4 * b;
    const int32_t p  = b + b / 2 + 1;
    const int32_t n2 = 3 * b + 5;

    // each prompt has one extra token for the probe
    const std::vector<llama_token> prompt1(tokens.begin(), tokens.begin() + n1 + 1);
    std::vector<llama_token> prompt2(tokens.begin(), tokens.begin() + p);
    prompt2.insert(prompt2.end(), tokens.end() - (n2 - p + 1), tokens.end());

    // prefill, save the new blocks, then take the reference logits
    server_prompt_tree tree(b);
    std::vector<float> ref1;
    std::vector<float> ref2;
    std::vector<float> ref_inside;
    for (int32_t i = 0; i < 2; ++i) {
        const std::vector<llama_token> & prompt = i == 0 ? prompt1 : prompt2;
        const int32_t n = i == 0 ? n1 : n2;
        llama_memory_clear(bc.mem, true);
        if (!decode_chunked(bc, prompt, n, b)) {
            return false;
        }
        if (!tree.save_blocks(bc.ctx, SEQ_MAIN, tree.insert(llama_tokens(prompt.begin(), prompt.begin() + n)))) {
            fprintf(stderr, "tree save failed: b = %d\n", b);
            return false;
        }
        if (!probe_logits(bc, prompt, n, i == 0 ? ref1 : ref2)) {
            return false;
        }
    }
    // the inside restore can take its last block from either prompt, so the reference trims a full prefill instead of a shorter one with other batch shapes
    llama_memory_clear(bc.mem, true);
    if (!decode_chunked(bc, prompt1, n1, b)) {
        return false;
    }
    llama_memory_seq_rm(bc.mem, SEQ_MAIN, p - 1, -1);
    if (!probe_logits(bc, prompt1, p - 1, ref_inside)) {
        return false;
    }

    check_tree_case(bc, tree, prompt2, n2,    b, ref2,       "check_tree_branch");
    check_tree_case(bc, tree, prompt1, n1,    b, ref1,       "check_tree_split");
    check_tree_case(bc, tree, prompt1, p - 1, b, ref_inside, "check_tree_inside");

    // a slot that holds prompt 1 keeps its first block, which prompt 2 shares, and gets the rest of prompt 2 from the tree
    llama_memory_clear(bc.mem, true);
    if (!decode_chunked(bc, prompt1, n1, b)) {
        return false;
    }
    check_tree_case(bc, tree, prompt2, n2, b, ref2, "check_tree_keep", b);

    return check_attach(bc, prompt1, n1, prompt2, n2, b, ref1, ref2);
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

#ifdef __GLIBC__
    // a fixed threshold keeps each block buffer on new pages, as in a growing prompt tree; by default glibc reuses freed memory and hides the page faults
    mallopt(M_MMAP_THRESHOLD, 128 * 1024);
#endif

    bench_args args;
    if (!parse_args(argc, argv, args)) {
        print_usage(argv[0]);
        return 1;
    }

    const int32_t n_max = *std::max_element(args.n_list.begin(), args.n_list.end());
    if (args.n_prefix + n_max + 1 > args.n_ctx) {
        fprintf(stderr, "error: --prefix + max(--n-list) + 1 = %d does not fit in -c %d\n", args.n_prefix + n_max + 1, args.n_ctx);
        return 1;
    }

    if (!args.verbose) {
        llama_log_set(log_quiet, nullptr);
    }
    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = args.ngl;

    llama_model * model = llama_model_load_from_file(args.model.c_str(), mparams);
    if (model == nullptr) {
        fprintf(stderr, "error: unable to load model %s\n", args.model.c_str());
        return 1;
    }

    // the same KV layout as the benchmarked server: unified pool, flash attention, one scratch sequence
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = args.n_ctx;
    cparams.n_batch         = std::max(n_max, args.n_prefix);
    cparams.n_ubatch        = args.n_ubatch;
    cparams.n_seq_max       = 3;
    cparams.kv_unified      = true;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    if (args.n_threads > 0) {
        cparams.n_threads       = args.n_threads;
        cparams.n_threads_batch = args.n_threads;
    }

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        fprintf(stderr, "error: failed to create the context\n");
        return 1;
    }

    FILE * out = fopen(args.out.c_str(), "w");
    if (out == nullptr) {
        fprintf(stderr, "error: cannot open %s\n", args.out.c_str());
        return 1;
    }

    char desc[256];
    llama_model_desc(model, desc, sizeof(desc));

    bench_ctx bc = { ctx, llama_get_memory(ctx), llama_batch_ext_init(ctx), out, args.device, desc, llama_vocab_n_tokens(llama_model_get_vocab(model)) };

    // the block path splits K/V by position, which is only valid when every layer keeps per-token K/V for the whole context
    const bool full_attn = !llama_model_is_recurrent(model) && !llama_model_is_hybrid(model) && llama_model_n_swa(model) == 0;
    const bool run_block = args.block == "on" || (args.block == "auto" && full_attn);
    fprintf(stderr, "model: %s, full attention: %s, block path: %s\n", desc, full_attn ? "yes" : "no", run_block ? "on" : "off");

    // prefill cost does not depend on token content, so seeded random ids are enough
    std::vector<llama_token> tokens(args.n_prefix + n_max + 1);
    std::mt19937 rng(TOKEN_SEED);
    std::uniform_int_distribution<llama_token> dist(0, bc.n_vocab - 1);
    for (auto & t : tokens) {
        t = dist(rng);
    }

    std::vector<uint8_t> whole_buf;
    std::vector<std::vector<uint8_t>> block_bufs;
    std::vector<std::vector<uint8_t>> batch_bufs;

    for (const int32_t n : args.n_list) {
        // reference logits for the probe token after a cold prefill of n tokens
        std::vector<float> ref;
        llama_memory_clear(bc.mem, true);
        if (!decode_range(bc, tokens, 0, n, SEQ_MAIN) || !probe_logits(bc, tokens, n, ref)) {
            fprintf(stderr, "error: reference decode failed for n = %d\n", n);
            return 1;
        }

        // rep -1 is the warm-up and is not written; the correctness checks run once, on the last rep
        for (int32_t rep = -1; rep < args.reps; ++rep) {
            const std::vector<float> * check = rep == args.reps - 1 ? &ref : nullptr;

            if (!bench_prefill(bc, tokens, n, args.n_prefix, rep) || !bench_whole(bc, tokens, n, rep, whole_buf, check)) {
                fprintf(stderr, "error: measurement failed for n = %d\n", n);
                return 1;
            }

            if (!run_block) {
                continue;
            }
            for (const int32_t b : args.b_list) {
                if (b > n || n % b != 0) {
                    continue;
                }
                if (!bench_block(bc, tokens, n, b, rep, block_bufs, batch_bufs, check)) {
                    fprintf(stderr, "error: block measurement failed for n = %d, b = %d\n", n, b);
                    return 1;
                }
            }
        }
        fprintf(stderr, "n = %d done\n", n);
    }

    for (const int32_t b : args.b_list) {
        // both prompts take distinct token ranges, and prompt 1 is one batch
        if (!run_block || 7 * b + 7 > (int32_t) tokens.size() || 4 * b > (int32_t) cparams.n_batch) {
            continue;
        }
        if (!check_tree(bc, tokens, b)) {
            fprintf(stderr, "error: tree check failed for b = %d\n", b);
            return 1;
        }
    }

    fclose(out);
    llama_batch_ext_free(bc.batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}

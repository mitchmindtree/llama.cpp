#include "llama-memory-hybrid-idx.h"

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-model.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iterator>
#include <map>
#include <stdexcept>

//
// llama_memory_hybrid_idx
//

llama_memory_hybrid_idx::llama_memory_hybrid_idx(
        const llama_model & model,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                 uint32_t   n_ubatch,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr,
    const layer_filter_cb & filter_idx) :
    llama_memory_hybrid(
        model,
        type_k, type_v, v_trans, kv_size, n_pad, n_swa, swa_type,
        type_r, type_s, rs_size,
        n_seq_max, n_rs_seq, offload, unified,
        filter_attn, filter_recr),
    hparams_idx(model.hparams),
    mem_idx(filter_idx == nullptr ? nullptr : [&] {
        // MQA with a single key head of indexer_head_size, as llama_kv_cache_dsa shapes its own
        std::fill(hparams_idx.n_head_kv_arr.begin(), hparams_idx.n_head_kv_arr.end(), 1);
        hparams_idx.n_embd_head_k_full = model.hparams.indexer_head_size;

        LLAMA_LOG_INFO("%s: creating indexer KV cache, size = %u cells\n", __func__, kv_size);

        return new llama_kv_cache(
            model, hparams_idx, type_k, type_v, v_trans, offload, unified,
            kv_size, n_seq_max, n_pad, n_swa, swa_type,
            nullptr, filter_idx, nullptr, nullptr, "idx_");
    }()) {
    // Pooled QSA block keys: one scoring-ready key per completed block, per QSA
    // layer. Sized by the largest compress ratio's block count, padded so the
    // graph view width is stable, plus one scratch row per possible commit slot:
    // set_rows forbids overlapping destination rows, so idle slots cannot share one.
    // Rows are indexed by block = pos/ratio, so with more than one sequence the
    // same row would be contested: the cache only exists for n_seq_max == 1
    // (the MTP configuration), everything else keeps the re-pooling path.
    if (mem_idx && filter_idx && n_seq_max == 1) {
        const auto & hp = model.hparams;

        pooled_ns_ = 1;

        uint32_t r_min = 0;
        for (uint32_t il = 0; il < hp.n_layer(); ++il) {
            if (filter_idx(il) && hp.dsv4_compress_ratios[il] > 0) {
                r_min = r_min == 0 ? hp.dsv4_compress_ratios[il]
                                   : std::min<uint32_t>(r_min, hp.dsv4_compress_ratios[il]);
            }
        }

        if (r_min > 0) {
            pooled_scratch_ = GGML_PAD((kv_size + r_min - 1)/r_min, 256);
            pooled_rows_    = pooled_scratch_ + n_ubatch/r_min + 2;

            struct buft_cmp {
                bool operator()(const ggml_backend_buffer_type_t & a, const ggml_backend_buffer_type_t & b) const {
                    return strcmp(ggml_backend_buft_name(a), ggml_backend_buft_name(b)) < 0;
                }
            };

            std::map<ggml_backend_buffer_type_t, ggml_context_ptr, buft_cmp> ctx_map;

            auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
                auto it = ctx_map.find(buft);
                if (it == ctx_map.end()) {
                    ggml_init_params params = {
                        /*.mem_size   =*/ size_t(2u*hp.n_layer()*ggml_tensor_overhead()),
                        /*.mem_buffer =*/ NULL,
                        /*.no_alloc   =*/ true,
                    };
                    ggml_context * ctx = ggml_init(params);
                    if (ctx) {
                        ctx_map.emplace(buft, ctx);
                    }
                    return ctx;
                }
                return it->second.get();
            };

            for (uint32_t il = 0; il < hp.n_layer(); ++il) {
                if (!filter_idx(il) || hp.dsv4_compress_ratios[il] == 0) {
                    continue;
                }

                ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
                if (offload) {
                    buft = ggml_backend_dev_buffer_type(model.dev_layer(il));
                }

                ggml_context * ctx = ctx_for_buft(buft);
                if (!ctx) {
                    throw std::runtime_error("failed to create ggml context for the pooled QSA keys");
                }

                ggml_tensor * t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                        hp.indexer_head_size, pooled_rows_, pooled_ns_);
                ggml_format_name(t, "qsa_pooled_l%d", il);

                pooled_map.emplace((int32_t) il, (int32_t) pooled_layers.size());
                pooled_layers.push_back({ il, t });
            }

            for (auto & [buft, ctx] : ctx_map) {
                ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft);
                if (!buf) {
                    throw std::runtime_error("failed to allocate buffer for the pooled QSA keys");
                }
                ggml_backend_buffer_clear(buf, 0);
                pooled_ctxs_bufs.emplace_back(std::move(ctx), ggml_backend_buffer_ptr(buf));
            }
        }
    }
}

ggml_tensor * llama_memory_hybrid_idx::get_pooled(int32_t il) const {
    const auto it = pooled_map.find(il);
    return it == pooled_map.end() ? nullptr : pooled_layers[it->second].t;
}

uint32_t llama_memory_hybrid_idx::pooled_rows() const {
    return pooled_rows_;
}

uint32_t llama_memory_hybrid_idx::pooled_scratch() const {
    return pooled_scratch_;
}

uint32_t llama_memory_hybrid_idx::pooled_n_stream() const {
    return pooled_ns_;
}

bool llama_memory_hybrid_idx::pooled_valid() const {
    return pooled_valid_;
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    // note: repeats llama_memory_hybrid::init_batch, as the indexer needs the attention slot infos that the base context hides
    do {
        balloc.split_reset();

        // follow the recurrent pattern for creating the ubatch splits
        std::vector<llama_ubatch> ubatches;

        while (true) {
            llama_ubatch ubatch;

            if (embd_all) {
                // if all tokens are output, split by sequence
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                // Use non-sequential split when KV cache is unified (needed for hellaswag/winogrande/multiple-choice)
                const bool unified = (get_mem_attn()->get_n_stream() == 1);

                // [TAG_RECURRENT_ROLLBACK_SPLITS]
                // the trailing (1 + n_rs_seq) tokens of each seq must stay in the same ubatch
                //   so that the rollback snapshots remain valid
                const uint32_t n_rs_seq = get_mem_recr()->n_rs_seq;

                ubatch = balloc.split_equal(n_ubatch, !unified, n_rs_seq > 0 ? n_rs_seq + 1 : 0);
            }

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        // prepare the recurrent batches first
        if (!get_mem_recr()->prepare(ubatches)) {
            // TODO: will the recurrent cache be in an undefined context at this point?
            LLAMA_LOG_ERROR("%s: failed to prepare recurrent ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // prepare the attention cache
        auto heads_attn = get_mem_attn()->prepare(ubatches);
        if (heads_attn.empty()) {
            LLAMA_LOG_ERROR("%s: failed to prepare attention ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // the indexer uses the attention cache's slot layout; a separate one can drift from it
        llama_kv_cache::slot_info_vec_t heads_idx;
        if (mem_idx) {
            heads_idx = heads_attn;
        }

        return std::make_unique<llama_memory_hybrid_idx_context>(
                this, std::move(heads_attn), std::move(heads_idx), std::move(ubatches));
    } while(false);

    return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_full() {
    return std::make_unique<llama_memory_hybrid_idx_context>(this);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_memory_hybrid_idx_context>(this, lctx, optimize);
}

void llama_memory_hybrid_idx::clear(bool data) {
    llama_memory_hybrid::clear(data);

    if (mem_idx) {
        mem_idx->clear(data);
    }

    // no filled blocks left, so every pooled row is unreachable: a fresh start
    pooled_valid_ = true;
}

bool llama_memory_hybrid_idx::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // same order as llama_memory_hybrid::seq_rm: the recurrent cache can refuse, so try it first
    if (!get_mem_recr()->seq_rm(seq_id, p0, p1)) {
        return false;
    }

    if (mem_idx) {
        mem_idx->seq_rm(seq_id, p0, p1);
    }

    // a partial removal needs nothing: un-filled blocks stop competing on their own,
    // and a block only re-completes through a commit that overwrites its row.
    // a full removal empties the (only, see the ctor) sequence, which revalidates like clear
    if (p0 <= 0 && p1 < 0) {
        pooled_valid_ = true;
    }

    return get_mem_attn()->seq_rm(seq_id, p0, p1);
}

void llama_memory_hybrid_idx::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    llama_memory_hybrid::seq_cp(seq_id_src, seq_id_dst, p0, p1);

    if (mem_idx) {
        mem_idx->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    }

    if (seq_id_src != seq_id_dst) {
        pooled_valid_ = false;
    }
}

void llama_memory_hybrid_idx::seq_keep(llama_seq_id seq_id) {
    llama_memory_hybrid::seq_keep(seq_id);

    if (mem_idx) {
        mem_idx->seq_keep(seq_id);
    }
}

void llama_memory_hybrid_idx::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    llama_memory_hybrid::seq_add(seq_id, p0, p1, shift);

    if (mem_idx) {
        mem_idx->seq_add(seq_id, p0, p1, shift);
    }

    // positions moved under the pooled keys (they carry rope), so they no longer match
    if (shift != 0) {
        pooled_valid_ = false;
    }
}

void llama_memory_hybrid_idx::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    llama_memory_hybrid::seq_div(seq_id, p0, p1, d);

    if (mem_idx) {
        mem_idx->seq_div(seq_id, p0, p1, d);
    }

    if (d != 1) {
        pooled_valid_ = false;
    }
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_hybrid_idx::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = llama_memory_hybrid::memory_breakdown();

    if (mem_idx) {
        for (const auto & buft_size : mem_idx->memory_breakdown()) {
            mb[buft_size.first] += buft_size.second;
        }
    }

    for (const auto & [ctx, buf] : pooled_ctxs_bufs) {
        mb[ggml_backend_buffer_get_type(buf.get())] += ggml_backend_buffer_get_size(buf.get());
    }

    return mb;
}

void llama_memory_hybrid_idx::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    llama_memory_hybrid::state_write(io, seq_id, flags);

    // [TAG_HYBRID_IDX_STATE] the indexer section goes last, so it is a pure suffix: an old reader stops early instead of misparsing it
    // The indexer mirrors the attention cache, so it uses the same PARTIAL_ONLY gate.
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        if (mem_idx) {
            mem_idx->state_write(io, seq_id, flags);
        }
    }

}

void llama_memory_hybrid_idx::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    // note: repeats llama_memory_hybrid::state_read
    // the indexer needs the attention cache's cells, and a half-failed restore must leave all three caches alike

    // [TAG_HYBRID_IDX_SINFO]
    // the indexer restore adopts the attention cache's layout instead of searching for cells of its own
    // two find_slot calls agree only while both caches see the same occupancy, which a restore cannot promise
    llama_kv_cache::slot_info_vec_t sinfos_attn;

    try {
        if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            get_mem_attn()->state_read_sinfo(io, seq_id, flags, mem_idx ? &sinfos_attn : nullptr, nullptr);
        }

        get_mem_recr()->state_read(io, seq_id, flags);

        // [TAG_HYBRID_IDX_STATE] must mirror the write order in state_write
        if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            if (mem_idx) {
                mem_idx->state_read_sinfo(io, seq_id, flags, nullptr, &sinfos_attn);
            }
        }

    } catch (...) {
        // a half-restored context is the one state the indexer cannot fix by itself: attention holds new cells, the indexer old ones
        // drop what was being restored from all of them, which is a state they do agree on.
        state_drop(seq_id);

        throw;
    }

    // the pooled rows are not part of the state blob: a restore brings back filled
    // blocks whose rows here are stale, so fall back to re-pooling until a clear
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        pooled_valid_ = false;
    }
}

void llama_memory_hybrid_idx::state_drop(llama_seq_id seq_id) {
    // dropped directly, not via seq_rm: the recurrent cache may refuse it and then only the other two get cleared
    if (seq_id < 0) {
        clear(true);

        return;
    }

    get_mem_attn()->seq_rm(seq_id, -1, -1);
    get_mem_recr()->seq_rm(seq_id, -1, -1);

    if (mem_idx) {
        mem_idx->seq_rm(seq_id, -1, -1);
    }
}

llama_kv_cache * llama_memory_hybrid_idx::get_mem_idx() const {
    return mem_idx.get();
}

//
// llama_memory_hybrid_idx_context
//

// streams in each ubatch's slot info, matching get_k/get_v's `ns`
static std::vector<uint32_t> llama_memory_hybrid_idx_ns(const llama_kv_cache::slot_info_vec_t & sinfos) {
    std::vector<uint32_t> res;
    res.reserve(sinfos.size());

    for (const auto & sinfo : sinfos) {
        res.push_back(sinfo.s1 - sinfo.s0 + 1);
    }

    return res;
}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_status status) :
    llama_memory_hybrid_context(status) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_hybrid_idx * mem) :
    llama_memory_hybrid_context(mem),
    mem(mem),
    // graph reservation walks a full context, and qwen4exp builds the sparse attention only when this is set
    // without it the reserved worst case is the dense graph, so ggml-alloc must grow the buffer on the first decode
    ns_ubatch(mem->get_mem_idx() == nullptr ?
        std::vector<uint32_t>() : std::vector<uint32_t>{ mem->get_mem_idx()->get_n_stream() }),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx())) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                  llama_context * lctx,
                           bool   optimize) :
    llama_memory_hybrid_context(mem, lctx, optimize),
    mem(mem) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                slot_info_vec_t   sinfos_attn,
                slot_info_vec_t   sinfos_idx,
      std::vector<llama_ubatch>   ubatches) :
    // note: the base copies the ubatches; ctx_idx gets a copy of its own
    llama_memory_hybrid_context(mem, std::move(sinfos_attn), ubatches),
    mem(mem),
    ns_ubatch(llama_memory_hybrid_idx_ns(sinfos_idx)),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx(), std::move(sinfos_idx), ubatches)) {}

bool llama_memory_hybrid_idx_context::next() {
    if (ctx_idx) {
        ctx_idx->next();
    }

    ++i_cur;

    return llama_memory_hybrid_context::next();
}

bool llama_memory_hybrid_idx_context::apply() {
    bool res = llama_memory_hybrid_context::apply();

    if (ctx_idx) {
        res = res & ctx_idx->apply();
    }

    return res;
}

const llama_kv_cache_context * llama_memory_hybrid_idx_context::get_idx() const {
    return static_cast<const llama_kv_cache_context *>(ctx_idx.get());
}

uint32_t llama_memory_hybrid_idx_context::get_n_stream() const {
    GGML_ASSERT(i_cur < ns_ubatch.size());

    return ns_ubatch[i_cur];
}

ggml_tensor * llama_memory_hybrid_idx_context::get_pooled(int32_t il) const {
    return mem ? mem->get_pooled(il) : nullptr;
}

uint32_t llama_memory_hybrid_idx_context::pooled_rows() const {
    return mem ? mem->pooled_rows() : 0;
}

bool llama_memory_hybrid_idx_context::pooled_valid() const {
    return mem && mem->pooled_valid();
}

void llama_memory_hybrid_idx_context::set_input_qsa(
        ggml_tensor * cell_blk,
        ggml_tensor * blk_cells,
        ggml_tensor * blk_pos,
        ggml_tensor * bias,
        const llama_ubatch * ubatch,
        uint32_t ratio,
        bool blk_bias,
        ggml_tensor * commit_cells,
        ggml_tensor * commit_rows,
        ggml_tensor * commit_pos) const {
    GGML_ASSERT(ratio > 0);
    GGML_ASSERT(mem != nullptr && mem->get_mem_idx() != nullptr);

    GGML_ASSERT(ggml_backend_buffer_is_host(cell_blk->buffer));

    // the pooled path drops blk_cells and blk_pos: nothing in its graph reads them
    GGML_ASSERT((blk_cells != nullptr && blk_pos != nullptr) || (blk_bias && commit_cells != nullptr));

    const int64_t n_kv     = cell_blk->ne[0];
    const int64_t n_ns     = cell_blk->ne[1];        // streams in this ubatch
    const int64_t n_blocks = blk_pos != nullptr ? blk_pos->ne[0]/(4*n_ns) : bias->ne[0];
    const int64_t n_tokens = ubatch->n_tokens;
    const int64_t r        = ratio;

    GGML_ASSERT(n_tokens % n_ns == 0);
    const int64_t n_tps = n_tokens/n_ns;             // tokens per stream

    int32_t * dst_cell_blk  = (int32_t *) cell_blk->data;
    int32_t * dst_blk_cells = blk_cells != nullptr ? (int32_t *) blk_cells->data : nullptr;
    int32_t * dst_blk_pos   = blk_pos   != nullptr ? (int32_t *) blk_pos->data   : nullptr;
    float   * dst_bias      = (float   *) bias->data;

    // block b covers [b*ratio, (b+1)*ratio), so its first token is at b*ratio
    // all mrope sections carry it: exact for text, approximate for images
    if (dst_blk_pos) {
        for (int64_t sec = 0; sec < 4; ++sec) {
            for (int64_t s = 0; s < n_ns; ++s) {
                for (int64_t b = 0; b < n_blocks; ++b) {
                    dst_blk_pos[sec*(n_blocks*n_ns) + s*n_blocks + b] = (int32_t) (b*r);
                }
            }
        }
    }

    // one pass per stream: cell j is a different token in each, so no mapping is shared
    std::vector<int32_t> blk_of(n_kv);
    std::vector<int32_t> filled(n_blocks);
    std::vector<int32_t> members(r*n_blocks);

    for (int64_t s = 0; s < n_ns; ++s) {
        // ubatch index s*n_tps belongs to this stream; ask which cells array it uses
        const llama_seq_id seq_of_stream = ubatch->seq_id[s*n_tps][0];
        const auto & cells = mem->get_mem_idx()->get_cells(seq_of_stream);

        int32_t * cur_cell_blk = dst_cell_blk + s*n_kv;

        // an incomplete block cannot be pooled; the bias below forces those tail cells in
        // -1 means no usable block, and block 0 only keeps the gather in range
        std::fill(blk_of.begin(),  blk_of.end(),  -1);
        std::fill(filled.begin(),  filled.end(),   0);
        std::fill(members.begin(), members.end(),  0);

        // a cell no block covers needs its own -inf, which a per-block bias cannot carry
        // every cache path keeps the position below the cell window, so this stays false
        bool oor = false;

        for (int64_t j = 0; j < n_kv; ++j) {
            if (cells.is_empty(j)) {
                continue;
            }

            const llama_pos p = cells.pos_get(j);
            const int64_t   b = p/r;

            if (b >= n_blocks) {
                oor = true;
                continue;
            }

            blk_of[j] = (int32_t) b;
            members[b*r + (p%r)] = (int32_t) j;
            filled[b]++;
        }

        GGML_ASSERT((!blk_bias || !oor) && "qsa: cell position runs past the cell window");

        if (dst_blk_cells) {
            std::copy(members.begin(), members.end(), dst_blk_cells + s*(r*n_blocks));
        }

        // per-block mode keeps an unpooled cell's real block, so the block's own -inf reaches it
        // per-cell mode carries that -inf itself and only needs the gather in range
        for (int64_t j = 0; j < n_kv; ++j) {
            if (blk_of[j] >= 0 && filled[blk_of[j]] < r && !blk_bias) {
                blk_of[j] = -1;
            }
            cur_cell_blk[j] = blk_of[j] < 0 ? 0 : blk_of[j];
        }

        // commit plan for the pooled key cache: every full block one of this
        // ubatch's tokens belongs to. That includes blocks completed here for
        // the first time, and refills after a rollback. Re-commits are
        // idempotent: the members are gathered fresh from the cache.
        // The recurrent half of the model keeps positions sequential, so a
        // ubatch touches contiguous blocks and the fixed slot count suffices.
        if (commit_cells) {
            GGML_ASSERT(commit_rows && commit_pos);
            GGML_ASSERT(n_ns == 1 && "qsa: the pooled cache only exists for one sequence");

            const int64_t cap = commit_rows->ne[0];

            // one scratch row per idle slot: set_rows destinations must not overlap
            const int32_t row_scratch = (int32_t) mem->pooled_scratch();
            GGML_ASSERT(row_scratch + cap <= (int64_t) mem->pooled_rows());

            int32_t * dst_c_cells = (int32_t *) commit_cells->data;
            int32_t * dst_c_rows  = (int32_t *) commit_rows->data;
            int32_t * dst_c_pos   = (int32_t *) commit_pos->data;

            int64_t n_commit = 0;
            int64_t last_b   = -1;

            for (int64_t ii = 0; ii < n_tps; ++ii) {
                const llama_pos q = ubatch->pos[ii];
                const int64_t   b = q/r;

                if (b == last_b || b >= n_blocks || filled[b] < r) {
                    continue;
                }
                last_b = b;

                GGML_ASSERT(n_commit < cap && "qsa: commit plan overflow");

                dst_c_rows[n_commit] = (int32_t) b;
                for (int64_t m = 0; m < r; ++m) {
                    dst_c_cells[n_commit*r + m] = members[b*r + m];
                }
                for (int64_t sec = 0; sec < 4; ++sec) {
                    dst_c_pos[sec*cap + n_commit] = (int32_t) (b*r);
                }
                n_commit++;
            }

            // idle slots pool cell 0 into their own scratch row, which nothing reads
            for (; n_commit < cap; ++n_commit) {
                dst_c_rows[n_commit] = row_scratch + (int32_t) n_commit;
                for (int64_t m = 0; m < r; ++m) {
                    dst_c_cells[n_commit*r + m] = 0;
                }
                for (int64_t sec = 0; sec < 4; ++sec) {
                    dst_c_pos[sec*cap + n_commit] = 0;
                }
            }
        }

        for (int64_t ii = 0; ii < n_tps; ++ii) {
            const int64_t      i      = s*n_tps + ii;
            const llama_seq_id seq_id = ubatch->seq_id[i][0];
            const llama_pos    q      = ubatch->pos[i];

            // the tail is an incomplete block and is always visible, as in the reference
            const llama_pos tail_start = (q + 1)/r*r;

            if (blk_bias) {
                // a block sits wholly inside or outside the tail, so one value covers it
                // the caller adds the attention mask, which drops empty, foreign and future cells
                float * cur_blk_bias = dst_bias + i*n_blocks;

                // Entirely-future blocks get -inf so the bias alone decides block-level
                // selection (the per-cell attention mask still guards the cells either
                // way). The tail comparison stays as before for the query's own block.
                const int64_t b_q = q/r;

                for (int64_t b = 0; b < n_blocks; ++b) {
                    // 1e9 is finite, so it can never meet a -inf and produce a nan
                    cur_blk_bias[b] = b > b_q          ? -INFINITY
                                    : b*r >= tail_start ? 1e9f
                                    : filled[b] < r     ? -INFINITY
                                                        : 0.0f;
                }

                continue;
            }

            float * cur_bias = dst_bias + i*n_kv;

            for (int64_t j = 0; j < n_kv; ++j) {
                float v = -INFINITY;

                if (!cells.is_empty(j) && cells.seq_has(j, seq_id) && cells.pos_get(j) <= q) {
                    // finite, so it can never meet a -inf and produce a nan
                    v = cells.pos_get(j) >= tail_start ? 1e9f : (blk_of[j] < 0 ? -INFINITY : 0.0f);
                }

                cur_bias[j] = v;
            }
        }
    }
}

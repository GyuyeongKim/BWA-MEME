#include "LearnedIndex_batch.h"
#include "kbtree.h"
#include <algorithm>

// Defined via KSORT_INIT in bwamem.cpp
extern void ks_introsort_mem_smem_sort_lt_learned(size_t n, mem_tl a[]);

// Defined in bwamem.cpp
extern int mem_chain_flt(const mem_opt_t *opt, int n_chn_, mem_chain_t *a_, int tid);
extern void mem_flt_chained_seeds(const mem_opt_t *opt, const bntseq_t *bns, const uint8_t *pac,
                                   bseq1_t *s, int n_chn, mem_chain_t *a);

#if (__AVX512BW__ || __AVX2__)
#include <immintrin.h>
#else
#include <smmintrin.h>
#endif

extern uint64_t tprof[LIM_R][LIM_C];

// Per-read context for interleaved seeding
struct ReadSeedCtx {
    // Per-read 2-bit packed buffers
    uint8_t shift1[LEARNED_MAX_READ_LEN/4+1];
    uint8_t shift2[LEARNED_MAX_READ_LEN/4+1];
    uint8_t shift3[LEARNED_MAX_READ_LEN/4+1];
    uint8_t shift4[LEARNED_MAX_READ_LEN/4+1];
    uint8_t rc_shift1[LEARNED_MAX_READ_LEN/4+1];
    uint8_t rc_shift2[LEARNED_MAX_READ_LEN/4+1];
    uint8_t rc_shift3[LEARNED_MAX_READ_LEN/4+1];
    uint8_t rc_shift4[LEARNED_MAX_READ_LEN/4+1];
    uint8_t rc_buf[ERT_MAX_READ_LEN];

    Learned_read_aux_t raux;

    // Seeding output
    mem_tlv* smems;
    u64v* hits;

    // Read info
    int hasN;
    int len;

    // Batch prefetch working state
    uint64_t pf_position;   // predicted SA position (from lookup or ref2sa)
    size_t   pf_err;        // error bound (Path C only)
    uint32_t pf_ambig;      // ambiguous position
};

// Prepare one read: nt4 conversion + 2-bit packing (same logic as original)
static void prepare_read(ReadSeedCtx* ctx, bseq1_t* seq, const mem_opt_t* opt,
                          mem_tlv* smems, u64v* hits)
{
    char *s = seq->seq;
    int len = seq->l_seq;

    ctx->smems = smems;
    ctx->hits = hits;
    ctx->len = len;

    ctx->hasN = 0;
    if (strchr(s, 'N') || strchr(s, 'n')) ctx->hasN = 1;

    // ASCII -> nt4 + reverse complement
    for (int i = 0; i < len; ++i) {
        s[i] = s[i] < 4 ? s[i] : nst_nt4_table[(int)s[i]];
        ctx->rc_buf[len - i - 1] = s[i] < 4 ? 3 - s[i] : 4;
    }

    // 2-bit packing into shift buffers
    uint8_t set_bit = 0, set_rc_bit = 0;
    int k;
    for (k = 0; k < len; ++k) {
        set_bit = set_bit << 2;
        set_rc_bit = set_rc_bit << 2;
        set_bit |= s[k] < 4 ? s[k] : 0;
        set_rc_bit |= s[len-1-k] < 4 ? 3 - s[len-1-k] : 0;
        if ((k&3) == 0) {
            ctx->shift1[k>>2] = BitReverseTable256[set_bit];
            ctx->rc_shift1[k>>2] = BitReverseTable256[set_rc_bit];
        } else if ((k&3) == 1) {
            ctx->shift2[k>>2] = BitReverseTable256[set_bit];
            ctx->rc_shift2[k>>2] = BitReverseTable256[set_rc_bit];
        } else if ((k&3) == 2) {
            ctx->shift3[k>>2] = BitReverseTable256[set_bit];
            ctx->rc_shift3[k>>2] = BitReverseTable256[set_rc_bit];
        } else {
            ctx->shift4[k>>2] = BitReverseTable256[set_bit];
            ctx->rc_shift4[k>>2] = BitReverseTable256[set_rc_bit];
        }
    }
    for (; k < len+4; ++k) {
        set_bit = set_bit << 2;
        set_rc_bit = set_rc_bit << 2;
        if ((k&3) == 0) {
            ctx->shift1[k>>2] = BitReverseTable256[set_bit];
            ctx->rc_shift1[k>>2] = BitReverseTable256[set_rc_bit];
        } else if ((k&3) == 1) {
            ctx->shift2[k>>2] = BitReverseTable256[set_bit];
            ctx->rc_shift2[k>>2] = BitReverseTable256[set_rc_bit];
        } else if ((k&3) == 2) {
            ctx->shift3[k>>2] = BitReverseTable256[set_bit];
            ctx->rc_shift3[k>>2] = BitReverseTable256[set_rc_bit];
        } else {
            ctx->shift4[k>>2] = BitReverseTable256[set_bit];
            ctx->rc_shift4[k>>2] = BitReverseTable256[set_rc_bit];
        }
    }

    // Wire raux pointers to local buffers
    ctx->raux.unpacked_queue_binary_buf_shift1 = ctx->shift1;
    ctx->raux.unpacked_queue_binary_buf_shift2 = ctx->shift2;
    ctx->raux.unpacked_queue_binary_buf_shift3 = ctx->shift3;
    ctx->raux.unpacked_queue_binary_buf_shift4 = ctx->shift4;
    ctx->raux.unpacked_rc_queue_binary_buf_shift1 = ctx->rc_shift1;
    ctx->raux.unpacked_rc_queue_binary_buf_shift2 = ctx->rc_shift2;
    ctx->raux.unpacked_rc_queue_binary_buf_shift3 = ctx->rc_shift3;
    ctx->raux.unpacked_rc_queue_binary_buf_shift4 = ctx->rc_shift4;

    ctx->raux.min_seed_len = opt->min_seed_len;
    ctx->raux.l_seq = len;
    ctx->raux.max_l_seq = 0;
    ctx->raux.read_name = seq->name;
    ctx->raux.unpacked_queue_buf = (uint8_t*)s;
    ctx->raux.unpacked_rc_queue_buf = ctx->rc_buf;
    ctx->raux.min_intv_limit = 1;
}

/*
 * Interleaved seeding: processes N reads with batched prefetching.
 *
 * Strategy: Batch the first pivot's SA prefetch across reads,
 * then delegate remaining pivots to the original step1 function
 * which correctly handles all zigzag edge cases.
 */
int mem_kernel1_core_Learned_Interleaved(
    const mem_opt_t *opt,
    const bntseq_t *bns,
    const uint8_t *pac,
    bseq1_t *seq_,
    int nseq,
    mem_chain_v *chain_ar,
    mem_seed_t *seedBuf,
    int64_t seedBufSize,
    uint8_t* sa_pos,
    uint8_t* ref2sa,
    uint8_t* ref_string,
    mem_tlv* smems_base,
    u64v* hits_base,
    int tid)
{
    int64_t seedBufCount = 0;
    uint64_t tim = __rdtsc();
    int64_t sa_num = bns->l_pac * 2;

    Learned_index_aux_t iaux;
    iaux.sa_pos = sa_pos;
    iaux.ref2sa = ref2sa;
    iaux.bns = bns;
    iaux.pac = pac;
    iaux.ref_string = ref_string;

    int split_len = (int)(opt->min_seed_len * opt->split_factor + .499);
    int split_width = opt->split_width;

    // Process reads in batches of INTERLEAVE_WIDTH
    for (int batch_start = 0; batch_start < nseq; batch_start += INTERLEAVE_WIDTH)
    {
        int batch_n = std::min(INTERLEAVE_WIDTH, nseq - batch_start);
        ReadSeedCtx ctx[INTERLEAVE_WIDTH];

        // Per-read SMEM/hit buffers to avoid cross-read overwrites
        mem_tlv batch_smems[INTERLEAVE_WIDTH];
        u64v batch_hits[INTERLEAVE_WIDTH];
        for (int i = 0; i < batch_n; i++) {
            kv_init(batch_smems[i]);
            kv_init(batch_hits[i]);
        }

        // ====================================================================
        // Phase 0: Prepare all reads in batch (nt4, 2-bit packing)
        // ====================================================================
        for (int i = 0; i < batch_n; i++) {
            int idx = batch_start + i;
            if (seq_[idx].l_seq > LEARNED_MAX_READ_LEN) {
                fprintf(stderr, "Read length %d exceeds LEARNED_MAX_READ_LEN. Recompile.\n", seq_[idx].l_seq);
                exit(EXIT_FAILURE);
            }
            prepare_read(&ctx[i], &seq_[idx], opt, &batch_smems[i], &batch_hits[i]);
            set_forward_pivot(&ctx[i].raux, 0);
        }

        // ====================================================================
        // Phase 1a: Batch first pivot - Tokenize + lookup + prefetch for all reads
        // All reads start at pivot=0, which is always Path C (learned_index_lookup)
        // ====================================================================
        for (int i = 0; i < batch_n; i++) {
            // Skip reads starting with N base
            if (ctx[i].raux.unpacked_queue_buf[0] >= 4) continue;
            ctx[i].pf_position = learned_index_lookup(
                Tokenization(&ctx[i].raux, true, &ctx[i].pf_ambig, ctx[i].hasN),
                &ctx[i].pf_err);
            _mm_prefetch(sa_pos + ctx[i].pf_position * SASIZE - SASIZE, _MM_HINT_T0);
        }

        // ====================================================================
        // Phase 1b: Execute first pivot right_smem_search (SA should be cached)
        // ====================================================================
        for (int i = 0; i < batch_n; i++) {
            if (ctx[i].raux.unpacked_queue_buf[0] >= 4) {
                // N base at pivot 0: skip, let step1 handle it
                continue;
            }

            uint32_t exact_match_len;
            uint64_t pos = right_smem_search(
                ref_string, sa_pos, pac, sa_num,
                &ctx[i].raux, ctx[i].pf_position, ctx[i].pf_err,
                &exact_match_len, ctx[i].smems, ctx[i].hits, &ctx[i].pf_ambig);

#if MEM_TRADEOFF_CACHED && PREFETCH
            if (exact_match_len >= (uint32_t)ctx[i].raux.min_seed_len) {
                uint64_t pv = *(uint32_t*)(sa_pos + pos * SASIZE);
                pv = pv << 8 | sa_pos[pos * SASIZE + 4];
                _mm_prefetch(ref2sa + pv * 5, _MM_HINT_T1);
                _mm_prefetch(ref2sa + (sa_num - pv - exact_match_len - 1) * 5, _MM_HINT_T1);
            }
#endif
            if (exact_match_len == (uint32_t)ctx[i].raux.l_seq) {
                ctx[i].raux.max_l_seq = exact_match_len;
                uint64_t pv = *(uint32_t*)(sa_pos + pos * SASIZE);
                pv = pv << 8 | sa_pos[pos * SASIZE + 4];
                ctx[i].raux.max_refpos = pv;
                ctx[i].raux.max_pivot = 0;
#if PREFETCH
                _mm_prefetch(ref2sa + pv * 5, _MM_HINT_T1);
                _mm_prefetch(ref2sa + (sa_num - pv - ctx[i].raux.l_seq - 1) * 5, _MM_HINT_T1);
#endif
            }
            uint32_t next_pivot = ctx[i].raux.pivot + exact_match_len;
            set_forward_pivot(&ctx[i].raux, next_pivot);

            // Step2 for the first pivot's SMEMs
            int before = 0;
            int after = ctx[i].smems->n;
            for (int k = before; k < after; ++k) {
                int np = ctx[i].raux.pivot;
                int orig_min_intv = ctx[i].raux.min_intv_limit;
                int qbeg = ctx[i].smems->a[k].start;
                int qend = ctx[i].smems->a[k].end;
                if ((qend - qbeg) < split_len || ctx[i].smems->a[k].hitcount > split_width) {
                    set_forward_pivot(&ctx[i].raux, np);
                    continue;
                }
                set_forward_pivot(&ctx[i].raux, (qbeg + qend) >> 1);
                ctx[i].raux.min_intv_limit = ctx[i].smems->a[k].hitcount + 1;
                if (ctx[i].smems->a[k].hitcount >= 1 && ctx[i].smems->a[k].hitcount < 10) {
                    ctx[i].raux.cache_pivot_end = ctx[i].smems->a[k].end;
                    ctx[i].raux.cache_pivot = ctx[i].smems->a[k].start;
                    ctx[i].raux.cache_refpos = ctx[i].smems->a[k].cache_refpos;
                    Learned_getSMEMsOnePosOneThread(&iaux, &ctx[i].raux, ctx[i].smems, ctx[i].hits, ctx[i].hasN, true);
                } else {
                    Learned_getSMEMsOnePosOneThread(&iaux, &ctx[i].raux, ctx[i].smems, ctx[i].hits, ctx[i].hasN, false);
                }
                ctx[i].raux.min_intv_limit = orig_min_intv;
                set_forward_pivot(&ctx[i].raux, np);
            }
        }

        // ====================================================================
        // Phase 1c: Remaining pivots - use original step1+step2 (correct zigzag)
        // Each outer iteration, batch prefetch for all reads' next step1 SA access,
        // then execute step1 sequentially per read.
        // ====================================================================
        bool any_remaining = true;
        while (any_remaining) {
            any_remaining = false;
            int active[INTERLEAVE_WIDTH];
            int n_active = 0;

            // Identify reads that still have pivots to process
            for (int i = 0; i < batch_n; i++) {
                if (ctx[i].raux.pivot >= ctx[i].raux.l_seq) continue;
                active[n_active++] = i;
                any_remaining = true;
            }
            if (!any_remaining) break;

            // Phase A: Prefetch first SA access for each active read's next step1
            for (int j = 0; j < n_active; j++) {
                int i = active[j];
                int pivot = ctx[i].raux.pivot;
                if (ctx[i].raux.unpacked_queue_buf[pivot] >= 4) continue;

                // Determine which path step1 will take
                if (pivot != 0 && ctx[i].raux.unpacked_queue_buf[pivot - 1] < 4) {
                    // Zigzag: step1 starts with LEFT search
#if MEM_TRADEOFF
                    if (ctx[i].raux.max_l_seq == ctx[i].raux.l_seq) {
                        // Path A: ref2sa → SA prefetch
                        uint64_t addr = sa_num - ctx[i].raux.max_refpos - pivot - 1;
                        uint64_t sa_position = *(uint32_t*)(ref2sa + addr * 5);
                        sa_position = sa_position << 8 | ref2sa[addr * 5 + 4];
                        _mm_prefetch(sa_pos + sa_position * SASIZE - SASIZE, _MM_HINT_T0);
                    } else
#endif
                    {
                        // Path C: learned_index_lookup → SA prefetch
                        uint32_t ambig;
                        size_t err;
                        uint64_t key = Tokenization(&ctx[i].raux, false, &ambig, ctx[i].hasN);
                        uint64_t pos = learned_index_lookup(key, &err);
                        _mm_prefetch(sa_pos + pos * SASIZE - SASIZE, _MM_HINT_T0);
                    }
                } else {
                    // First pivot (after N base): Path C
                    uint32_t ambig;
                    size_t err;
                    uint64_t key = Tokenization(&ctx[i].raux, true, &ambig, ctx[i].hasN);
                    uint64_t pos = learned_index_lookup(key, &err);
                    _mm_prefetch(sa_pos + pos * SASIZE - SASIZE, _MM_HINT_T0);
                }
            }

            // Phase B: Execute step1 + step2 for each active read (original functions)
            for (int j = 0; j < n_active; j++) {
                int i = active[j];
                int before = ctx[i].smems->n;
                Learned_getSMEMsOnePosOneThread_step1(&iaux, &ctx[i].raux,
                    ctx[i].smems, ctx[i].hits, ctx[i].hasN);
                int after = ctx[i].smems->n;

                // Step2: re-seeding for long SMEMs
#if MEM_TRADEOFF_CACHED
                for (int k = before; k < after; ++k) {
                    int np = ctx[i].raux.pivot;
                    int orig_min_intv = ctx[i].raux.min_intv_limit;
                    int qbeg = ctx[i].smems->a[k].start;
                    int qend = ctx[i].smems->a[k].end;
                    if ((qend - qbeg) < split_len || ctx[i].smems->a[k].hitcount > split_width) {
                        set_forward_pivot(&ctx[i].raux, np);
                        continue;
                    }
                    set_forward_pivot(&ctx[i].raux, (qbeg + qend) >> 1);
                    ctx[i].raux.min_intv_limit = ctx[i].smems->a[k].hitcount + 1;
                    if (ctx[i].smems->a[k].hitcount >= 1 && ctx[i].smems->a[k].hitcount < 10) {
                        ctx[i].raux.cache_pivot_end = ctx[i].smems->a[k].end;
                        ctx[i].raux.cache_pivot = ctx[i].smems->a[k].start;
                        ctx[i].raux.cache_refpos = ctx[i].smems->a[k].cache_refpos;
                        Learned_getSMEMsOnePosOneThread(&iaux, &ctx[i].raux, ctx[i].smems, ctx[i].hits, ctx[i].hasN, true);
                    } else {
                        Learned_getSMEMsOnePosOneThread(&iaux, &ctx[i].raux, ctx[i].smems, ctx[i].hits, ctx[i].hasN, false);
                    }
                    ctx[i].raux.min_intv_limit = orig_min_intv;
                    set_forward_pivot(&ctx[i].raux, np);
                }
#endif
            }
        }

        // ====================================================================
        // Phase 2: Second seeding + sorting + chaining (per read)
        // ====================================================================
        for (int i = 0; i < batch_n; i++) {
            int idx = batch_start + i;

            if (opt->max_mem_intv > 0) {
                ctx[i].raux.min_intv_limit = opt->max_mem_intv;
                ctx[i].raux.min_seed_len = opt->min_seed_len + 1;
                if (ctx[i].raux.max_l_seq != 0) {
                    Learned_bwtSeedStrategyAllPosOneThread_mem_tradeoff(
                        &iaux, &ctx[i].raux, ctx[i].smems, ctx[i].hits, ctx[i].hasN);
                } else {
                    Learned_bwtSeedStrategyAllPosOneThread(
                        &iaux, &ctx[i].raux, ctx[i].smems, ctx[i].hits, ctx[i].hasN);
                }
            }

            ks_introsort(mem_smem_sort_lt_learned, ctx[i].smems->n, ctx[i].smems->a);

            kv_init(chain_ar[idx]);
            mem_chain_Learned(opt, bns, ctx[i].len,
                              ctx[i].smems, &chain_ar[idx], idx,
                              ctx[i].hits,
                              seedBuf, seedBufSize, seedBufCount,
                              tid);
            mem_chain_v *chn = &chain_ar[idx];
            chn->n = mem_chain_flt(opt, chn->n, chn->a, tid);
            mem_flt_chained_seeds(opt, bns, pac, seq_, chn->n, chn->a);
        }

        // Free per-read SMEM/hit buffers
        for (int i = 0; i < batch_n; i++) {
            free(batch_smems[i].a);
            free(batch_hits[i].a);
        }
    } // end batch loop

    tprof[LEARNED_SEED_CHAIN][tid] += __rdtsc() - tim;
    return 1;
}

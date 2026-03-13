#ifndef LEARNED_INDEX_BATCH_H
#define LEARNED_INDEX_BATCH_H

#include "LearnedIndex_seeding.h"
#include "bwamem.h"

#define INTERLEAVE_WIDTH 8

// Declared in bwamem.cpp
void mem_chain_Learned(const mem_opt_t *opt,
                   const bntseq_t *bns,
                   int len,
                   mem_tlv* smems,
                   mem_chain_v* chain,
                   int seqid,
                   u64v* hits,
                   mem_seed_t *seedBuf,
                   int64_t seedBufSize,
                   int64_t& seedBufCount,
                   int tid);

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
    int tid);

#endif

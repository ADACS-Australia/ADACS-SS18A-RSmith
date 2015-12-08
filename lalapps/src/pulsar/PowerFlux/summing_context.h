#ifndef __SUMMING_CONTEXT_H__
#define __SUMMING_CONTEXT_H__

#include "power_cache.h"

struct S_POWER_SUM;

typedef ALIGN_DECLSPEC struct S_SUMMING_CONTEXT {
	void (*get_uncached_power_sum)(struct S_SUMMING_CONTEXT  *ctx, SEGMENT_INFO *si, int count, PARTIAL_POWER_SUM_F *pps);
	void (*accumulate_power_sum_cached)(struct S_SUMMING_CONTEXT  *ctx, SEGMENT_INFO *si, int count, PARTIAL_POWER_SUM_F *pps);
	void (*accumulate_power_sums)(struct S_SUMMING_CONTEXT *ctx, struct S_POWER_SUM *ps, int count, double gps_start, double gps_stop, int veto_mask);

	int cache_granularity;
	double inv_cache_granularity;
	double half_inv_cache_granularity;

	int diff_shift_granularity;
	double inv_diff_shift_granularity;
	double half_inv_diff_shift_granularity;

	int sidereal_group_count; /* group sfts falling on similar times of the day in this many groups */
	double summing_step; /* process SFTs in blocks of this many seconds each */
	int time_group_count; /* group SFTs by their GPS time within a block into this many groups - used by loosely coherent code */
	
	int cross_terms_present; /* when set to 1 indicates that power_im_pc terms in PARTIAL_POWER_SUM_F is used by the code - for loosely coherent search */

	void *cache;
	void (*free_cache)(struct S_SUMMING_CONTEXT *ctx);
	void (*print_cache_stats)(struct S_SUMMING_CONTEXT *ctx);
	void (*reset_cache)(struct S_SUMMING_CONTEXT *ctx, int segment_count, int template_count);

	void *patch_private_data;
	
	PARTIAL_POWER_SUM_F **partial_power_sum_pool;
	int pps_pool_size;
	int pps_pool_free;
	long pps_hits;
	long pps_misses;
	long pps_rollbacks;

	/* templates memory pool */
	char * power_sums_scratch;
	long power_sums_scratch_size;
	int nchunks;
	int power_sums_idx;

	/* log_extremes private vars */
	PARTIAL_POWER_SUM_F *log_extremes_pps;
	char *log_extremes_pstats_scratch;
	long log_extremes_pstats_scratch_size;

	/* dynamic parameters */
	int loose_first_half_count;
	} SUMMING_CONTEXT;

#include "power_sums.h"

#define PATCH_PRIVATE_SINGLE_BIN_LOOSELY_COHERENT_SIGNATURE 1
#define PATCH_PRIVATE_MATCHED_LOOSELY_COHERENT_SIGNATURE 2

SUMMING_CONTEXT *create_summing_context(void);
void free_summing_context(SUMMING_CONTEXT *ctx);

PARTIAL_POWER_SUM_F * get_partial_power_sum_F(SUMMING_CONTEXT *ctx, int pps_bins, int cross_terms_present);
void put_partial_power_sum_F(SUMMING_CONTEXT *ctx, PARTIAL_POWER_SUM_F *pps);

#endif

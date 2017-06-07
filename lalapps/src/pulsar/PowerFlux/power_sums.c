#include <stdio.h>
#include <stdlib.h>
#include <string.h>
/* We need this define to get NAN values */
// #define __USE_ISOC99
#include <math.h>

#include "global.h"
#include "power_cache.h"
#include "power_sums.h"
#include "dataset.h"
#include "grid.h"
#include "skymarks.h"
#include "summing_context.h"
#include "cmdline.h"

extern struct gengetopt_args_info args_info;

extern SKY_GRID *fine_grid, *patch_grid;
extern SKY_SUPERGRID *super_grid;

extern int first_bin, nbins, useful_bins;
extern SKYMARK *compiled_skymarks;

extern DATASET *datasets;

extern double resolution;

INT64 spindown_start;

void generate_patch_templates(SUMMING_CONTEXT *ctx, int pi, POWER_SUM **ps, int *count)
{
POWER_SUM *p;
int i, j, k, kk, ifmf, ifmd, ifmp, idd;
int skyband;
int fshift_count=args_info.nfshift_arg; /* number of frequency offsets */
long p_size;

float e[GRID_E_COUNT];
float patch_e[GRID_E_COUNT];

*count=0;

p_size=ctx->nchunks*super_grid->max_npatch*args_info.spindown_count_arg*
        args_info.freq_modulation_freq_count_arg*
        args_info.freq_modulation_depth_count_arg*
        args_info.freq_modulation_phase_count_arg*
        args_info.fdotdot_count_arg*
        fshift_count*sizeof(*p);

if(p_size>ctx->power_sums_scratch_size) {
	free(ctx->power_sums_scratch);
	ctx->power_sums_scratch_size=p_size;
	ctx->power_sums_scratch=do_alloc(p_size, 1);
	fprintf(stderr, "Expanded context power sums scratch to %ld\n", p_size);
	}

p=(POWER_SUM *)ctx->power_sums_scratch;
/*p=do_alloc(super_grid->max_npatch*args_info.spindown_count_arg*
	args_info.freq_modulation_freq_count_arg*
	args_info.freq_modulation_depth_count_arg*
	args_info.freq_modulation_phase_count_arg*
	args_info.fdotdot_count_arg*
	fshift_count, sizeof(*p));
*/

*ps=p;

for(ifmf=0;ifmf<args_info.freq_modulation_freq_count_arg;ifmf++)
for(ifmd=0;ifmd<args_info.freq_modulation_depth_count_arg;ifmd++)
for(ifmp=0;ifmp<args_info.freq_modulation_phase_count_arg;ifmp++)
for(idd=0;idd<args_info.fdotdot_count_arg;idd++)
for(i=0;i<args_info.spindown_count_arg;i++) {
	for(kk=super_grid->first_map[pi];kk>=0;kk=super_grid->list_map[kk]) {
		for(k=0;k<GRID_E_COUNT;k++) {
			e[k]=fine_grid->e[k][kk];
			patch_e[k]=patch_grid->e[k][pi];
			}

		/* TODO - this effectively requires skybands to not depend on spindown it would be nice if that was not so */
		if(args_info.fine_grid_skymarks_arg)skyband=fine_grid->band[kk];
			else skyband=mark_sky_point(compiled_skymarks, kk, fine_grid->longitude[kk], fine_grid->latitude[kk], e, args_info.spindown_start_arg+i*args_info.spindown_step_arg);
		if(skyband<0)continue;

		for(j=0;j<fshift_count;j++) {
			p->freq_shift=args_info.frequency_offset_arg+j/(args_info.sft_coherence_time_arg*fshift_count);
			p->spindown=args_info.spindown_start_arg+i*args_info.spindown_step_arg;
			p->fdotdot=args_info.fdotdot_arg+idd*args_info.fdotdot_step_arg;
			p->freq_modulation_freq=args_info.freq_modulation_freq_arg+ifmf*args_info.freq_modulation_freq_step_arg;
			p->freq_modulation_depth=args_info.freq_modulation_depth_arg+ifmd*args_info.freq_modulation_depth_step_arg;
			p->freq_modulation_phase=args_info.freq_modulation_phase_arg+ifmp*args_info.freq_modulation_phase_step_arg;
			p->ra=fine_grid->longitude[kk];
			p->dec=fine_grid->latitude[kk];
			p->min_gps=-1;
			p->max_gps=-1;
			p->patch_ra=patch_grid->longitude[pi];
			p->patch_dec=patch_grid->latitude[pi];

			memcpy(p->e, e, GRID_E_COUNT*sizeof(float));
			memcpy(p->patch_e, patch_e, GRID_E_COUNT*sizeof(float));

			/* TODO - this effectively requires skybands do not depend on spindown it would be nice if that was not so */			
			p->skyband=skyband;

			//p->pps=allocate_partial_power_sum_F(useful_bins, ctx->cross_terms_present);
			p->pps=get_partial_power_sum_F(ctx, useful_bins, ctx->cross_terms_present);
			
			zero_partial_power_sum_F(p->pps);

			(*count)++;
			p++;
			}
		}
	}
}

void generate_followup_templates(SUMMING_CONTEXT *ctx, TEMPLATE_INFO *template_info, int ti_count, POWER_SUM **ps, int *count)
{
POWER_SUM *p;
TEMPLATE_INFO *ti;
int i, j, ti_idx, ifmf, ifmd, ifmp, idd, ira, idec;
int fshift_count=args_info.nfshift_arg; /* number of frequency offsets */
long p_size;

float ra, dec, a, b, x, y;
float e[GRID_E_COUNT];
float patch_e[GRID_E_COUNT];
float e1[3], e2[3];

*count=0;

p_size=ti_count*ctx->nchunks*args_info.binary_template_nsky_arg*args_info.binary_template_nsky_arg*
	args_info.spindown_count_arg*
        args_info.freq_modulation_freq_count_arg*
        args_info.freq_modulation_depth_count_arg*
        args_info.freq_modulation_phase_count_arg*
        args_info.fdotdot_count_arg*
        fshift_count*sizeof(*p);

if(p_size>ctx->power_sums_scratch_size) {
	free(ctx->power_sums_scratch);
	ctx->power_sums_scratch_size=p_size;
	ctx->power_sums_scratch=do_alloc(p_size, 1);
	fprintf(stderr, "Expanded context power sums scratch to %ld\n", p_size);
	}

p=(POWER_SUM *)ctx->power_sums_scratch;
/*p=do_alloc(super_grid->max_npatch*args_info.spindown_count_arg*
	args_info.freq_modulation_freq_count_arg*
	args_info.freq_modulation_depth_count_arg*
	args_info.freq_modulation_phase_count_arg*
	args_info.fdotdot_count_arg*
	fshift_count, sizeof(*p));
*/

*ps=p;


for(ti_idx=0;ti_idx<ti_count;ti_idx++) {

	ti=&(template_info[ti_idx]);

	compute_e_vector(ti->ra, ti->dec, patch_e);

	for(ifmf=0;ifmf<args_info.freq_modulation_freq_count_arg;ifmf++)
	for(ifmd=0;ifmd<args_info.freq_modulation_depth_count_arg;ifmd++)
	for(ifmp=0;ifmp<args_info.freq_modulation_phase_count_arg;ifmp++)
	for(idd=0;idd<args_info.fdotdot_count_arg;idd++)
	for(i=0;i<args_info.spindown_count_arg;i++) {
		for(ira=0;ira<args_info.binary_template_nsky_arg;ira++)
		for(idec=0;idec<args_info.binary_template_nsky_arg;idec++) {
			x=ira-0.5*(args_info.binary_template_nsky_arg-1);
			y=idec-0.5*(args_info.binary_template_nsky_arg-1);
			
			/* Scan disk around template center */
			if(4*(x*x+y*y)>args_info.binary_template_nsky_arg*args_info.binary_template_nsky_arg+1)continue;
			
			
			a=x*resolution;
			b=y*resolution;
			
			/* (0, 0) -> (1, 0, 0) */
			e1[0]=cosf(b)*cosf(a);
			e1[1]=cosf(b)*sinf(a);
			e1[2]=sinf(b);

			/* rotate by DEC around Oy */

			e2[0]=e1[0]*cosf(ti->dec)-e1[2]*sinf(ti->dec);
			e2[1]=e1[1];
			e2[2]=e1[0]*sinf(ti->dec)+e1[2]*cosf(ti->dec);

			/* rotate by RA around 0z */

			e1[0]=e2[0]*cosf(ti->ra)-e2[1]*sinf(ti->ra);
			e1[1]=e2[0]*sinf(ti->ra)+e2[1]*cosf(ti->ra);
			e1[2]=e2[2];


			dec=asinf(e1[2]);
			ra=atan2f(e1[1], e1[0]);
			/* Fixup (0, 0, 1) vector which would produce NaNs */
			if(e1[0]*e1[0]+e1[1]*e1[1]<=0)ra=0.0;

			/* make sure right ascension is positive as in other grids */
			if(ra<0.0)ra+=2*M_PI;
			
			compute_e_vector(ra, dec, e);
			
			for(j=0;j<fshift_count;j++) {
				p->freq_shift=args_info.frequency_offset_arg+j/(args_info.sft_coherence_time_arg*fshift_count);
				p->spindown=ti->spindown+(i-0.5*(args_info.spindown_count_arg-1))*args_info.spindown_step_arg;
				p->fdotdot=ti->fdotdot+(idd-0.5*(args_info.fdotdot_count_arg-1))*args_info.fdotdot_step_arg;
				p->freq_modulation_freq=ti->freq_modulation_freq+(ifmf-0.5*(args_info.freq_modulation_freq_count_arg-1))*args_info.freq_modulation_freq_step_arg;
				p->freq_modulation_depth=ti->freq_modulation_depth+(ifmd-0.5*(args_info.freq_modulation_depth_count_arg-1))*args_info.freq_modulation_depth_step_arg;
				p->freq_modulation_phase=ti->freq_modulation_phase+(ifmp-0.5*(args_info.freq_modulation_phase_count_arg-1))*args_info.freq_modulation_phase_step_arg;
				p->ra=ra;
				p->dec=dec;
				p->min_gps=-1;
				p->max_gps=-1;
				p->patch_ra=ti->ra;
				p->patch_dec=ti->dec;

				memcpy(p->e, e, GRID_E_COUNT*sizeof(float));
				memcpy(p->patch_e, patch_e, GRID_E_COUNT*sizeof(float));

				p->skyband=ti->skyband;

				//p->pps=allocate_partial_power_sum_F(useful_bins, ctx->cross_terms_present);
				p->pps=get_partial_power_sum_F(ctx, useful_bins, ctx->cross_terms_present);
				
				zero_partial_power_sum_F(p->pps);

				(*count)++;
				p++;
				}
			}
		}
	}
}

void clone_templates(SUMMING_CONTEXT *ctx, POWER_SUM *ps, int count, POWER_SUM **ps_out)
{
int i, k;
//*ps_out=do_alloc(count, sizeof(POWER_SUM));
*ps_out=&(((POWER_SUM *)(ctx->power_sums_scratch))[count*ctx->power_sums_idx]);

for(i=0;i<count;i++) {
	(*ps_out)[i].freq_shift=ps[i].freq_shift;
	(*ps_out)[i].spindown=ps[i].spindown;
	(*ps_out)[i].fdotdot=ps[i].fdotdot;
	(*ps_out)[i].freq_modulation_freq=ps[i].freq_modulation_freq;
	(*ps_out)[i].freq_modulation_depth=ps[i].freq_modulation_depth;
	(*ps_out)[i].freq_modulation_phase=ps[i].freq_modulation_phase;
	(*ps_out)[i].ra=ps[i].ra;
	(*ps_out)[i].dec=ps[i].dec;
	(*ps_out)[i].min_gps=ps[i].min_gps;
	(*ps_out)[i].max_gps=ps[i].max_gps;
	(*ps_out)[i].patch_ra=ps[i].patch_ra;
	(*ps_out)[i].patch_dec=ps[i].patch_dec;

	for(k=0;k<GRID_E_COUNT;k++) {
		(*ps_out)[i].e[k]=ps[i].e[k];
		(*ps_out)[i].patch_e[k]=ps[i].patch_e[k];
		}

	(*ps_out)[i].skyband=ps[i].skyband;

	//(*ps_out)[i].pps=allocate_partial_power_sum_F(useful_bins, ctx->cross_terms_present);
	(*ps_out)[i].pps=get_partial_power_sum_F(ctx, useful_bins, ctx->cross_terms_present);
	zero_partial_power_sum_F((*ps_out)[i].pps);
	}
}

void free_templates(POWER_SUM *ps, int count)
{
int i;
for(i=0;i<count;i++) {
	free_partial_power_sum_F(ps[i].pps);
	ps[i].pps=NULL;
	}
free(ps);
}

void free_templates_ctx(SUMMING_CONTEXT *ctx, POWER_SUM *ps, int count)
{
int i;
for(i=0;i<count;i++) {
	put_partial_power_sum_F(ctx, ps[i].pps);
	ps[i].pps=NULL;
	}
//free(ps);
}

void accumulate_power_sums_sidereal_step(SUMMING_CONTEXT *ctx, POWER_SUM *ps, int count, double gps_start, double gps_stop, int veto_mask)
{
int segment_count;
SEGMENT_INFO *si, *si_local;
POWER_SUM *ps_local;
DATASET *d;
POLARIZATION *pl;
int gps_step=ctx->summing_step;
int i, j, k;
//float min_shift, max_shift;
float a;
double gps_idx, gps_idx_next;
float center_frequency=(first_bin+nbins*0.5);
float mid_t;
int group_count=ctx->sidereal_group_count;
SEGMENT_INFO **groups;
int *group_segment_count;
float avg_spindown=args_info.spindown_start_arg+0.5*args_info.spindown_step_arg*(args_info.spindown_count_arg-1);
float avg_fdotdot=args_info.fdotdot_arg+0.5*args_info.fdotdot_step_arg*(args_info.fdotdot_count_arg-1);
double fmodomega_t;

float *patch_e=ps[0].patch_e; /* set of coefficients for this patch, used for amplitude response and bin shift estimation */

//fprintf(stderr, "%p %p %d %lf %lf 0x%08x\n", ctx, ps, count, gps_start, gps_stop, veto_mask);

for(gps_idx=gps_start; gps_idx<gps_stop; gps_idx+=gps_step) {

	gps_idx_next=gps_idx+gps_step;
	si=find_segments(gps_idx, (gps_idx_next<=gps_stop ? gps_idx_next : gps_stop), veto_mask, &segment_count);
	if(segment_count<1) {
		free(si);
		continue;
		}

	/* This assumes that we are patch bound !! *** WARNING ***
	TODO: make this assumption automatic in the data layout.
	 */
	si_local=si;
// 	min_shift=1000000; /* we should never load this many bins */
// 	max_shift=-1000000;
	for(j=0;j<segment_count;j++) {

		d=&(datasets[si_local->dataset]);
		pl=&(d->polarizations[0]);

		si_local->f_plus=F_plus_coeff(si_local->segment,  patch_e, d->AM_coeffs_plus);
		si_local->f_cross=F_plus_coeff(si_local->segment,  patch_e, d->AM_coeffs_cross);

#if 0
		a=center_frequency*(float)args_info.doppler_multiplier_arg*(patch_e[0]*si_local->detector_velocity[0]
						+patch_e[1]*si_local->detector_velocity[1]
						+patch_e[2]*si_local->detector_velocity[2])
			+si_local->coherence_time*(avg_spindown+0.5*avg_fdotdot*(float)(si_local->gps-spindown_start))*(float)(si_local->gps-spindown_start);
		/* This computation involves doubles and trigonometric functions. Avoid it if there is no modulation */
		if(ps_local->freq_modulation_freq>0) {
			fmodomega_t=(si_local->gps-spindown_start+0.5*si_local->coherence_time)*ps_local->freq_modulation_freq;
			fmodomega_t=fmodomega_t-floor(fmodomega_t);
								
			a+=si_local->coherence_time*ps_local->freq_modulation_depth*cosf(2.0*M_PI*fmodomega_t+ps_local->freq_modulation_phase)*(1.0+(float)args_info.doppler_multiplier_arg*(ps_local->e[0]*si_local->detector_velocity[0]
				+ps_local->e[1]*si_local->detector_velocity[1]
				+ps_local->e[2]*si_local->detector_velocity[2]));
			}
		if(a<min_shift)min_shift=a;
		if(a>max_shift)max_shift=a;
#endif
		si_local++;
		}

	if(group_count>200) {
		fprintf(stderr, "Warning group count too large: %d\n", group_count);
		group_count=200;
		}

	group_segment_count=do_alloc(group_count, sizeof(*group_segment_count));
	groups=do_alloc(group_count, sizeof(*groups));

	for(k=0;k<group_count;k++) {
		group_segment_count[k]=0;
		groups[k]=do_alloc(segment_count, sizeof(SEGMENT_INFO));
		}

	/* group segments into bunches with similar shifts - mostly by sidereal time
           this way there is larger correllation of frequency shifts during summing and better use of power cache */
	si_local=si;
	for(j=0;j<segment_count;j++) {
		a=(center_frequency*(float)args_info.doppler_multiplier_arg*(patch_e[0]*si_local->detector_velocity[0]
						+patch_e[1]*si_local->detector_velocity[1]
						+patch_e[2]*si_local->detector_velocity[2])
			+si_local->coherence_time*(avg_spindown+0.5*avg_fdotdot*(float)(si_local->gps-spindown_start))*(float)(si_local->gps-spindown_start));
		/* This computation involves doubles and trigonometric functions. Avoid it if there is no modulation */
		/* Approximate */
		ps_local=ps;
		if(ps_local->freq_modulation_freq>0) {
			fmodomega_t=(si_local->gps-spindown_start+0.5*si_local->coherence_time)*ps_local->freq_modulation_freq;
			fmodomega_t=fmodomega_t-floor(fmodomega_t);
								
			a+=si_local->coherence_time*ps_local->freq_modulation_depth*cosf(2.0*M_PI*fmodomega_t+ps_local->freq_modulation_phase)*(1.0+(float)args_info.doppler_multiplier_arg*(ps_local->e[0]*si_local->detector_velocity[0]
				+ps_local->e[1]*si_local->detector_velocity[1]
				+ps_local->e[2]*si_local->detector_velocity[2]));
			}
		//a*=0.25;
		k=floorf((a-floorf(a))*group_count);
		if(k<0)k=0;
		if(k>=group_count)k=group_count-1;

		memcpy(&(groups[k][group_segment_count[k]]), si_local, sizeof(SEGMENT_INFO));
		group_segment_count[k]++;
		
		si_local++;
		}

// 	for(k=0;k<GROUP_COUNT;k++) {
// 		fprintf(stderr, "group %d has %d segments\n", k, group_segment_count[k]);
// 		}

	/* loop over groups */

	for(k=0;k<group_count;k++) {
 		//fprintf(stderr, "group %d has %d segments\n", k, group_segment_count[k]);
		if(group_segment_count[k]<1)continue;
		ctx->reset_cache(ctx, group_segment_count[k], count);
	
		/* loop over templates */
		ps_local=ps;
		for(i=0;i<count;i++) {
			/* fill in segment info appropriate to this template */
			si_local=groups[k];
			for(j=0;j<group_segment_count[k];j++) {	
				
				mid_t=(float)(si_local->gps-spindown_start);

				si_local->bin_shift=si_local->coherence_time*(ps_local->freq_shift+ps_local->spindown*mid_t+0.5*ps_local->fdotdot*mid_t*mid_t)+
					center_frequency*(float)args_info.doppler_multiplier_arg*(ps_local->e[0]*si_local->detector_velocity[0]
						+ps_local->e[1]*si_local->detector_velocity[1]
						+ps_local->e[2]*si_local->detector_velocity[2]);
					
				/* This computation involves doubles and trigonometric functions. Avoid it if there is no modulation */
				if(ps_local->freq_modulation_freq>0) {
					fmodomega_t=(si_local->gps-spindown_start+0.5*si_local->coherence_time)*ps_local->freq_modulation_freq;
					fmodomega_t=fmodomega_t-floor(fmodomega_t);
										
					si_local->bin_shift+=si_local->coherence_time*ps_local->freq_modulation_depth*cosf(2.0*M_PI*fmodomega_t+ps_local->freq_modulation_phase)*(1.0+(float)args_info.doppler_multiplier_arg*(ps_local->e[0]*si_local->detector_velocity[0]
						+ps_local->e[1]*si_local->detector_velocity[1]
						+ps_local->e[2]*si_local->detector_velocity[2]));
					}
				si_local++;
				}
	
			ctx->accumulate_power_sum_cached(ctx, groups[k], group_segment_count[k], ps_local->pps);
			ps_local++;
			}
		}
	for(k=0;k<group_count;k++) {
		free(groups[k]);
		}
	free(groups);
	free(group_segment_count);
	free(si);
	}

for(i=0;i<count;i++) {
	if(ps[i].min_gps<0 || ps[i].min_gps>gps_start)ps[i].min_gps=gps_start;
	if(ps[i].max_gps<0 || ps[i].max_gps<gps_stop)ps[i].max_gps=gps_stop;
	}
}

void accumulate_power_sums_plain(SUMMING_CONTEXT *ctx, POWER_SUM *ps, int count, double gps_start, double gps_stop, int veto_mask)
{
int segment_count;
SEGMENT_INFO *si, *si_local;
POWER_SUM *ps_local;
DATASET *d;
POLARIZATION *pl;
int gps_step=ctx->summing_step;
int i, j;
//float min_shift, max_shift;
float a;
float mid_t;
double gps_idx, gps_idx_next;
float center_frequency=(first_bin+nbins*0.5);
float avg_spindown=args_info.spindown_start_arg+0.5*args_info.spindown_step_arg*(args_info.spindown_count_arg-1);

float *patch_e=ps[0].patch_e; /* set of coefficients for this patch, used for amplitude response and bin shift estimation */
double fmodomega_t;

//fprintf(stderr, "%p %p %d %lf %lf 0x%08x\n", ctx, ps, count, gps_start, gps_stop, veto_mask);

for(gps_idx=gps_start; gps_idx<gps_stop; gps_idx+=gps_step) {

	gps_idx_next=gps_idx+gps_step;
	si=find_segments(gps_idx, (gps_idx_next<=gps_stop ? gps_idx_next : gps_stop), veto_mask, &segment_count);
	if(segment_count<1) {
		free(si);
		continue;
		}

	/* This assumes that we are patch bound !! *** WARNING ***
	TODO: make this assumption automatic in the data layout.
	 */
	si_local=si;
// 	min_shift=1000000; /* we should never load this many bins */
// 	max_shift=-1000000;
	for(j=0;j<segment_count;j++) {

		d=&(datasets[si_local->dataset]);
		pl=&(d->polarizations[0]);

		si_local->f_plus=F_plus_coeff(si_local->segment,  patch_e, d->AM_coeffs_plus);
		si_local->f_cross=F_plus_coeff(si_local->segment,  patch_e, d->AM_coeffs_cross);


#if 0
		a=center_frequency*(float)args_info.doppler_multiplier_arg*(patch_e[0]*si_local->detector_velocity[0]
						+patch_e[1]*si_local->detector_velocity[1]
						+patch_e[2]*si_local->detector_velocity[2])
			+si_local->coherence_time*(avg_spindown+0.5*args_info.fdotdot_arg*(float)(si_local->gps-spindown_start))*(float)(si_local->gps-spindown_start);
		if(a<min_shift)min_shift=a;
		if(a>max_shift)max_shift=a;
#endif
		si_local++;
		}

	/* loop over groups */

	ctx->reset_cache(ctx, segment_count, count);
	
		/* loop over templates */
		ps_local=ps;
		for(i=0;i<count;i++) {
			/* fill in segment info appropriate to this template */
			si_local=si;
			for(j=0;j<segment_count;j++) {
	
				mid_t=(float)(si_local->gps-spindown_start);
				si_local->bin_shift=si_local->coherence_time*(ps_local->freq_shift+ps_local->spindown*mid_t+0.5*(float)args_info.fdotdot_arg*mid_t*mid_t)+
					center_frequency*(float)args_info.doppler_multiplier_arg*(ps_local->e[0]*si_local->detector_velocity[0]
						+ps_local->e[1]*si_local->detector_velocity[1]
						+ps_local->e[2]*si_local->detector_velocity[2]);

				/* This computation involves doubles and trigonometric functions. Avoid it if there is no modulation */
				if(ps_local->freq_modulation_freq>0) {
					fmodomega_t=(si_local->gps-spindown_start)*ps_local->freq_modulation_freq;

					si_local->bin_shift+=si_local->coherence_time*ps_local->freq_modulation_depth*cos(2.0*M_PI*(fmodomega_t-floor(fmodomega_t))+ps_local->freq_modulation_phase)*(1.0+(float)args_info.doppler_multiplier_arg*(ps_local->e[0]*si_local->detector_velocity[0]
						+ps_local->e[1]*si_local->detector_velocity[1]
						+ps_local->e[2]*si_local->detector_velocity[2]));
					}

				si_local++;
				}
	
			ctx->accumulate_power_sum_cached(ctx, si, segment_count, ps_local->pps);
			ps_local++;
			}
	free(si);
	}

for(i=0;i<count;i++) {
	if(ps[i].min_gps<0 || ps[i].min_gps>gps_start)ps[i].min_gps=gps_start;
	if(ps[i].max_gps<0 || ps[i].max_gps<gps_stop)ps[i].max_gps=gps_stop;
	}
}


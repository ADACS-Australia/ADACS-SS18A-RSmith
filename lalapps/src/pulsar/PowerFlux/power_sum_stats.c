#include <stdio.h>
#include <stdlib.h>
#include <string.h>
/* We need this define to get NAN values */
//#define __USE_ISOC99
#include <math.h>
#include <gsl/gsl_cblas.h>
#include <gsl/gsl_cdf.h>
#include <xmmintrin.h>

#include "global.h"
#include "power_cache.h"
#include "power_sums.h"
#include "power_sum_stats.h"
#include "dataset.h"
#include "statistics.h"
#include "grid.h"
#include "cmdline.h"

extern struct gengetopt_args_info args_info;
extern int nbins, first_bin, side_cut, useful_bins;

double upper_limit_comp=NAN, strain_comp=NAN; 

ALIGNMENT_COEFFS *alignment_grid=NULL;
int alignment_grid_free=0;
int alignment_grid_size=0;

extern FILE *LOG;

/* Include file with Feldman-Cousins upper limits directly
   so as to benefit from function inlining */

#include "fc.c"

/* this is identical to the one above, but follows the formula in polarization.pdf */

void compute_alignment_coeffs(ALIGNMENT_COEFFS *ac)
{
double a, a_plus_sq, a_cross_sq, cpsi, spsi, asum, adiff;
a=cos(ac->iota);
a=a*a;

a_plus_sq=(1+a)*0.5;
a_plus_sq=a_plus_sq*a_plus_sq;
a_cross_sq=a;

cpsi=cos(4*ac->psi);
spsi=sin(4*ac->psi);

asum=0.25*(a_plus_sq+a_cross_sq);
adiff=0.25*(a_plus_sq-a_cross_sq);

ac->pp=(asum+adiff*cpsi);
ac->pc=2*adiff*spsi;
ac->cc=(asum-adiff*cpsi);
ac->im_pc=0.125*(1+a)*cos(ac->iota)*2;

ac->pppp=ac->pp*ac->pp;
ac->pppc=2*ac->pp*ac->pc;
ac->ppcc=2*ac->pp*ac->cc+ac->pc*ac->pc;
ac->pccc=2*ac->pc*ac->cc;
ac->cccc=ac->cc*ac->cc;
ac->im_ppcc=ac->im_pc*ac->im_pc;
}

void generate_alignment_grid(void)
{
int i, j, k;
int npsi=args_info.npsi_arg, niota=args_info.niota_arg;

if(args_info.compute_cross_terms_arg)
	alignment_grid_size=npsi*(2*niota-1)+2;
	else
	alignment_grid_size=npsi*niota+1;
alignment_grid_free=alignment_grid_size;
alignment_grid=do_alloc(alignment_grid_size, sizeof(*alignment_grid));

/* First polarization is circular one */
alignment_grid[0].psi=0.0;
alignment_grid[0].iota=0.0;

k=1;
for(j=0;j<niota;j++)
	for(i=0;i<npsi;i++) {
		/* note: make better grid by not overcovering circular polarization neighbourhood */
		alignment_grid[k].psi=(0.5*M_PI*(i+0.5*(j&1)))/npsi;
		alignment_grid[k].iota=acos((1.0*j)/niota); 
		k++;
		}
		
if(args_info.compute_cross_terms_arg) {
	alignment_grid[k].psi=0.0;
	alignment_grid[k].iota=M_PI;
	k++;

	for(j=1;j<niota;j++)
		for(i=0;i<npsi;i++) {
			/* note: make better grid by not overcovering circular polarization neighbourhood */
			alignment_grid[k].psi=(0.5*M_PI*(i+0.5*(j&1)))/npsi;
			alignment_grid[k].iota=acos(-(1.0*j)/niota); 
			k++;
			}	
	}
	
if(k-1>alignment_grid_free) {
	fprintf(stderr, "*** INTERNAL ERROR: insuffient space allocated for alignment coefficients array (%d vs %d)\n", k, alignment_grid_free);
	exit(-1);
	}

for(k=0;k<alignment_grid_free;k++) {
	fprintf(LOG, "alignment entry %d: %f %f\n", k, alignment_grid[k].iota, alignment_grid[k].psi);
	compute_alignment_coeffs(&(alignment_grid[k]));
	}
fprintf(stderr, "Alignment grid size: %d\n", alignment_grid_free);
fprintf(LOG, "Alignment grid size: %d\n", alignment_grid_free);
}

/* an implementation of merge sort - this modifies input array */
void merge_sort_floats2(float *data, int count, int step)
{
int i, k, touched;
float a,b;
/* handcode cases of small number of items */
k=step;
if(k>=count)return;
k+=step;
if(k>=count) {
	a=data[0];
	b=data[step];
	if(a>b) {
		data[0]=b;
		data[step]=a;
		}
	return;
	}
merge_sort_floats2(data, count, step<<1);
merge_sort_floats2(data+step, count-step, step<<1);
touched=1;
/*while(touched) {
	touched=0;
	a=data[0];
	for(i=step;i<count;i+=step) {
		b=data[i];
		if(a>b) {
			data[i-step]=b;
			data[i]=a;
			touched=1;
			} else {
			a=b;
			}
		}
	}*/
while(touched) {
	touched=0;
	for(i=0;i<count-step;i+=step<<1) {
		a=data[i];
		b=data[i+step];
		if(a>b) {
			data[i]=b;
			data[i+step]=a;
			touched=1;
			}
		}
	for(i=step;i<count-step;i+=step<<1) {
		a=data[i];
		b=data[i+step];
		if(a>b) {
			data[i]=b;
			data[i+step]=a;
			touched=1;
			}
		}
	}
}

/* an implementation of merge sort - this modifies input array */
void merge_sort_floats(float *data, int count)
{
int i, j, k, m;
float a,b;
int step;
float *tmp;
/* handcode cases of small number of items */
if(count<=1)return;

a=data[0];
b=data[1];
if(a>b) {
	data[0]=b;
	data[1]=a;
	}
step=2;

tmp=alloca(count*sizeof(*tmp));
while(step<count) {
	k=step*2;
	if(k>count) {
		k=count;
		}
	merge_sort_floats(&(data[step]), k-step);

	a=data[0];
	b=data[step];
	for(i=0,j=step, m=0;(i<step) && (j<k);m++) {
		if(a<b) {
			tmp[m]=a;
			i++;
			a=data[i];
			} else {
			tmp[m]=b;
			j++;
			b=data[j];
			}
		}
	for(;i<step;i++,m++) tmp[m]=data[i];
	for(;j<k;j++,m++) tmp[m]=data[j];
	memcpy(data, tmp, k*sizeof(*tmp));
	step=k;
	}

}

/* an implementation of quick sort - this modifies input array */
void quick_sort_floats1(float *data, int count)
{
int i, j;
float a,b;
/* handcode cases of small number of items */
if(count<=1)return;
a=data[0];
if(count==2) {
	b=data[1];
	if(a>b) {
		data[0]=b;
		data[1]=a;
		}
	return;
	}

/* we do an average so that qsort performs well in the case of sloped array */
/*	for(i=0;i<count;i++)fprintf(stderr, "%g ", data[i]);
	fprintf(stderr, "\n");*/
i=0;
j=count-1;
//a=0.5*(a+data[j]);
while(i<j) {
	if(data[i]<=a) {
		i++;
		continue;
		}
	if(data[j]>=a) {
		j--;
		continue;
		}
	b=data[i];
	data[i]=data[j];
	data[j]=b;
	i++;
	j--;
	}
if(i==j) {
	if(data[j]>a)j--;
	}
if(data[j]<a)j++;
if(j==count) {
	b=data[0];
	data[0]=data[count-1];
	data[count-1]=b;
	quick_sort_floats1(data, count-1);
	return;
	}
if(j==0) {
	quick_sort_floats1(data+1, count-1);
	return;
	}
/*if(!j || j==count) {
	fprintf(stderr, "*** INTERNAL ERROR: recursion on quick sort count=%d j=%d a=%f\n", count, j, a);
	for(i=0;i<count;i++)fprintf(stderr, "%g ", data[i]);
	fprintf(stderr, "\n");
	exit(-1);
	}*/
quick_sort_floats1(data, j);
quick_sort_floats1(&(data[j]), count-j);
}

/* manually sort arrays of up to 3 floats */
static inline void manual_sort_floats(float *data, int count)
{
float a,b,c;
if(count<=1)return;
a=data[0];
if(count==2) {
	b=data[1];
	if(a>b) {
		data[0]=b;
		data[1]=a;
		}
	return;
	}
if(count==3) {
	b=data[1];
	if(a>b) {
		b=a;
		a=data[1];
		}
	c=data[2];
	if(c<b) {
		if(c<a) {
			c=b;
			b=a;
			a=data[2];
			} else {
			c=b;
			b=data[2];
			}
		}
	data[0]=a;
	data[1]=b;
	data[2]=c;
	return;
	}
fprintf(stderr, "*** ERROR: manual_sort_floats cannot sort array this large %p %d\n", data, count);
exit(-1);
return;
}

static inline int partition_floats(float *data, int count)
{
float a,b,c;
float *first=data;
float *last=&(data[count-1]);

a=*first;
first++;
//a=0.5*(a+data[j]);
while(first<last) {
	b=*first;
	if(b<=a) {
		first++;
		continue;
		}
	c=*last;
	if(c>=a) {
		last--;
		continue;
		}
	*first=c;
	*last=b;
	first++;
	last--;
	}
if(first==last) {
	if(*last>a)last--;
	}
if(*last<a)last++;
return (last-data);
}

/* an implementation of quick sort - this modifies input array */
void quick_sort_floats(float *data, int count)
{
int j;
float b;
int stack_len=0;

while(1) {
	if(count<4) {
		manual_sort_floats(data, count);
		return;
		} else {
		j=partition_floats(data, count);
		}
	if(j<0) return;
	if(j==count) {
		b=data[0];
		data[0]=data[count-1];
		data[count-1]=b;
		count--;
		continue;
		}
	if(j==0) {
		data++;
		count--;
		continue;
		}
/*if(!j || j==count) {
	fprintf(stderr, "*** INTERNAL ERROR: recursion on quick sort count=%d j=%d a=%f\n", count, j, a);
	for(i=0;i<count;i++)fprintf(stderr, "%g ", data[i]);
	fprintf(stderr, "\n");
	exit(-1);
	}*/
	if(j<4)manual_sort_floats(data, j);
		else quick_sort_floats(data, j);
	stack_len++;
	data+=j;
	count-=j;
	continue;
	}
}

void bucket_sort_floats(float *data, int count)
{
float *tmp[4];
int tmp_count[4];
float a,b,c, mult;
int i, k, m;
if(count<=16) {
	merge_sort_floats2(data, count, 1);
	return;
	}

// if(count<=2) {
// 	if(count<=1)return;
// 	a=data[0];
// 	b=data[1];
// 	if(a>b) {
// 		data[0]=b;
// 		data[1]=a;
// 		}
// 	return;
// 	}

for(k=0;k<4;k++) {
	tmp[k]=alloca(count*sizeof(**tmp));
	tmp_count[k]=0;
	}

a=data[0];
b=a;
for(i=1;i<count;i++) {
	c=data[i];
	if(c>b)b=c;
		else
	if(c<a)a=c;
	}
if(b<=a)b=a+1;
mult=4.0/(b-a);

for(i=0;i<count;i++) {
	c=data[i];
	k=((c-a)*mult);
	if(k>3)k=3;
	if(k<0)k=0;
	tmp[k][tmp_count[k]]=c;
	tmp_count[k]++;
	}
i=0;
for(k=0;k<4;k++) {
	if(tmp_count[k]==count) {
		merge_sort_floats2(data, count, 1);
		return;
		}
	bucket_sort_floats(tmp[k], tmp_count[k]);
	for(m=0;m<tmp_count[k];m++) {
		data[i]=tmp[k][m];
		i++;
		}
	}
}

int is_sorted(float *data, int count)
{
int i;
float a,b;
a=data[0];
for(i=1;i<count;i++) {
	b=data[i];
	if(a>b)return 0;
	a=b;
	}
return 1;
}

void set_missing_point_stats(POINT_STATS *pst)
{
pst->bin=0;
pst->iota=-1;
pst->psi=-1;

pst->S=-1;
pst->M=-1;

pst->ul=-1;
pst->ll=-1;
pst->centroid=-1;
pst->snr=-1;

pst->max_weight=-1;
pst->weight_loss_fraction=1;
pst->ks_value=-1;
pst->ks_count=-1;
}

int compare_point_stats(char *prefix, POINT_STATS *ref, POINT_STATS *test)
{

#define TEST(field, format, tolerance) \
	if( (ref->field!=test->field) && !(fabs(test->field-ref->field)<tolerance*(fabs(ref->field)+fabs(test->field)))) { \
		fprintf(stderr, "%s" #field " fields do not match ref=" format " test=" format " test-ref=" format "\n", \
			prefix, \
			ref->field, test->field, test->field-ref->field); \
		return -1; \
		}

TEST(iota, "%g", 1e-4)
TEST(psi, "%g", 1e-4)

TEST(ul, "%g", 1e-4)
TEST(ll, "%g", 1e-4)
TEST(centroid, "%g", 1e-4)
TEST(snr, "%g", 1e-4)

TEST(M, "%g", 1e-4)
TEST(S, "%g", 1e-4)
TEST(ks_value, "%g", 1e-4)
TEST(m1_neg, "%g", 1e-4)
TEST(m3_neg, "%g", 1e-4)
TEST(m4, "%g", 1e-4)
TEST(max_weight, "%g", 1e-4)
TEST(weight_loss_fraction, "%g", 1e-4)

TEST(ks_count, "%d", -1)

TEST(bin, "%d", -1)

/* the following fields are for convenience and are filled in by outside code based on value of bin */
TEST(frequency, "%g", 1e-4)
TEST(spindown, "%g", 1e-4)
TEST(ra, "%g", 1e-4)
TEST(dec, "%g", 1e-4)


#undef TEST

return 0;
}

/* Special version for testing universal statistics 
 * 
 * Ignore m1_neg field (used for diagnostics) as it can jump due to floating point errors.
 * 
 */
int compare_point_stats_universal(char *prefix, POINT_STATS *ref, POINT_STATS *test)
{

#define TEST(field, format, tolerance) \
	if( (ref->field!=test->field) && !(fabs(test->field-ref->field)<tolerance*(fabs(ref->field)+fabs(test->field)))) { \
		fprintf(stderr, "%s" #field " fields do not match ref=" format " test=" format " test-ref=" format "\n", \
			prefix, \
			ref->field, test->field, test->field-ref->field); \
		return -1; \
		}

TEST(iota, "%g", 1e-4)
TEST(psi, "%g", 1e-4)

TEST(ul, "%g", 1e-4)
TEST(ll, "%g", 1e-4)
TEST(centroid, "%g", 1e-4)
TEST(snr, "%g", 1e-4)

TEST(M, "%g", 1e-4)
TEST(S, "%g", 1e-4)
TEST(ks_value, "%g", 1e-4)
TEST(m3_neg, "%g", 1e-4)
TEST(m4, "%g", 1e-4)
TEST(max_weight, "%g", 1e-4)
TEST(weight_loss_fraction, "%g", 1e-4)

TEST(ks_count, "%d", -1)

TEST(bin, "%d", -1)

/* the following fields are for convenience and are filled in by outside code based on value of bin */
TEST(frequency, "%g", 1e-4)
TEST(spindown, "%g", 1e-4)
TEST(ra, "%g", 1e-4)
TEST(dec, "%g", 1e-4)


#undef TEST

return 0;
}


void point_power_sum_stats_sorted(PARTIAL_POWER_SUM_F *pps, ALIGNMENT_COEFFS *ag, POINT_STATS *pst)
{
int i;
float M, S, a, inv_S, inv_weight;
float *tmp=NULL;
NORMAL_STATS nstats;
float max_dx;
int max_dx_bin;
float weight, min_weight, max_weight;

/* allocate on stack, for speed */
tmp=aligned_alloca(useful_bins*sizeof(*tmp));

memset(&nstats, 0, sizeof(nstats));

/* sort to compute robust estimates */
nstats.flag= STAT_FLAG_ESTIMATE_MEAN
	| STAT_FLAG_ESTIMATE_SIGMA;

if(args_info.ks_test_arg){
	nstats.flag|=STAT_FLAG_ESTIMATE_KS_LEVEL
		| STAT_FLAG_COMPUTE_KS_TEST;
	}

if(pps->power_im_pc==NULL) {
	fprintf(stderr, "*** INTERNAL ERROR: %s requires pps->power_im_pc!=NULL\n", __FUNCTION__);
	exit(-1);
	}

if(pps->weight_arrays_non_zero) {
	max_weight=0;
	min_weight=1e50;

	if(!pps->collapsed_weight_arrays) {
		for(i=0;i<useful_bins;i++) {
			pps->weight_pppp[i]+=pps->c_weight_pppp;
			pps->weight_pppc[i]+=pps->c_weight_pppc;
			pps->weight_ppcc[i]+=pps->c_weight_ppcc;
			pps->weight_pccc[i]+=pps->c_weight_pccc;
			pps->weight_cccc[i]+=pps->c_weight_cccc;
			}
		pps->c_weight_pppp=0;
		pps->c_weight_pppc=0;
		pps->c_weight_ppcc=0;
		pps->c_weight_pccc=0;
		pps->c_weight_cccc=0;
		pps->collapsed_weight_arrays=1;
		}

	for(i=0;i<useful_bins;i++) {
		weight=(pps->weight_pppp[i]*ag->pppp+
			pps->weight_pppc[i]*ag->pppc+
			pps->weight_ppcc[i]*ag->ppcc+
			pps->weight_pccc[i]*ag->pccc+
			pps->weight_cccc[i]*ag->cccc);
	
		if(weight>max_weight)max_weight=weight;
		if(weight<min_weight)min_weight=weight;

		tmp[i]=(pps->power_pp[i]*ag->pp+pps->power_pc[i]*ag->pc+pps->power_cc[i]*ag->cc+pps->power_im_pc[i]*ag->im_pc)/weight;
		}
	} else {
	weight=(pps->c_weight_pppp*ag->pppp+
		pps->c_weight_pppc*ag->pppc+
		pps->c_weight_ppcc*ag->ppcc+
		pps->c_weight_pccc*ag->pccc+
		pps->c_weight_cccc*ag->cccc+
		pps->c_weight_im_ppcc*ag->im_ppcc);
	max_weight=weight;
	min_weight=weight;

	inv_weight=1.0/weight;

	for(i=0;i<useful_bins;i++) {
		tmp[i]=((float)pps->power_pp[i]*ag->pp+(float)pps->power_pc[i]*ag->pc+(float)pps->power_cc[i]*ag->cc+(float)pps->power_im_pc[i]*ag->im_pc)*inv_weight;
		}
	}

/* 0 weight can happen due to extreme line veto at low frequencies and small spindowns */
if(min_weight<= args_info.small_weight_ratio_arg*max_weight) {
	set_missing_point_stats(pst);
	return;
	}
	
/* find highest bin */
max_dx=tmp[0];
max_dx_bin=0;

for(i=1;i<useful_bins;i++){
	a=tmp[i];
	if(a>max_dx) {
		max_dx=a;
		max_dx_bin=i;
		}
	}

/* sort to compute statistics */
/*merge_sort_floats(tmp, useful_bins, 1);*/
/*bucket_sort_floats(tmp, useful_bins);*/
/* merge_sort_floats(tmp, useful_bins); */
quick_sort_floats(tmp, useful_bins);

if(!is_sorted(tmp, useful_bins)) {
	fprintf(stderr, "Internal error: incorrectly sorted array\n");
	for(i=0;i<useful_bins;i++)fprintf(stderr, " %f", tmp[i]);
	fprintf(stderr, "\n");
	exit(-1);
	}

compute_normal_stats(tmp, useful_bins, &nstats);

M=nstats.mean;
S=nstats.sigma;

inv_S=1.0/S;

/* convert to SNR from the highest power */
max_dx=(max_dx-M)*inv_S;

if(max_dx<=0) {
	/* In theory we could have max_dx=0 because the distribution is flat, but we really should not have this */
	fprintf(stderr, "***ERROR - max_dx<=0  max_dx=%g max_dx_bin=%d M=%g S=%g inv_S=%g\n",
			max_dx,
			max_dx_bin,
			M,
			S,
			inv_S);
	}

pst->bin=max_dx_bin;
pst->iota=ag->iota;
pst->psi=ag->psi;

/* convert to upper limit units */
pst->S=sqrt(S)*strain_comp;
pst->M=sqrt(M)*strain_comp;

pst->ul=sqrt(upper_limit95(max_dx)*S)*strain_comp*upper_limit_comp;
pst->ll=sqrt(lower_limit95(max_dx)*S)*strain_comp;
pst->centroid=sqrt(max_dx*S)*strain_comp*upper_limit_comp;
pst->snr=max_dx;

pst->max_weight=max_weight;
pst->weight_loss_fraction=(max_weight-min_weight)/max_weight;
pst->ks_value=nstats.ks_test;
pst->ks_count=nstats.ks_count;
}

void point_power_sum_stats_linear(PARTIAL_POWER_SUM_F *pps, ALIGNMENT_COEFFS *ag, POINT_STATS *pst)
{
int i, count;
float M, S, a, b, inv_S, inv_weight, inv_count, normalizer, inv_normalizer;
float *tmp=NULL;
float max_dx, min_val;
int max_dx_bin;
float weight, min_weight, max_weight;
float sum, sum_sq, sum1, sum3, sum4;
int half_window=args_info.half_window_arg;

/* allocate on stack, for speed */
tmp=aligned_alloca(useful_bins*sizeof(*tmp));

if(pps->power_im_pc==NULL) {
	fprintf(stderr, "*** INTERNAL ERROR: %s requires pps->power_im_pc!=NULL\n", __FUNCTION__);
	exit(-1);
	}

if(pps->weight_arrays_non_zero) {
	max_weight=0;
	min_weight=1e50;

	if(!pps->collapsed_weight_arrays) {
		for(i=0;i<useful_bins;i++) {
			pps->weight_pppp[i]+=pps->c_weight_pppp;
			pps->weight_pppc[i]+=pps->c_weight_pppc;
			pps->weight_ppcc[i]+=pps->c_weight_ppcc;
			pps->weight_pccc[i]+=pps->c_weight_pccc;
			pps->weight_cccc[i]+=pps->c_weight_cccc;
			}
		pps->c_weight_pppp=0;
		pps->c_weight_pppc=0;
		pps->c_weight_ppcc=0;
		pps->c_weight_pccc=0;
		pps->c_weight_cccc=0;
		pps->collapsed_weight_arrays=1;
		}

	for(i=0;i<useful_bins;i++) {
		weight=(pps->weight_pppp[i]*ag->pppp+
			pps->weight_pppc[i]*ag->pppc+
			pps->weight_ppcc[i]*ag->ppcc+
			pps->weight_pccc[i]*ag->pccc+
			pps->weight_cccc[i]*ag->cccc);
	
		if(weight>max_weight)max_weight=weight;
		if(weight<min_weight)min_weight=weight;

		tmp[i]=(pps->power_pp[i]*ag->pp+pps->power_pc[i]*ag->pc+pps->power_cc[i]*ag->cc+pps->power_im_pc[i]*ag->im_pc)/weight;
			
		}
	} else {
	weight=(pps->c_weight_pppp*ag->pppp+
		pps->c_weight_pppc*ag->pppc+
		pps->c_weight_ppcc*ag->ppcc+
		pps->c_weight_pccc*ag->pccc+
		pps->c_weight_cccc*ag->cccc+
		pps->c_weight_im_ppcc*ag->im_ppcc);
	max_weight=weight;
	min_weight=weight;

	inv_weight=1.0/weight;

	for(i=0;i<useful_bins;i++) {
		tmp[i]=((float)pps->power_pp[i]*ag->pp+(float)pps->power_pc[i]*ag->pc+(float)pps->power_cc[i]*ag->cc+(float)pps->power_im_pc[i]*ag->im_pc)*inv_weight;
		}
	}

/* 0 weight can happen due to extreme line veto at low frequencies and small spindowns */
if(min_weight<= args_info.small_weight_ratio_arg*max_weight) {
	set_missing_point_stats(pst);
	return;
	}
	
/* find highest bin */
compute_range_F(tmp, useful_bins, &max_dx, &min_val, &max_dx_bin);

/* doing everything in one pass and then subtracting does not work due to precision errors if we chance upon a very high max_dx and because float does not keep many digits */

/* there is also a possible issue with normalization, fortunately we have a ready constant now to normalize with: max_dx */


count=0;
sum=0.0;

for(i=0;i<max_dx_bin-half_window;i++) {
	a=tmp[i];
	sum+=a;
	count++;
	}

for(i=max_dx_bin+half_window+1;i<useful_bins;i++) {
	a=tmp[i];
	sum+=a;
	count++;
	}

inv_count=1.0/count;
M=sum*inv_count;

/* better normalization now that we have min_val */
inv_normalizer=max_dx-min_val;
if(inv_normalizer<=0.0)inv_normalizer=1.0; /* in case we have all 0 for missing data */
normalizer=1.0/inv_normalizer;

sum_sq=0.0;
sum1=0.0;
sum3=0.0;
sum4=0.0;
for(i=0;i<max_dx_bin-half_window;i++) {
	a=(tmp[i]-M)*normalizer;
	b=a*a;
	/* collect negative first and second moment statistics - these would describe background behaviour */
	if(a<0) {
		sum1-=a;
		sum3-=a*b;
		}
	sum_sq+=b;
	sum4+=b*b;
	}

for(i=max_dx_bin+half_window+1;i<useful_bins;i++) {
	a=(tmp[i]-M)*normalizer;
	b=a*a;
	/* collect negative first and second moment statistics - these would describe background behaviour */
	if(a<0) {
		sum1-=a;
		sum3-=a*b;
		}
	sum_sq+=b;
	sum4+=b*b;
	}

S=sqrt(sum_sq/(count-1))*inv_normalizer;
inv_S=1.0/S;

/* convert to SNR from the highest power */
max_dx=(max_dx-M)*inv_S;

if(max_dx<=0 || !isfinite(max_dx)) {
	/* In theory we could have max_dx=0 because the distribution is flat, but we really should not have this */
	fprintf(stderr, "***ERROR - irregular max_dx  max_dx=%g max_dx_bin=%d M=%g S=%g inv_S=%g tmp=%p tmp={%g %g ... %g %g ...}\n",
			max_dx,
			max_dx_bin,
			M,
			S,
			inv_S,
			tmp, 
			tmp[0], tmp[1], tmp[250], tmp[251]);
	/* this is not fatal - complain, but continue */
	}

pst->bin=max_dx_bin;
pst->iota=ag->iota;
pst->psi=ag->psi;

/* convert to upper limit units */
pst->S=sqrt(S)*strain_comp;
pst->M=sqrt(M)*strain_comp;

pst->ul=sqrt(upper_limit95(max_dx)*S)*strain_comp*upper_limit_comp;
pst->ll=sqrt(lower_limit95(max_dx)*S)*strain_comp;
pst->centroid=sqrt(max_dx*S)*strain_comp*upper_limit_comp;
pst->snr=max_dx;

pst->max_weight=max_weight;
pst->weight_loss_fraction=(max_weight-min_weight)/max_weight;

/* Apply normalization */
S*=normalizer;
//pst->ks_value=(sum4*inv_count-4*sum3*inv_count*M+6*sum_sq*inv_count*M*M-3*M*M*M*M)/(S*S*S*S);
pst->m1_neg=(sum1*inv_count)/S;
pst->m3_neg=(sum3*inv_count)/(S*S*S);
pst->m4=(sum4*inv_count)/(S*S*S*S);
//fprintf(stderr, "%g = %g %g %g %g (%d %g %g)\n", pst->ks_value, M, sum_sq*inv_count, sum3*inv_count, sum4*inv_count, count, inv_count, S);
pst->ks_value=0;
pst->ks_count=0;
}

void cblas_point_power_sum_stats_linear(PARTIAL_POWER_SUM_F *pps, ALIGNMENT_COEFFS *ag, POINT_STATS *pst)
{
int i, count;
float M, S, a, b, inv_S, inv_weight, inv_count, normalizer, inv_normalizer;
float *tmp=NULL, *weight_tmp=NULL;
NORMAL_STATS nstats;
float max_dx, min_val;
int max_dx_bin;
float weight, min_weight, max_weight;
float sum, sum_sq, sum1, sum3, sum4;
int half_window=args_info.half_window_arg;

/* allocate on stack, for speed */
tmp=aligned_alloca(useful_bins*sizeof(*tmp));

memset(&nstats, 0, sizeof(nstats));

/* sort to compute robust estimates */
nstats.flag= STAT_FLAG_ESTIMATE_MEAN
	| STAT_FLAG_ESTIMATE_SIGMA;

if(args_info.ks_test_arg){
	nstats.flag|=STAT_FLAG_ESTIMATE_KS_LEVEL
		| STAT_FLAG_COMPUTE_KS_TEST;
	}


if(pps->weight_arrays_non_zero) {
	max_weight=0;
	min_weight=1e50;


	if(!pps->collapsed_weight_arrays) {
		for(i=0;i<useful_bins;i++) {
			pps->weight_pppp[i]+=pps->c_weight_pppp;
			pps->weight_pppc[i]+=pps->c_weight_pppc;
			pps->weight_ppcc[i]+=pps->c_weight_ppcc;
			pps->weight_pccc[i]+=pps->c_weight_pccc;
			pps->weight_cccc[i]+=pps->c_weight_cccc;
			}
		pps->c_weight_pppp=0;
		pps->c_weight_pppc=0;
		pps->c_weight_ppcc=0;
		pps->c_weight_pccc=0;
		pps->c_weight_cccc=0;
		pps->collapsed_weight_arrays=1;
		}

	/* allocate on stack, for speed */
	weight_tmp=aligned_alloca(useful_bins*sizeof(*weight_tmp));

	memcpy(weight_tmp, pps->weight_pppp, useful_bins*sizeof(*weight_tmp));

	cblas_sscal(useful_bins, ag->pppp, weight_tmp, 1);
	cblas_saxpy(useful_bins, ag->pppc, pps->weight_pppc, 1, weight_tmp, 1);
	cblas_saxpy(useful_bins, ag->ppcc, pps->weight_ppcc, 1, weight_tmp, 1);
	cblas_saxpy(useful_bins, ag->pccc, pps->weight_pccc, 1, weight_tmp, 1);
	cblas_saxpy(useful_bins, ag->cccc, pps->weight_cccc, 1, weight_tmp, 1);

	memcpy(tmp,  pps->power_pp, useful_bins*sizeof(*tmp));

	cblas_sscal(useful_bins, ag->pp, tmp, 1);
	cblas_saxpy(useful_bins, ag->pc, pps->power_pc, 1, tmp, 1);
	cblas_saxpy(useful_bins, ag->cc, pps->power_cc, 1, tmp, 1);
	if(pps->power_im_pc!=NULL)cblas_saxpy(useful_bins, ag->im_pc, pps->power_im_pc, 1, tmp, 1);

	for(i=0;i<useful_bins;i++) {

/*		weight=(pps->weight_pppp[i]*ag->pppp+
			pps->weight_pppc[i]*ag->pppc+
			pps->weight_ppcc[i]*ag->ppcc+
			pps->weight_pccc[i]*ag->pccc+
			pps->weight_cccc[i]*ag->cccc);*/
	
		weight=weight_tmp[i];	

		if(weight>max_weight)max_weight=weight;
		if(weight<min_weight)min_weight=weight;

		tmp[i]/=weight;
		}

	} else {
	weight=(pps->c_weight_pppp*ag->pppp+
		pps->c_weight_pppc*ag->pppc+
		pps->c_weight_ppcc*ag->ppcc+
		pps->c_weight_pccc*ag->pccc+
		pps->c_weight_cccc*ag->cccc+
		pps->c_weight_im_ppcc*ag->im_ppcc);
	max_weight=weight;
	min_weight=weight;

	inv_weight=1.0/weight;

	memcpy(tmp,  pps->power_pp, useful_bins*sizeof(*tmp));

	cblas_sscal(useful_bins, ag->pp*inv_weight, tmp, 1);
	cblas_saxpy(useful_bins, ag->pc*inv_weight, pps->power_pc, 1, tmp, 1);
	cblas_saxpy(useful_bins, ag->cc*inv_weight, pps->power_cc, 1, tmp, 1);
	if(pps->power_im_pc!=NULL)cblas_saxpy(useful_bins, ag->im_pc*inv_weight, pps->power_im_pc, 1, tmp, 1);
	}

/* 0 weight can happen due to extreme line veto at low frequencies and small spindowns */
if(min_weight<= args_info.small_weight_ratio_arg*max_weight) {
	set_missing_point_stats(pst);
	return;
	}
	
/* find highest bin */
compute_range_F(tmp, useful_bins, &max_dx, &min_val, &max_dx_bin);

/* doing everything in one pass and then subtracting does not work due to precision errors if we chance upon a very high max_dx and because float does not keep many digits */

/* there is also a possible issue with normalization, fortunately we have a ready constant now to normalize with: max_dx */


count=0;
sum=0.0;

for(i=0;i<max_dx_bin-half_window;i++) {
	a=tmp[i];
	sum+=a;
	count++;
	}

for(i=max_dx_bin+half_window+1;i<useful_bins;i++) {
	a=tmp[i];
	sum+=a;
	count++;
	}

inv_count=1.0/count;
M=sum*inv_count;

/* better normalization now that we have min_val */
inv_normalizer=max_dx-min_val;
if(inv_normalizer<=0.0)inv_normalizer=1.0; /* in case we have all 0 for missing data */
normalizer=1.0/inv_normalizer;

sum_sq=0.0;
sum1=0.0;
sum3=0.0;
sum4=0.0;
for(i=0;i<max_dx_bin-half_window;i++) {
	a=(tmp[i]-M)*normalizer;
	b=a*a;
	/* collect negative first and second moment statistics - these would describe background behaviour */
	if(a<0) {
		sum1-=a;
		sum3-=a*b;
		}
	sum_sq+=b;
	sum4+=b*b;
	}

for(i=max_dx_bin+half_window+1;i<useful_bins;i++) {
	a=(tmp[i]-M)*normalizer;
	b=a*a;
	/* collect negative first and second moment statistics - these would describe background behaviour */
	if(a<0) {
		sum1-=a;
		sum3-=a*b;
		}
	sum_sq+=b;
	sum4+=b*b;
	}

S=sqrt(sum_sq/(count-1))*inv_normalizer;
inv_S=1.0/S;

/* convert to SNR from the highest power */
max_dx=(max_dx-M)*inv_S;

if(max_dx<=0 || !isfinite(max_dx)) {
	/* In theory we could have max_dx=0 because the distribution is flat, but we really should not have this */
	fprintf(stderr, "***ERROR - irregular max_dx  max_dx=%g max_dx_bin=%d M=%g S=%g inv_S=%g tmp=%p tmp={%g %g ... %g %g ...}\n",
			max_dx,
			max_dx_bin,
			M,
			S,
			inv_S,
			tmp, 
			tmp[0], tmp[1], tmp[250], tmp[251]);
	/* this is not fatal - complain, but continue */
	}

pst->bin=max_dx_bin;
pst->iota=ag->iota;
pst->psi=ag->psi;

/* convert to upper limit units */
pst->S=sqrt(S)*strain_comp;
pst->M=sqrt(M)*strain_comp;

pst->ul=sqrt(upper_limit95(max_dx)*S)*strain_comp*upper_limit_comp;
pst->ll=sqrt(lower_limit95(max_dx)*S)*strain_comp;
pst->centroid=sqrt(max_dx*S)*strain_comp*upper_limit_comp;
pst->snr=max_dx;

pst->max_weight=max_weight;
pst->weight_loss_fraction=(max_weight-min_weight)/max_weight;

/* Apply normalization */
S*=normalizer;
//pst->ks_value=(sum4*inv_count-4*sum3*inv_count*M+6*sum_sq*inv_count*M*M-3*M*M*M*M)/(S*S*S*S);
pst->m1_neg=(sum1*inv_count)/S;
pst->m3_neg=(sum3*inv_count)/(S*S*S);
pst->m4=(sum4*inv_count)/(S*S*S*S);
//fprintf(stderr, "%g = %g %g %g %g (%d %g %g)\n", pst->ks_value, M, sum_sq*inv_count, sum3*inv_count, sum4*inv_count, count, inv_count, S);
pst->ks_value=0;
pst->ks_count=0;
}

void sse_point_power_sum_stats_linear(PARTIAL_POWER_SUM_F *pps, ALIGNMENT_COEFFS *ag, POINT_STATS *pst)
{
#if MANUAL_SSE
int i, count;
float M, S, a, b, inv_S, inv_weight, inv_count, normalizer, inv_normalizer;
float *tmp=NULL;
NORMAL_STATS nstats;
float max_dx, min_val;
int max_dx_bin;
float weight, min_weight, max_weight;
float sum, sum_sq, sum1, sum3, sum4;
int half_window=args_info.half_window_arg;
float *tmp2=NULL;
__m128 v4a,v4b, v4c, v4d, v4weight, v4tmp, v4sum, v4sum_sq, v4sum3, v4sum4, v4zero;

/* allocate on stack, for speed */
tmp=aligned_alloca(useful_bins*sizeof(*tmp));
tmp2=aligned_alloca(4*sizeof(tmp2));

memset(&nstats, 0, sizeof(nstats));

/* sort to compute robust estimates */
nstats.flag= STAT_FLAG_ESTIMATE_MEAN
	| STAT_FLAG_ESTIMATE_SIGMA;

if(args_info.ks_test_arg){
	nstats.flag|=STAT_FLAG_ESTIMATE_KS_LEVEL
		| STAT_FLAG_COMPUTE_KS_TEST;
	}


if(pps->power_im_pc==NULL) {
	fprintf(stderr, "*** INTERNAL ERROR: %s requires pps->power_im_pc!=NULL\n", __FUNCTION__);
	exit(-1);
	}

if(pps->weight_arrays_non_zero) {
	max_weight=0;
	min_weight=1e50;

	if(!pps->collapsed_weight_arrays) {
		for(i=0;i<useful_bins;i++) {
			pps->weight_pppp[i]+=pps->c_weight_pppp;
			pps->weight_pppc[i]+=pps->c_weight_pppc;
			pps->weight_ppcc[i]+=pps->c_weight_ppcc;
			pps->weight_pccc[i]+=pps->c_weight_pccc;
			pps->weight_cccc[i]+=pps->c_weight_cccc;
			}
		pps->c_weight_pppp=0;
		pps->c_weight_pppc=0;
		pps->c_weight_ppcc=0;
		pps->c_weight_pccc=0;
		pps->c_weight_cccc=0;
		pps->collapsed_weight_arrays=1;
		}

	for(i=0;i<(useful_bins-3);i+=4) {
		/* compute weight */
		v4a=_mm_load_ps(&(pps->weight_pppp[i]));
		v4b=_mm_load1_ps(&ag->pppp);
		v4weight=_mm_mul_ps(v4a, v4b);

		v4a=_mm_load_ps(&(pps->weight_pppc[i]));
		v4b=_mm_load1_ps(&ag->pppc);
		v4c=_mm_mul_ps(v4a, v4b);

		v4weight=_mm_add_ps(v4weight, v4c);

		v4a=_mm_load_ps(&(pps->weight_ppcc[i]));
		v4b=_mm_load1_ps(&ag->ppcc);
		v4c=_mm_mul_ps(v4a, v4b);

		v4weight=_mm_add_ps(v4weight, v4c);

		v4a=_mm_load_ps(&(pps->weight_pccc[i]));
		v4b=_mm_load1_ps(&ag->pccc);
		v4c=_mm_mul_ps(v4a, v4b);

		v4weight=_mm_add_ps(v4weight, v4c);

		v4a=_mm_load_ps(&(pps->weight_cccc[i]));
		v4b=_mm_load1_ps(&ag->cccc);
		v4c=_mm_mul_ps(v4a, v4b);

		v4weight=_mm_add_ps(v4weight, v4c);

		_mm_store_ps(tmp2, v4weight);

		/* update max and min weight variables */

		weight=tmp2[0];
		if(weight>max_weight)max_weight=weight;
		if(weight<min_weight)min_weight=weight;

		weight=tmp2[1];
		if(weight>max_weight)max_weight=weight;
		if(weight<min_weight)min_weight=weight;

		weight=tmp2[2];
		if(weight>max_weight)max_weight=weight;
		if(weight<min_weight)min_weight=weight;

		weight=tmp2[3];
		if(weight>max_weight)max_weight=weight;
		if(weight<min_weight)min_weight=weight;

		/* compute power sum */

		v4a=_mm_load_ps(&(pps->power_pp[i]));
		v4b=_mm_load1_ps(&ag->pp);
		v4tmp=_mm_mul_ps(v4a, v4b);

		v4a=_mm_load_ps(&(pps->power_pc[i]));
		v4b=_mm_load1_ps(&ag->pc);
		v4c=_mm_mul_ps(v4a, v4b);

		v4tmp=_mm_add_ps(v4tmp, v4c);

		v4a=_mm_load_ps(&(pps->power_cc[i]));
		v4b=_mm_load1_ps(&ag->cc);
		v4c=_mm_mul_ps(v4a, v4b);

		v4tmp=_mm_add_ps(v4tmp, v4c);

		v4a=_mm_load_ps(&(pps->power_im_pc[i]));
		v4b=_mm_load1_ps(&ag->im_pc);
		v4c=_mm_mul_ps(v4a, v4b);

		v4tmp=_mm_add_ps(v4tmp, v4c);

		v4tmp=_mm_div_ps(v4tmp, v4weight);

		_mm_store_ps(&(tmp[i]), v4tmp);

		}

	for(;i<useful_bins;i++) {
		weight=(pps->weight_pppp[i]*ag->pppp+
			pps->weight_pppc[i]*ag->pppc+
			pps->weight_ppcc[i]*ag->ppcc+
			pps->weight_pccc[i]*ag->pccc+
			pps->weight_cccc[i]*ag->cccc);
	
		if(weight>max_weight)max_weight=weight;
		if(weight<min_weight)min_weight=weight;

		tmp[i]=(pps->power_pp[i]*ag->pp+pps->power_pc[i]*ag->pc+pps->power_cc[i]*ag->cc+pps->power_im_pc[i]*ag->im_pc)/weight;
			
		}

	/* verify */
	#if 0
	if(0){
		float a1, a2, m1, m2;
		int *b1=&a1, *b2=&a2;
		m1=0;
		m2=1e50;
		for(i=0;i<useful_bins;i++) {
			weight=(pps->weight_pppp[i]*ag->pppp+
				pps->weight_pppc[i]*ag->pppc+
				pps->weight_ppcc[i]*ag->ppcc+
				pps->weight_pccc[i]*ag->pccc+
				pps->weight_cccc[i]*ag->cccc);
		
			if(weight>max_weight) fprintf(stderr, "*1*  %d %g %g %g\n", i, weight, max_weight, tmp[i]);
			if(weight<min_weight) fprintf(stderr, "*2*  %d %g %g %g\n", i, weight, min_weight, tmp[i]);
	
			if(weight>m1) m1=weight;
			if(weight<m2) m2=weight;
	
			a1=tmp[i];
			a2=(pps->power_pp[i]*ag->pp+pps->power_pc[i]*ag->pc+pps->power_cc[i]*ag->cc)/weight;
	
			if(*b1!=*b2){
				fprintf(stderr, " *3* %d %g %g %g %g %g\n", i, (pps->power_pp[i]*ag->pp+pps->power_pc[i]*ag->pc+pps->power_cc[i]*ag->cc)/weight, tmp[i], weight, min_weight, max_weight);
				}
				
			}
		a1=m1;
		a2=max_weight;
	
		if(*b1!=*b2){
			fprintf(stderr, " *4* %g %g %g %g\n", m2, m1, min_weight, max_weight);
			}
	
		a1=m2;
		a2=min_weight;
	
		if(*b1!=*b2) {
			fprintf(stderr, " *5* %g %g %g %g\n", m2, m1, min_weight, max_weight);
			}
		}
	#endif
	} else {
	weight=(pps->c_weight_pppp*ag->pppp+
		pps->c_weight_pppc*ag->pppc+
		pps->c_weight_ppcc*ag->ppcc+
		pps->c_weight_pccc*ag->pccc+
		pps->c_weight_cccc*ag->cccc+
		pps->c_weight_im_ppcc*ag->im_ppcc);
	max_weight=weight;
	min_weight=weight;

	inv_weight=1.0/weight;

	for(i=0;i<(useful_bins-3);i+=4) {
		/* compute power sum */

		v4a=_mm_load_ps(&(pps->power_pp[i]));
		v4b=_mm_load1_ps(&ag->pp);
		v4tmp=_mm_mul_ps(v4a, v4b);

		v4a=_mm_load_ps(&(pps->power_pc[i]));
		v4b=_mm_load1_ps(&ag->pc);
		v4c=_mm_mul_ps(v4a, v4b);

		v4tmp=_mm_add_ps(v4tmp, v4c);

		v4a=_mm_load_ps(&(pps->power_cc[i]));
		v4b=_mm_load1_ps(&ag->cc);
		v4c=_mm_mul_ps(v4a, v4b);

		v4tmp=_mm_add_ps(v4tmp, v4c);

		v4a=_mm_load_ps(&(pps->power_im_pc[i]));
		v4b=_mm_load1_ps(&ag->im_pc);
		v4c=_mm_mul_ps(v4a, v4b);

		v4tmp=_mm_add_ps(v4tmp, v4c);

		v4a=_mm_load1_ps(&inv_weight);

		v4tmp=_mm_mul_ps(v4tmp, v4a);

		_mm_store_ps(&(tmp[i]), v4tmp);
		}

	for(;i<useful_bins;i++) {
		tmp[i]=((float)pps->power_pp[i]*ag->pp+(float)pps->power_pc[i]*ag->pc+(float)pps->power_cc[i]*ag->cc+pps->power_im_pc[i]*ag->im_pc)*inv_weight;
		}
	}

/* 0 weight can happen due to extreme line veto at low frequencies and small spindowns */
if(min_weight<= args_info.small_weight_ratio_arg*max_weight) {
	set_missing_point_stats(pst);
	return;
	}
	
/* find highest bin */
compute_range_F(tmp, useful_bins, &max_dx, &min_val, &max_dx_bin);


/* doing everything in one pass and then subtracting does not work due to precision errors if we chance upon a very high max_dx and because float does not keep many digits */

/* there is also a possible issue with normalization, fortunately we have a ready constant now to normalize with: max_dx */


sum=0.0;

v4sum=_mm_setzero_ps();
for(i=0;i<max_dx_bin-half_window-3;i+=4) {
	v4sum=_mm_add_ps(v4sum, _mm_load_ps(&(tmp[i])));
	}

for(;i<max_dx_bin-half_window;i++) {
	sum+=tmp[i];
	}

count=i;
for(i=max_dx_bin+half_window+1;(i& 3) && (i<useful_bins);i++) {
	sum+=tmp[i];
	}

for(;i<useful_bins-3;i+=4) {
	v4sum=_mm_add_ps(v4sum, _mm_load_ps(&(tmp[i])));
	}

for(;i<useful_bins;i++) {
	sum+=tmp[i];
	}
_mm_store_ps(tmp2, v4sum);
sum+=tmp2[0]+tmp2[1]+tmp2[2]+tmp2[3];

count+=i-max_dx_bin-half_window-1;

inv_count=1.0/count;
M=sum*inv_count;

/* better normalization now that we have min_val */
inv_normalizer=max_dx-min_val;
if(inv_normalizer<=0.0)inv_normalizer=1.0; /* in case we have all 0 for missing data */
normalizer=1.0/inv_normalizer;

sum_sq=0.0;
sum1=0.0;
sum3=0.0;
sum4=0.0;
v4sum=_mm_setzero_ps();
v4sum_sq=_mm_setzero_ps();
v4sum3=_mm_setzero_ps();
v4sum4=_mm_setzero_ps();
v4zero=_mm_setzero_ps();

v4a=_mm_load1_ps(&M);
v4b=_mm_load1_ps(&normalizer);
for(i=0;i<max_dx_bin-half_window-3;i+=4) {
	v4tmp=_mm_mul_ps(_mm_sub_ps(_mm_load_ps(&(tmp[i])), v4a), v4b);
	v4c=_mm_mul_ps(v4tmp, v4tmp);
	/* collect negative first and second moment statistics - these would describe background behaviour */
	v4d=_mm_min_ps(v4tmp, v4zero);
	v4sum=_mm_sub_ps(v4sum, v4d);
	v4sum3=_mm_sub_ps(v4sum3, _mm_mul_ps(v4d, v4c));

	v4sum_sq=_mm_add_ps(v4sum_sq, v4c);
	v4sum4=_mm_add_ps(v4sum4, _mm_mul_ps(v4c, v4c));
	}

for(;i<max_dx_bin-half_window;i++) {
	a=(tmp[i]-M)*normalizer;
	b=a*a;
	/* collect negative first and second moment statistics - these would describe background behaviour */
	if(a<0) {
		sum1-=a;
		sum3-=a*b;
		}
	sum_sq+=b;
	sum4+=b*b;
	}

for(i=max_dx_bin+half_window+1;(i & 3) && (i<useful_bins);i++) {
	a=(tmp[i]-M)*normalizer;
	b=a*a;
	/* collect negative first and second moment statistics - these would describe background behaviour */
	if(a<0) {
		sum1-=a;
		sum3-=a*b;
		}
	sum_sq+=b;
	sum4+=b*b;
	}

for(;i<useful_bins-3;i+=4) {
	v4tmp=_mm_mul_ps(_mm_sub_ps(_mm_load_ps(&(tmp[i])), v4a), v4b);
	v4c=_mm_mul_ps(v4tmp, v4tmp);
	/* collect negative first and second moment statistics - these would describe background behaviour */
	v4d=_mm_min_ps(v4tmp, v4zero);
	v4sum=_mm_sub_ps(v4sum, v4d);
	v4sum3=_mm_sub_ps(v4sum3, _mm_mul_ps(v4d, v4c));

	v4sum_sq=_mm_add_ps(v4sum_sq, v4c);
	v4sum4=_mm_add_ps(v4sum4, _mm_mul_ps(v4c, v4c));
	}

for(;i<useful_bins;i++) {
	a=(tmp[i]-M)*normalizer;
	b=a*a;
	/* collect negative first and second moment statistics - these would describe background behaviour */
	if(a<0) {
		sum1-=a;
		sum3-=a*b;
		}
	sum_sq+=b;
	sum4+=b*b;
	}

_mm_store_ps(tmp2, v4sum);
sum1+=tmp2[0]+tmp2[1]+tmp2[2]+tmp2[3];

_mm_store_ps(tmp2, v4sum_sq);
sum_sq+=tmp2[0]+tmp2[1]+tmp2[2]+tmp2[3];

_mm_store_ps(tmp2, v4sum3);
sum3+=tmp2[0]+tmp2[1]+tmp2[2]+tmp2[3];

_mm_store_ps(tmp2, v4sum4);
sum4+=tmp2[0]+tmp2[1]+tmp2[2]+tmp2[3];

S=sqrt(sum_sq/(count-1))*inv_normalizer;
inv_S=1.0/S;

/* convert to SNR from the highest power */
max_dx=(max_dx-M)*inv_S;

if(max_dx<=0 || !isfinite(max_dx)) {
	/* In theory we could have max_dx=0 because the distribution is flat, but we really should not have this */
	fprintf(stderr, "***ERROR - irregular max_dx  max_dx=%g max_dx_bin=%d M=%g S=%g inv_S=%g tmp=%p tmp={%g %g ... %g %g ...}\n",
			max_dx,
			max_dx_bin,
			M,
			S,
			inv_S,
			tmp, 
			tmp[0], tmp[1], tmp[250], tmp[251]);
	/* this is not fatal - complain, but continue */
	}

pst->bin=max_dx_bin;
pst->iota=ag->iota;
pst->psi=ag->psi;

/* convert to upper limit units */
pst->S=sqrt(S)*strain_comp;
pst->M=sqrt(M)*strain_comp;

pst->ul=sqrt(upper_limit95(max_dx)*S)*strain_comp*upper_limit_comp;
pst->ll=sqrt(lower_limit95(max_dx)*S)*strain_comp;
pst->centroid=sqrt(max_dx*S)*strain_comp*upper_limit_comp;
pst->snr=max_dx;

pst->max_weight=max_weight;
pst->weight_loss_fraction=(max_weight-min_weight)/max_weight;

/* Apply normalization */
S*=normalizer;
//pst->ks_value=(sum4*inv_count-4*sum3*inv_count*M+6*sum_sq*inv_count*M*M-3*M*M*M*M)/(S*S*S*S);
pst->m1_neg=(sum1*inv_count)/S;
pst->m3_neg=(sum3*inv_count)/(S*S*S);
pst->m4=(sum4*inv_count)/(S*S*S*S);
//fprintf(stderr, "%g = %g %g %g %g (%d %g %g)\n", pst->ks_value, M, sum_sq*inv_count, sum3*inv_count, sum4*inv_count, count, inv_count, S);
pst->ks_value=0;
pst->ks_count=0;
#else
fprintf(stderr, "**** MANUAL_SSE disabled in %s\n", __FUNCTION__);
exit(-2);
#endif
}

void point_power_sum_stats_universal(PARTIAL_POWER_SUM_F *pps, ALIGNMENT_COEFFS *ag, POINT_STATS *pst)
{
int i, count;
float M, S, a, b, s3, inv_S, inv_weight, inv_count, normalizer, inv_normalizer;
float *tmp=NULL;
float max_dx, min_val;
int max_dx_bin;
float weight, min_weight, max_weight;
float sum, sum_sq, sum1, sum3, sum4, sum_abs, sum_c;
int half_window=args_info.half_window_arg;

/* allocate on stack, for speed */
tmp=aligned_alloca(useful_bins*sizeof(*tmp));

if(pps->power_im_pc==NULL) {
	fprintf(stderr, "*** INTERNAL ERROR: %s requires pps->power_im_pc!=NULL\n", __FUNCTION__);
	exit(-1);
	}

if(pps->weight_arrays_non_zero) {
	max_weight=0;
	min_weight=1e50;

	if(!pps->collapsed_weight_arrays) {
		for(i=0;i<useful_bins;i++) {
			pps->weight_pppp[i]+=pps->c_weight_pppp;
			pps->weight_pppc[i]+=pps->c_weight_pppc;
			pps->weight_ppcc[i]+=pps->c_weight_ppcc;
			pps->weight_pccc[i]+=pps->c_weight_pccc;
			pps->weight_cccc[i]+=pps->c_weight_cccc;
			}
		pps->c_weight_pppp=0;
		pps->c_weight_pppc=0;
		pps->c_weight_ppcc=0;
		pps->c_weight_pccc=0;
		pps->c_weight_cccc=0;
		pps->collapsed_weight_arrays=1;
		}

	for(i=0;i<useful_bins;i++) {
		weight=(pps->weight_pppp[i]*ag->pppp+
			pps->weight_pppc[i]*ag->pppc+
			pps->weight_ppcc[i]*ag->ppcc+
			pps->weight_pccc[i]*ag->pccc+
			pps->weight_cccc[i]*ag->cccc);
	
		if(weight>max_weight)max_weight=weight;
		if(weight<min_weight)min_weight=weight;

		tmp[i]=(pps->power_pp[i]*ag->pp+pps->power_pc[i]*ag->pc+pps->power_cc[i]*ag->cc+pps->power_im_pc[i]*ag->im_pc)/weight;
			
		}
	} else {
	weight=(pps->c_weight_pppp*ag->pppp+
		pps->c_weight_pppc*ag->pppc+
		pps->c_weight_ppcc*ag->ppcc+
		pps->c_weight_pccc*ag->pccc+
		pps->c_weight_cccc*ag->cccc+
		pps->c_weight_im_ppcc*ag->im_ppcc);
	max_weight=weight;
	min_weight=weight;

	inv_weight=1.0/weight;

	for(i=0;i<useful_bins;i++) {
		tmp[i]=((float)pps->power_pp[i]*ag->pp+(float)pps->power_pc[i]*ag->pc+(float)pps->power_cc[i]*ag->cc+(float)pps->power_im_pc[i]*ag->im_pc)*inv_weight;
		}
	}

/* 0 weight can happen due to extreme line veto at low frequencies and small spindowns */
if(min_weight<= args_info.small_weight_ratio_arg*max_weight) {
	set_missing_point_stats(pst);
	return;
	}
	
/* find highest bin */

compute_range_F(tmp, useful_bins, &max_dx, &min_val, &max_dx_bin);

/*
max_dx=tmp[0];
max_dx_bin=0;

for(i=1;i<useful_bins;i++) {
	a=tmp[i];
	if(a>max_dx) {
		max_dx=a;
		max_dx_bin=i;
		}
	}*/

/* Constants for 0.95 confidence level */
#define LEVEL 0.95
#define SIGMA 1.644854

/* Constant for mean(|x|) */
#define SQRT_PI_2   1.253314

/* doing everything in one pass and then subtracting does not work due to precision errors if we chance upon a very high max_dx and because float does not keep many digits */

/* there is also a possible issue with normalization, fortunately we have a ready constant now to normalize with: max_dx */


count=0;
sum=0.0;

for(i=0;i<max_dx_bin-half_window;i++) {
	a=tmp[i];
	sum+=a;
	count++;
	}

for(i=max_dx_bin+half_window+1;i<useful_bins;i++) {
	a=tmp[i];
	sum+=a;
	count++;
	}

inv_count=1.0/count;
M=sum*inv_count;

/* better normalization now that we have min_val */
inv_normalizer=max_dx-min_val;
if(inv_normalizer<=0.0)inv_normalizer=1.0; /* in case we have all 0 for missing data */
normalizer=1.0/inv_normalizer;

sum_sq=0.0;
sum1=0.0;
sum3=0.0;
sum4=0.0;
sum_abs=0.0;
for(i=0;i<max_dx_bin-half_window;i++) {
	a=(tmp[i]-M)*normalizer;
	b=a*a;
	/* collect negative first and second moment statistics - these would describe background behaviour */
	if(a<0) {
		sum1-=a;
		sum3-=a*b;
		}
	sum_abs+=fabs(a);
	sum_sq+=b;
	sum4+=b*b;
	}

for(i=max_dx_bin+half_window+1;i<useful_bins;i++) {
	a=(tmp[i]-M)*normalizer;
	b=a*a;
	/* collect negative first and second moment statistics - these would describe background behaviour */
	if(a<0) {
		sum1-=a;
		sum3-=a*b;
		}
	sum_abs+=fabs(a);
	sum_sq+=b;
	sum4+=b*b;
	}

// sqrt-based
//S=sqrt(sum_sq/(count-1))*max_dx;
//inv_S=1.0/S;

// abs-based
S=sum_abs*inv_normalizer*SQRT_PI_2/count;
inv_S=1.0/S;

/* convert to SNR from the highest power */
max_dx=(max_dx-M)*inv_S;

if(max_dx<=0 || !isfinite(max_dx)) {
	/* In theory we could have max_dx=0 because the distribution is flat, but we really should not have this */
	fprintf(stderr, "***ERROR - irregular max_dx  max_dx=%g max_dx_bin=%d M=%g S=%g inv_S=%g tmp=%p tmp={%g %g ... %g %g ...}\n",
			max_dx,
			max_dx_bin,
			M,
			S,
			inv_S,
			tmp, 
			tmp[0], tmp[1], tmp[250], tmp[251]);
	/* this is not fatal - complain, but continue */
	}
	
/* Third pass - compute c */
	
sum_c=0.0;
for(i=0;i<max_dx_bin-half_window;i++) {
	a=(M-tmp[i]);
	/* collect negative threshold statistics */
	if(a>S*SIGMA) {
		b=a/(S*SIGMA);
		sum_c+=(b-1.0)/(b+3.0);
		}
	}

for(i=max_dx_bin+half_window+1;i<useful_bins;i++) {
	a=(M-tmp[i]);
	/* collect negative threshold statistics */
	if(a>S*SIGMA) {
		b=a/(S*SIGMA);
		sum_c+=(b-1.0)/(b+3.0);
		}
	}
sum_c=sum_c/count;
	
s3=SIGMA/(1.0-sum_c/(1.0-LEVEL));

pst->bin=max_dx_bin;
pst->iota=ag->iota;
pst->psi=ag->psi;

/* convert to upper limit units */
pst->S=sqrt(S)*strain_comp;
pst->M=sqrt(M)*strain_comp;

pst->ul=sqrt(S*(max_dx+s3))*strain_comp*upper_limit_comp;
/* for debugging store s3 in lower limit variable */
pst->ll=s3;
pst->centroid=sqrt(max_dx*S)*strain_comp*upper_limit_comp;
pst->snr=max_dx;

pst->max_weight=max_weight;
pst->weight_loss_fraction=(max_weight-min_weight)/max_weight;

/* Apply normalization */
S*=normalizer;
//pst->ks_value=(sum4*inv_count-4*sum3*inv_count*M+6*sum_sq*inv_count*M*M-3*M*M*M*M)/(S*S*S*S);
pst->m1_neg=(sum1*inv_count)/S;
pst->m3_neg=(sum3*inv_count)/(S*S*S);
pst->m4=(sum4*inv_count)/(S*S*S*S);
//fprintf(stderr, "%g = %g %g %g %g (%d %g %g)\n", pst->ks_value, M, sum_sq*inv_count, sum3*inv_count, sum4*inv_count, count, inv_count, S);
pst->ks_value=0;
pst->ks_count=0;
}

void compute_power(PARTIAL_POWER_SUM_F *pps, ALIGNMENT_COEFFS *ag, float *tmp, float *min_weight, float *max_weight)
{
int i;
float weight, inv_weight;

if(pps->weight_arrays_non_zero) {
	*max_weight=0;
	*min_weight=1e50;

	if(!pps->collapsed_weight_arrays) {
		PRAGMA_IVDEP
		for(i=0;i<useful_bins;i++) {
			pps->weight_pppp[i]+=pps->c_weight_pppp;
			pps->weight_pppc[i]+=pps->c_weight_pppc;
			pps->weight_ppcc[i]+=pps->c_weight_ppcc;
			pps->weight_pccc[i]+=pps->c_weight_pccc;
			pps->weight_cccc[i]+=pps->c_weight_cccc;
			}
		pps->c_weight_pppp=0;
		pps->c_weight_pppc=0;
		pps->c_weight_ppcc=0;
		pps->c_weight_pccc=0;
		pps->c_weight_cccc=0;
		pps->collapsed_weight_arrays=1;
		}


	if(pps->power_im_pc!=NULL) {

		PRAGMA_IVDEP
		for(i=0;i<useful_bins;i++) {
			weight=(pps->weight_pppp[i]*ag->pppp+
				pps->weight_pppc[i]*ag->pppc+
				pps->weight_ppcc[i]*ag->ppcc+
				pps->weight_pccc[i]*ag->pccc+
				pps->weight_cccc[i]*ag->cccc);
		
			if(weight>*max_weight)*max_weight=weight;
			if(weight<*min_weight)*min_weight=weight;

			tmp[i]=(pps->power_pp[i]*ag->pp+pps->power_pc[i]*ag->pc+pps->power_cc[i]*ag->cc+pps->power_im_pc[i]*ag->im_pc)/weight;
				
			}
		} else {
		PRAGMA_IVDEP
		for(i=0;i<useful_bins;i++) {
			weight=(pps->weight_pppp[i]*ag->pppp+
				pps->weight_pppc[i]*ag->pppc+
				pps->weight_ppcc[i]*ag->ppcc+
				pps->weight_pccc[i]*ag->pccc+
				pps->weight_cccc[i]*ag->cccc);
		
			if(weight>*max_weight)*max_weight=weight;
			if(weight<*min_weight)*min_weight=weight;

			tmp[i]=(pps->power_pp[i]*ag->pp+pps->power_pc[i]*ag->pc+pps->power_cc[i]*ag->cc)/weight;
				
			}
		}
	} else {
	weight=(pps->c_weight_pppp*ag->pppp+
		pps->c_weight_pppc*ag->pppc+
		pps->c_weight_ppcc*ag->ppcc+
		pps->c_weight_pccc*ag->pccc+
		pps->c_weight_cccc*ag->cccc+
		pps->c_weight_im_ppcc*ag->im_ppcc);
	*max_weight=weight;
	*min_weight=weight;

	inv_weight=1.0/weight;

	if(pps->power_im_pc!=NULL) {
		PRAGMA_IVDEP
		for(i=0;i<useful_bins;i++) {
			tmp[i]=((float)pps->power_pp[i]*ag->pp+(float)pps->power_pc[i]*ag->pc+(float)pps->power_cc[i]*ag->cc+(float)pps->power_im_pc[i]*ag->im_pc)*inv_weight;
			}
		} else {
		PRAGMA_IVDEP
		for(i=0;i<useful_bins;i++) {
			tmp[i]=((float)pps->power_pp[i]*ag->pp+(float)pps->power_pc[i]*ag->pc+(float)pps->power_cc[i]*ag->cc)*inv_weight;
			}
		}
	}
}


double x_epsilon=0.0;

void compute_universal_statistics(float *tmp, float min_weight, float max_weight, ALIGNMENT_COEFFS *ag, POINT_STATS *pst)
{
int half_window=args_info.half_window_arg;
float sum, sum_sq, sum1, sum3, sum4, sum_abs, sum_c;
float max_dx, min_val;
int max_dx_bin, i, count;
float M, S, a, b, s3, inv_S, inv_count, normalizer, inv_normalizer;

/* 0 weight can happen due to extreme line veto at low frequencies and small spindowns */
if(min_weight<= args_info.small_weight_ratio_arg*max_weight) {
	set_missing_point_stats(pst);
	return;
	}
	
/* find highest bin */
compute_range_F(tmp, useful_bins, &max_dx, &min_val, &max_dx_bin);

/* Constant for mean(|x|) */
#define SQRT_PI_2   1.253314

#define SQRT_2PI    2.506628

#define UNIV_INV_B  2.0

/* doing everything in one pass and then subtracting does not work due to precision errors if we chance upon a very high max_dx and because float does not keep many digits */

/* there is also a possible issue with normalization, fortunately we have a ready constant now to normalize with: max_dx */


count=0;
sum=0.0;

PRAGMA_IVDEP
for(i=0;i<max_dx_bin-half_window;i++) {
	a=tmp[i];
	sum+=a;
	count++;
	}

PRAGMA_IVDEP
for(i=max_dx_bin+half_window+1;i<useful_bins;i++) {
	a=tmp[i];
	sum+=a;
	count++;
	}

inv_count=1.0/count;
M=sum*inv_count;

/* better normalization now that we have min_val */
inv_normalizer=max_dx-min_val;
if(inv_normalizer<=0.0)inv_normalizer=1.0; /* in case we have all 0 for missing data */
normalizer=1.0/inv_normalizer;

sum_sq=0.0;
sum1=0.0;
sum3=0.0;
sum4=0.0;
sum_abs=0.0;
PRAGMA_IVDEP
for(i=0;i<max_dx_bin-half_window;i++) {
	a=(tmp[i]-M)*normalizer;
	b=a*a;
	/* collect negative first and second moment statistics - these would describe background behaviour */
	if(a<0) {
		sum1-=a;
		sum3-=a*b;
		}
	sum_abs+=fabs(a);
	sum_sq+=b;
	sum4+=b*b;
	}

PRAGMA_IVDEP
for(i=max_dx_bin+half_window+1;i<useful_bins;i++) {
	a=(tmp[i]-M)*normalizer;
	b=a*a;
	/* collect negative first and second moment statistics - these would describe background behaviour */
	if(a<0) {
		sum1-=a;
		sum3-=a*b;
		}
	sum_abs+=fabs(a);
	sum_sq+=b;
	sum4+=b*b;
	}

// sqrt-based
//S=sqrt(sum_sq/(count-1))*max_dx;
//inv_S=1.0/S;

// use lower tail only
S=sum1*inv_normalizer*SQRT_2PI/count;
inv_S=1.0/S;

/* convert to SNR from the highest power */
max_dx=(max_dx-M)*inv_S;

if(max_dx<=0 || !isfinite(max_dx)) {
	/* In theory we could have max_dx=0 because the distribution is flat, but we really should not have this */
	fprintf(stderr, "***ERROR - irregular max_dx  max_dx=%g max_dx_bin=%d M=%g S=%g inv_S=%g tmp=%p tmp={%g %g ... %g %g ...}\n",
			max_dx,
			max_dx_bin,
			M,
			S,
			inv_S,
			tmp, 
			tmp[0], tmp[1], tmp[250], tmp[251]);
	/* this is not fatal - complain, but continue */
	}
	
/* Third pass - compute delta */
	
sum_c=0.0;

PRAGMA_IVDEP
for(i=0;i<max_dx_bin-half_window;i++) {
	a=(M-x_epsilon*S)-tmp[i];
	/* collect negative threshold statistics */
	if(a>=0) {
		sum_c+=a+UNIV_INV_B*S;
		}
	}

PRAGMA_IVDEP
for(i=max_dx_bin+half_window+1;i<useful_bins;i++) {
	a=(M-x_epsilon*S)-tmp[i];
	/* collect negative threshold statistics */
	if(a>=0) {
		sum_c+=a+UNIV_INV_B*S;
		}
	}

// fprintf(stderr, "___ sum_c=%g inv_S=%g count=%d max_dx_bin=%d %f M=%g\n", sum_c*inv_S, inv_S, count, max_dx_bin, UNIV_INV_B, M);
sum_c=sum_c*inv_S/(UNIV_INV_B*count*(1.0-args_info.confidence_level_arg));

if(sum_c<=1)
	s3=x_epsilon;
	else
	s3=x_epsilon+(sum_c-1)*UNIV_INV_B;

pst->bin=max_dx_bin;
pst->iota=ag->iota;
pst->psi=ag->psi;

/* convert to upper limit units */
pst->S=sqrt(S)*strain_comp;
pst->M=sqrt(M)*strain_comp;

pst->ul=sqrt(S*(max_dx+s3))*strain_comp*upper_limit_comp;
/* for debugging store s3 in lower limit variable */
pst->ll=s3;
pst->centroid=sqrt(max_dx*S)*strain_comp*upper_limit_comp;
pst->snr=max_dx;

pst->max_weight=max_weight;
pst->weight_loss_fraction=(max_weight-min_weight)/max_weight;

/* Apply normalization */
S*=normalizer;
//pst->ks_value=(sum4*inv_count-4*sum3*inv_count*M+6*sum_sq*inv_count*M*M-3*M*M*M*M)/(S*S*S*S);
pst->m1_neg=sum_c;
pst->m3_neg=(sum3*inv_count)/(S*S*S);
pst->m4=(sum4*inv_count)/(S*S*S*S);
//fprintf(stderr, "%g = %g %g %g %g (%d %g %g)\n", pst->ks_value, M, sum_sq*inv_count, sum3*inv_count, sum4*inv_count, count, inv_count, S);
pst->ks_value=0;
pst->ks_count=0;
}

void point_power_sum_stats_universal_piecewise_linear(PARTIAL_POWER_SUM_F *pps, ALIGNMENT_COEFFS *ag, POINT_STATS *pst)
{
float *tmp=NULL;
float min_weight, max_weight;

/* allocate on stack, for speed */
tmp=aligned_alloca(useful_bins*sizeof(*tmp));

compute_power(pps, ag, tmp, &min_weight, &max_weight);
compute_universal_statistics(tmp, min_weight, max_weight, ag, pst);
}

void point_power_sum_stats_universal_piecewise_linearA(PARTIAL_POWER_SUM_F *pps, ALIGNMENT_COEFFS *ag, POINT_STATS *pst)
{
int i, count;
float M, S, a, b, s3, inv_S, inv_weight, inv_count, normalizer, inv_normalizer;
float *tmp=NULL;
float max_dx, min_val;
int max_dx_bin;
float weight, min_weight, max_weight;
float sum, sum_sq, sum1, sum3, sum4, sum_abs, sum_c;
int half_window=args_info.half_window_arg;

/* allocate on stack, for speed */
tmp=aligned_alloca(useful_bins*sizeof(*tmp));

if(pps->power_im_pc==NULL) {
	fprintf(stderr, "*** INTERNAL ERROR: %s requires pps->power_im_pc!=NULL\n", __FUNCTION__);
	exit(-1);
	}

if(pps->weight_arrays_non_zero) {
	max_weight=0;
	min_weight=1e50;

	if(!pps->collapsed_weight_arrays) {
		PRAGMA_IVDEP
		for(i=0;i<useful_bins;i++) {
			pps->weight_pppp[i]+=pps->c_weight_pppp;
			pps->weight_pppc[i]+=pps->c_weight_pppc;
			pps->weight_ppcc[i]+=pps->c_weight_ppcc;
			pps->weight_pccc[i]+=pps->c_weight_pccc;
			pps->weight_cccc[i]+=pps->c_weight_cccc;
			}
		pps->c_weight_pppp=0;
		pps->c_weight_pppc=0;
		pps->c_weight_ppcc=0;
		pps->c_weight_pccc=0;
		pps->c_weight_cccc=0;
		pps->collapsed_weight_arrays=1;
		}

		PRAGMA_IVDEP
	for(i=0;i<useful_bins;i++) {
		weight=(pps->weight_pppp[i]*ag->pppp+
			pps->weight_pppc[i]*ag->pppc+
			pps->weight_ppcc[i]*ag->ppcc+
			pps->weight_pccc[i]*ag->pccc+
			pps->weight_cccc[i]*ag->cccc);
	
		if(weight>max_weight)max_weight=weight;
		if(weight<min_weight)min_weight=weight;

		tmp[i]=(pps->power_pp[i]*ag->pp+pps->power_pc[i]*ag->pc+pps->power_cc[i]*ag->cc+pps->power_im_pc[i]*ag->im_pc)/weight;
			
		}
	} else {
	weight=(pps->c_weight_pppp*ag->pppp+
		pps->c_weight_pppc*ag->pppc+
		pps->c_weight_ppcc*ag->ppcc+
		pps->c_weight_pccc*ag->pccc+
		pps->c_weight_cccc*ag->cccc+
		pps->c_weight_im_ppcc*ag->im_ppcc);
	max_weight=weight;
	min_weight=weight;

	inv_weight=1.0/weight;

	PRAGMA_IVDEP
	for(i=0;i<useful_bins;i++) {
		tmp[i]=((float)pps->power_pp[i]*ag->pp+(float)pps->power_pc[i]*ag->pc+(float)pps->power_cc[i]*ag->cc+(float)pps->power_im_pc[i]*ag->im_pc)*inv_weight;
		}
	}

/* 0 weight can happen due to extreme line veto at low frequencies and small spindowns */
if(min_weight<= args_info.small_weight_ratio_arg*max_weight) {
	set_missing_point_stats(pst);
	return;
	}
	
/* find highest bin */
compute_range_F(tmp, useful_bins, &max_dx, &min_val, &max_dx_bin);

/* Constant for mean(|x|) */
#define SQRT_PI_2   1.253314

#define SQRT_2PI    2.506628

#define UNIV_INV_B  2.0

/* doing everything in one pass and then subtracting does not work due to precision errors if we chance upon a very high max_dx and because float does not keep many digits */

/* there is also a possible issue with normalization, fortunately we have a ready constant now to normalize with: max_dx */


count=0;
sum=0.0;

PRAGMA_IVDEP
for(i=0;i<max_dx_bin-half_window;i++) {
	a=tmp[i];
	sum+=a;
	count++;
	}

PRAGMA_IVDEP
for(i=max_dx_bin+half_window+1;i<useful_bins;i++) {
	a=tmp[i];
	sum+=a;
	count++;
	}

inv_count=1.0/count;
M=sum*inv_count;

/* better normalization now that we have min_val */
inv_normalizer=max_dx-min_val;
if(inv_normalizer<=0.0)inv_normalizer=1.0; /* in case we have all 0 for missing data */
normalizer=1.0/inv_normalizer;

sum_sq=0.0;
sum1=0.0;
sum3=0.0;
sum4=0.0;
sum_abs=0.0;
PRAGMA_IVDEP
for(i=0;i<max_dx_bin-half_window;i++) {
	a=(tmp[i]-M)*normalizer;
	b=a*a;
	/* collect negative first and second moment statistics - these would describe background behaviour */
	if(a<0) {
		sum1-=a;
		sum3-=a*b;
		}
	sum_abs+=fabs(a);
	sum_sq+=b;
	sum4+=b*b;
	}

PRAGMA_IVDEP
for(i=max_dx_bin+half_window+1;i<useful_bins;i++) {
	a=(tmp[i]-M)*normalizer;
	b=a*a;
	/* collect negative first and second moment statistics - these would describe background behaviour */
	if(a<0) {
		sum1-=a;
		sum3-=a*b;
		}
	sum_abs+=fabs(a);
	sum_sq+=b;
	sum4+=b*b;
	}

// sqrt-based
//S=sqrt(sum_sq/(count-1))*max_dx;
//inv_S=1.0/S;

// use lower tail only
S=sum1*inv_normalizer*SQRT_2PI/count;
inv_S=1.0/S;

/* convert to SNR from the highest power */
max_dx=(max_dx-M)*inv_S;

if(max_dx<=0 || !isfinite(max_dx)) {
	/* In theory we could have max_dx=0 because the distribution is flat, but we really should not have this */
	fprintf(stderr, "***ERROR - irregular max_dx  max_dx=%g max_dx_bin=%d M=%g S=%g inv_S=%g tmp=%p tmp={%g %g ... %g %g ...}\n",
			max_dx,
			max_dx_bin,
			M,
			S,
			inv_S,
			tmp, 
			tmp[0], tmp[1], tmp[250], tmp[251]);
	/* this is not fatal - complain, but continue */
	}
	
/* Third pass - compute delta */
	
sum_c=0.0;

PRAGMA_IVDEP
for(i=0;i<max_dx_bin-half_window;i++) {
	a=(M-x_epsilon*S)-tmp[i];
	/* collect negative threshold statistics */
	if(a>=0) {
		sum_c+=a+UNIV_INV_B*S;
		}
	}

PRAGMA_IVDEP
for(i=max_dx_bin+half_window+1;i<useful_bins;i++) {
	a=(M-x_epsilon*S)-tmp[i];
	/* collect negative threshold statistics */
	if(a>=0) {
		sum_c+=a+UNIV_INV_B*S;
		}
	}

// fprintf(stderr, "___ sum_c=%g inv_S=%g count=%d max_dx_bin=%d %f M=%g\n", sum_c*inv_S, inv_S, count, max_dx_bin, UNIV_INV_B, M);
sum_c=sum_c*inv_S/(UNIV_INV_B*count*(1.0-args_info.confidence_level_arg));

if(sum_c<=1)
	s3=x_epsilon;
	else
	s3=x_epsilon+(sum_c-1)*UNIV_INV_B;

pst->bin=max_dx_bin;
pst->iota=ag->iota;
pst->psi=ag->psi;

/* convert to upper limit units */
pst->S=sqrt(S)*strain_comp;
pst->M=sqrt(M)*strain_comp;

pst->ul=sqrt(S*(max_dx+s3))*strain_comp*upper_limit_comp;
/* for debugging store s3 in lower limit variable */
pst->ll=s3;
pst->centroid=sqrt(max_dx*S)*strain_comp*upper_limit_comp;
pst->snr=max_dx;

pst->max_weight=max_weight;
pst->weight_loss_fraction=(max_weight-min_weight)/max_weight;

/* Apply normalization */
S*=normalizer;
//pst->ks_value=(sum4*inv_count-4*sum3*inv_count*M+6*sum_sq*inv_count*M*M-3*M*M*M*M)/(S*S*S*S);
pst->m1_neg=sum_c;
pst->m3_neg=(sum3*inv_count)/(S*S*S);
pst->m4=(sum4*inv_count)/(S*S*S*S);
//fprintf(stderr, "%g = %g %g %g %g (%d %g %g)\n", pst->ks_value, M, sum_sq*inv_count, sum3*inv_count, sum4*inv_count, count, inv_count, S);
pst->ks_value=0;
pst->ks_count=0;
}

void sse_compute_power(PARTIAL_POWER_SUM_F *pps, ALIGNMENT_COEFFS *ag, float *tmp, float *min_weight, float *max_weight)
{
#if MANUAL_SSE
int i;
float *tmp2;
__m128 v4a,v4b, v4c, v4weight, v4tmp, v4_max_weight, v4_min_weight;
float weight, inv_weight;

tmp2=aligned_alloca(4*sizeof(tmp2));
	
if(pps->weight_arrays_non_zero) {
	*max_weight=0;
	*min_weight=1e50;
	
	v4_max_weight=_mm_load1_ps(max_weight);
	v4_min_weight=_mm_load1_ps(min_weight);

	if(!pps->collapsed_weight_arrays) {
		for(i=0;i<useful_bins;i++) {
			pps->weight_pppp[i]+=pps->c_weight_pppp;
			pps->weight_pppc[i]+=pps->c_weight_pppc;
			pps->weight_ppcc[i]+=pps->c_weight_ppcc;
			pps->weight_pccc[i]+=pps->c_weight_pccc;
			pps->weight_cccc[i]+=pps->c_weight_cccc;
			}
		pps->c_weight_pppp=0;
		pps->c_weight_pppc=0;
		pps->c_weight_ppcc=0;
		pps->c_weight_pccc=0;
		pps->c_weight_cccc=0;
		pps->collapsed_weight_arrays=1;
		}

	for(i=0;i<(useful_bins-3);i+=4) {
		/* compute weight */
		v4a=_mm_load_ps(&(pps->weight_pppp[i]));
		v4b=_mm_load1_ps(&ag->pppp);
		v4weight=_mm_mul_ps(v4a, v4b);

		v4a=_mm_load_ps(&(pps->weight_pppc[i]));
		v4b=_mm_load1_ps(&ag->pppc);
		v4c=_mm_mul_ps(v4a, v4b);

		v4weight=_mm_add_ps(v4weight, v4c);

		v4a=_mm_load_ps(&(pps->weight_ppcc[i]));
		v4b=_mm_load1_ps(&ag->ppcc);
		v4c=_mm_mul_ps(v4a, v4b);

		v4weight=_mm_add_ps(v4weight, v4c);

		v4a=_mm_load_ps(&(pps->weight_pccc[i]));
		v4b=_mm_load1_ps(&ag->pccc);
		v4c=_mm_mul_ps(v4a, v4b);

		v4weight=_mm_add_ps(v4weight, v4c);

		v4a=_mm_load_ps(&(pps->weight_cccc[i]));
		v4b=_mm_load1_ps(&ag->cccc);
		v4c=_mm_mul_ps(v4a, v4b);

		v4weight=_mm_add_ps(v4weight, v4c);

		/* update max and min weight variables */

		v4_max_weight=_mm_max_ps(v4_max_weight, v4weight);
		v4_min_weight=_mm_min_ps(v4_min_weight, v4weight);

		/* compute power sum */

		v4a=_mm_load_ps(&(pps->power_pp[i]));
		v4b=_mm_load1_ps(&ag->pp);
		v4tmp=_mm_mul_ps(v4a, v4b);

		v4a=_mm_load_ps(&(pps->power_pc[i]));
		v4b=_mm_load1_ps(&ag->pc);
		v4c=_mm_mul_ps(v4a, v4b);

		v4tmp=_mm_add_ps(v4tmp, v4c);

		v4a=_mm_load_ps(&(pps->power_cc[i]));
		v4b=_mm_load1_ps(&ag->cc);
		v4c=_mm_mul_ps(v4a, v4b);

		v4tmp=_mm_add_ps(v4tmp, v4c);

		if(pps->power_im_pc!=NULL) {
			v4a=_mm_load_ps(&(pps->power_im_pc[i]));
			v4b=_mm_load1_ps(&ag->im_pc);
			v4c=_mm_mul_ps(v4a, v4b);

			v4tmp=_mm_add_ps(v4tmp, v4c);
			}

		v4tmp=_mm_div_ps(v4tmp, v4weight);

		_mm_store_ps(&(tmp[i]), v4tmp);

		}

	_mm_store_ps(tmp2, v4_max_weight);
	
	for(i=0;i<4;i++) {
		weight=tmp2[i];
		if(weight>*max_weight)*max_weight=weight;
		}	

	_mm_store_ps(tmp2, v4_min_weight);
	
	for(i=0;i<4;i++) {
		weight=tmp2[i];
		if(weight<*min_weight)*min_weight=weight;
		}	

	if(pps->power_im_pc!=NULL) {
		for(;i<useful_bins;i++) {
			weight=(pps->weight_pppp[i]*ag->pppp+
				pps->weight_pppc[i]*ag->pppc+
				pps->weight_ppcc[i]*ag->ppcc+
				pps->weight_pccc[i]*ag->pccc+
				pps->weight_cccc[i]*ag->cccc);
		
			if(weight>*max_weight)*max_weight=weight;
			if(weight<*min_weight)*min_weight=weight;

			tmp[i]=(pps->power_pp[i]*ag->pp+pps->power_pc[i]*ag->pc+pps->power_cc[i]*ag->cc+pps->power_im_pc[i]*ag->im_pc)/weight;
				
			}
		} else {
		for(;i<useful_bins;i++) {
			weight=(pps->weight_pppp[i]*ag->pppp+
				pps->weight_pppc[i]*ag->pppc+
				pps->weight_ppcc[i]*ag->ppcc+
				pps->weight_pccc[i]*ag->pccc+
				pps->weight_cccc[i]*ag->cccc);
		
			if(weight>*max_weight)*max_weight=weight;
			if(weight<*min_weight)*min_weight=weight;

			tmp[i]=(pps->power_pp[i]*ag->pp+pps->power_pc[i]*ag->pc+pps->power_cc[i]*ag->cc)/weight;
				
			}
		}

	/* verify */
	#if 0
	if(0){
		float a1, a2, m1, m2;
		int *b1=&a1, *b2=&a2;
		m1=0;
		m2=1e50;
		for(i=0;i<useful_bins;i++) {
			weight=(pps->weight_pppp[i]*ag->pppp+
				pps->weight_pppc[i]*ag->pppc+
				pps->weight_ppcc[i]*ag->ppcc+
				pps->weight_pccc[i]*ag->pccc+
				pps->weight_cccc[i]*ag->cccc);
		
			if(weight>max_weight) fprintf(stderr, "*1*  %d %g %g %g\n", i, weight, max_weight, tmp[i]);
			if(weight<min_weight) fprintf(stderr, "*2*  %d %g %g %g\n", i, weight, min_weight, tmp[i]);
	
			if(weight>m1) m1=weight;
			if(weight<m2) m2=weight;
	
			a1=tmp[i];
			a2=(pps->power_pp[i]*ag->pp+pps->power_pc[i]*ag->pc+pps->power_cc[i]*ag->cc)/weight;
	
			if(*b1!=*b2){
				fprintf(stderr, " *3* %d %g %g %g %g %g\n", i, (pps->power_pp[i]*ag->pp+pps->power_pc[i]*ag->pc+pps->power_cc[i]*ag->cc)/weight, tmp[i], weight, min_weight, max_weight);
				}
				
			}
		a1=m1;
		a2=max_weight;
	
		if(*b1!=*b2){
			fprintf(stderr, " *4* %g %g %g %g\n", m2, m1, min_weight, max_weight);
			}
	
		a1=m2;
		a2=min_weight;
	
		if(*b1!=*b2) {
			fprintf(stderr, " *5* %g %g %g %g\n", m2, m1, min_weight, max_weight);
			}
		}
	#endif
	} else {
	weight=(pps->c_weight_pppp*ag->pppp+
		pps->c_weight_pppc*ag->pppc+
		pps->c_weight_ppcc*ag->ppcc+
		pps->c_weight_pccc*ag->pccc+
		pps->c_weight_cccc*ag->cccc+
		pps->c_weight_im_ppcc*ag->im_ppcc);
	*max_weight=weight;
	*min_weight=weight;

	inv_weight=1.0/weight;

	for(i=0;i<(useful_bins-3);i+=4) {
		/* compute power sum */

		v4a=_mm_load_ps(&(pps->power_pp[i]));
		v4b=_mm_load1_ps(&ag->pp);
		v4tmp=_mm_mul_ps(v4a, v4b);

		v4a=_mm_load_ps(&(pps->power_pc[i]));
		v4b=_mm_load1_ps(&ag->pc);
		v4c=_mm_mul_ps(v4a, v4b);

		v4tmp=_mm_add_ps(v4tmp, v4c);

		v4a=_mm_load_ps(&(pps->power_cc[i]));
		v4b=_mm_load1_ps(&ag->cc);
		v4c=_mm_mul_ps(v4a, v4b);

		v4tmp=_mm_add_ps(v4tmp, v4c);

		if(pps->power_im_pc!=NULL) {
			v4a=_mm_load_ps(&(pps->power_im_pc[i]));
			v4b=_mm_load1_ps(&ag->im_pc);
			v4c=_mm_mul_ps(v4a, v4b);

			v4tmp=_mm_add_ps(v4tmp, v4c);
			}

		v4a=_mm_load1_ps(&inv_weight);

		v4tmp=_mm_mul_ps(v4tmp, v4a);

		_mm_store_ps(&(tmp[i]), v4tmp);
		}

	if(pps->power_im_pc!=NULL) {
		for(;i<useful_bins;i++) {
			tmp[i]=((float)pps->power_pp[i]*ag->pp+(float)pps->power_pc[i]*ag->pc+(float)pps->power_cc[i]*ag->cc+pps->power_im_pc[i]*ag->im_pc)*inv_weight;
			}
		} else {
		for(;i<useful_bins;i++) {
			tmp[i]=((float)pps->power_pp[i]*ag->pp+(float)pps->power_pc[i]*ag->pc+(float)pps->power_cc[i]*ag->cc)*inv_weight;
			}
		}
	}

#else
fprintf(stderr, "**** MANUAL_SSE disabled in %s\n", __FUNCTION__);
exit(-2);
#endif
}

void sse_compute_universal_statistics(float *tmp, float min_weight, float max_weight, ALIGNMENT_COEFFS *ag, POINT_STATS *pst)
{
#if MANUAL_SSE
int i, count;
float M, S, a, b, s3, inv_S, inv_count, normalizer, inv_normalizer;
float max_dx, min_val;
int max_dx_bin;
float sum, sum_sq, sum1, sum3, sum4, sum_c;
int half_window=args_info.half_window_arg;
float *tmp2=NULL;
__m128 v4a,v4b, v4c, v4d, v4tmp, v4sum, v4sum_sq, v4sum3, v4sum4, v4zero;
	
tmp2=aligned_alloca(4*sizeof(*tmp2));

/* 0 weight can happen due to extreme line veto at low frequencies and small spindowns */
if(min_weight<= args_info.small_weight_ratio_arg*max_weight) {
	set_missing_point_stats(pst);
	return;
	}
	
/* find highest bin */
compute_range_F(tmp, useful_bins, &max_dx, &min_val, &max_dx_bin);

/* doing everything in one pass and then subtracting does not work due to precision errors if we chance upon a very high max_dx and because float does not keep many digits */

/* there is also a possible issue with normalization, fortunately we have a ready constant now to normalize with: max_dx */


sum=0.0;

v4sum=_mm_setzero_ps();
for(i=0;i<max_dx_bin-half_window-3;i+=4) {
	v4sum=_mm_add_ps(v4sum, _mm_load_ps(&(tmp[i])));
	}

for(;i<max_dx_bin-half_window;i++) {
	sum+=tmp[i];
	}

count=i;
for(i=max_dx_bin+half_window+1;(i& 3) && (i<useful_bins);i++) {
	sum+=tmp[i];
	}

for(;i<useful_bins-3;i+=4) {
	v4sum=_mm_add_ps(v4sum, _mm_load_ps(&(tmp[i])));
	}

for(;i<useful_bins;i++) {
	sum+=tmp[i];
	}
_mm_store_ps(tmp2, v4sum);
sum+=tmp2[0]+tmp2[1]+tmp2[2]+tmp2[3];

count+=i-max_dx_bin-half_window-1;

inv_count=1.0/count;
M=sum*inv_count;

/* better normalization now that we have min_val */
inv_normalizer=max_dx-min_val;
if(inv_normalizer<=0.0)inv_normalizer=1.0; /* in case we have all 0 for missing data */
normalizer=1.0/inv_normalizer;

sum_sq=0.0;
sum1=0.0;
sum3=0.0;
sum4=0.0;
v4sum=_mm_setzero_ps();
v4sum_sq=_mm_setzero_ps();
v4sum3=_mm_setzero_ps();
v4sum4=_mm_setzero_ps();
v4zero=_mm_setzero_ps();

v4a=_mm_load1_ps(&M);
v4b=_mm_load1_ps(&normalizer);
for(i=0;i<max_dx_bin-half_window-3;i+=4) {
	v4tmp=_mm_mul_ps(_mm_sub_ps(_mm_load_ps(&(tmp[i])), v4a), v4b);
	v4c=_mm_mul_ps(v4tmp, v4tmp);
	/* collect negative first and second moment statistics - these would describe background behaviour */
	v4d=_mm_min_ps(v4tmp, v4zero);
	v4sum=_mm_sub_ps(v4sum, v4d);
	v4sum3=_mm_sub_ps(v4sum3, _mm_mul_ps(v4d, v4c));

	v4sum_sq=_mm_add_ps(v4sum_sq, v4c);
	v4sum4=_mm_add_ps(v4sum4, _mm_mul_ps(v4c, v4c));
	}

for(;i<max_dx_bin-half_window;i++) {
	a=(tmp[i]-M)*normalizer;
	b=a*a;
	/* collect negative first and second moment statistics - these would describe background behaviour */
	if(a<0) {
		sum1-=a;
		sum3-=a*b;
		}
	sum_sq+=b;
	sum4+=b*b;
	}

for(i=max_dx_bin+half_window+1;(i & 3) && (i<useful_bins);i++) {
	a=(tmp[i]-M)*normalizer;
	b=a*a;
	/* collect negative first and second moment statistics - these would describe background behaviour */
	if(a<0) {
		sum1-=a;
		sum3-=a*b;
		}
	sum_sq+=b;
	sum4+=b*b;
	}

for(;i<useful_bins-3;i+=4) {
	v4tmp=_mm_mul_ps(_mm_sub_ps(_mm_load_ps(&(tmp[i])), v4a), v4b);
	v4c=_mm_mul_ps(v4tmp, v4tmp);
	/* collect negative first and second moment statistics - these would describe background behaviour */
	v4d=_mm_min_ps(v4tmp, v4zero);
	v4sum=_mm_sub_ps(v4sum, v4d);
	v4sum3=_mm_sub_ps(v4sum3, _mm_mul_ps(v4d, v4c));

	v4sum_sq=_mm_add_ps(v4sum_sq, v4c);
	v4sum4=_mm_add_ps(v4sum4, _mm_mul_ps(v4c, v4c));
	}

for(;i<useful_bins;i++) {
	a=(tmp[i]-M)*normalizer;
	b=a*a;
	/* collect negative first and second moment statistics - these would describe background behaviour */
	if(a<0) {
		sum1-=a;
		sum3-=a*b;
		}
	sum_sq+=b;
	sum4+=b*b;
	}

_mm_store_ps(tmp2, v4sum);
sum1+=tmp2[0]+tmp2[1]+tmp2[2]+tmp2[3];

_mm_store_ps(tmp2, v4sum_sq);
sum_sq+=tmp2[0]+tmp2[1]+tmp2[2]+tmp2[3];

_mm_store_ps(tmp2, v4sum3);
sum3+=tmp2[0]+tmp2[1]+tmp2[2]+tmp2[3];

_mm_store_ps(tmp2, v4sum4);
sum4+=tmp2[0]+tmp2[1]+tmp2[2]+tmp2[3];

// S=sqrt(sum_sq/(count-1))*max_dx;
S=sum1*inv_normalizer*SQRT_2PI/count;
inv_S=1.0/S;

/* convert to SNR from the highest power */
max_dx=(max_dx-M)*inv_S;

if(max_dx<=0 || !isfinite(max_dx)) {
	/* In theory we could have max_dx=0 because the distribution is flat, but we really should not have this */
	fprintf(stderr, "***ERROR - irregular max_dx  max_dx=%g max_dx_bin=%d M=%g S=%g inv_S=%g tmp=%p tmp={%g %g ... %g %g ...}\n",
			max_dx,
			max_dx_bin,
			M,
			S,
			inv_S,
			tmp, 
			tmp[0], tmp[1], tmp[250], tmp[251]);
	/* this is not fatal - complain, but continue */
	}

/* Third pass - compute delta */

/* Original code */

// sum_c=0.0;
// for(i=0;i<max_dx_bin-half_window;i++) {
// 	a=(M-tmp[i])-x_epsilon*S;
// 	/* collect negative threshold statistics */
// 	if(a>=0) {
// 		sum_c+=a+UNIV_INV_B*S;
// 		}
// 	}
// 
// for(i=max_dx_bin+half_window+1;i<useful_bins;i++) {
// 	a=(M-tmp[i])-x_epsilon*S;
// 	/* collect negative threshold statistics */
// 	if(a>=0) {
// 		sum_c+=a+UNIV_INV_B*S;
// 		}
// 	}

a=M-x_epsilon*S;
b=UNIV_INV_B*S;

v4a=_mm_load1_ps(&a);
v4b=_mm_load1_ps(&b);
v4sum=_mm_setzero_ps();

sum_c=0;
for(i=0;i<max_dx_bin-half_window-3;i+=4) {
	v4tmp=_mm_sub_ps(v4a, _mm_load_ps(&(tmp[i])));
	v4sum=_mm_add_ps(v4sum, _mm_add_ps(_mm_max_ps(v4tmp, v4zero), _mm_and_ps(_mm_cmpge_ps(v4tmp, v4zero), v4b)));
	}

for(;i<max_dx_bin-half_window;i++) {
	a=(M-x_epsilon*S)-tmp[i];
	/* collect negative threshold statistics */
	if(a>=0) {
		sum_c+=a+UNIV_INV_B*S;
		}
	}

for(i=max_dx_bin+half_window+1;(i & 3) && (i<useful_bins);i++) {
	a=(M-x_epsilon*S)-tmp[i];
	/* collect negative threshold statistics */
	if(a>=0) {
		sum_c+=a+UNIV_INV_B*S;
		}
	}

for(;i<useful_bins-3;i+=4) {
	v4tmp=_mm_sub_ps(v4a, _mm_load_ps(&(tmp[i])));
	v4sum=_mm_add_ps(v4sum, _mm_add_ps(_mm_max_ps(v4tmp, v4zero), _mm_and_ps(_mm_cmpge_ps(v4tmp, v4zero), v4b)));
	}
	
for(;i<useful_bins;i++) {
	a=(M-x_epsilon*S)-tmp[i];
	/* collect negative threshold statistics */
	if(a>=0) {
		sum_c+=a+UNIV_INV_B*S;
		}
	}

_mm_store_ps(tmp2, v4sum);
//fprintf(stderr, "sum_c=%g tmp2 %g %g %g %g max_dx_bin=%d\n", sum_c, tmp2[0], tmp2[1], tmp2[2], tmp2[3], max_dx_bin);
sum_c+=tmp2[0]+tmp2[1]+tmp2[2]+tmp2[3];

//fprintf(stderr, "sse sum_c=%g inv_S=%g count=%d max_dx_bin=%d %f M=%g\n", sum_c*inv_S, inv_S, count, max_dx_bin, UNIV_INV_B, M);

sum_c=sum_c*inv_S/(UNIV_INV_B*count*(1.0-args_info.confidence_level_arg));

if(sum_c<=1)
	s3=x_epsilon;
	else
	s3=x_epsilon+(sum_c-1)*UNIV_INV_B;

pst->bin=max_dx_bin;
pst->iota=ag->iota;
pst->psi=ag->psi;

/* convert to upper limit units */
pst->S=sqrt(S)*strain_comp;
pst->M=sqrt(M)*strain_comp;

pst->ul=sqrt(S*(max_dx+s3))*strain_comp*upper_limit_comp;
/* for debugging store s3 in lower limit variable */
pst->ll=s3;
pst->centroid=sqrt(max_dx*S)*strain_comp*upper_limit_comp;
pst->snr=max_dx;

pst->max_weight=max_weight;
pst->weight_loss_fraction=(max_weight-min_weight)/max_weight;

/* Apply normalization */
S*=normalizer;
//pst->ks_value=(sum4*inv_count-4*sum3*inv_count*M+6*sum_sq*inv_count*M*M-3*M*M*M*M)/(S*S*S*S);
pst->m1_neg=sum_c;
pst->m3_neg=(sum3*inv_count)/(S*S*S);
pst->m4=(sum4*inv_count)/(S*S*S*S);
//fprintf(stderr, "%g = %g %g %g %g (%d %g %g)\n", pst->ks_value, M, sum_sq*inv_count, sum3*inv_count, sum4*inv_count, count, inv_count, S);
pst->ks_value=0;
pst->ks_count=0;
#else
fprintf(stderr, "**** MANUAL_SSE disabled in %s\n", __FUNCTION__);
exit(-2);
#endif
}

void sse_point_power_sum_stats_universal_piecewise_linear(PARTIAL_POWER_SUM_F *pps, ALIGNMENT_COEFFS *ag, POINT_STATS *pst)
{
float *tmp=NULL;
float min_weight, max_weight;

/* allocate on stack, for speed */
tmp=aligned_alloca(useful_bins*sizeof(*tmp));

sse_compute_power(pps, ag, tmp, &min_weight, &max_weight);
sse_compute_universal_statistics(tmp, min_weight, max_weight, ag, pst);
}

void sse_point_power_sum_stats_universal_piecewise_linearA(PARTIAL_POWER_SUM_F *pps, ALIGNMENT_COEFFS *ag, POINT_STATS *pst)
{
#if MANUAL_SSE
int i, count;
float M, S, a, b, s3, inv_S, inv_weight, inv_count, normalizer, inv_normalizer;
float *tmp=NULL;
NORMAL_STATS nstats;
float max_dx, min_val;
int max_dx_bin;
float weight, min_weight, max_weight;
float sum, sum_sq, sum1, sum3, sum4, sum_c;
int half_window=args_info.half_window_arg;
float *tmp2=NULL;
__m128 v4a,v4b, v4c, v4d, v4weight, v4tmp, v4sum, v4sum_sq, v4sum3, v4sum4, v4zero, v4_max_weight, v4_min_weight;

/* allocate on stack, for speed */
tmp=aligned_alloca(useful_bins*sizeof(*tmp));
tmp2=aligned_alloca(4*sizeof(tmp2));

memset(&nstats, 0, sizeof(nstats));

/* sort to compute robust estimates */
nstats.flag= STAT_FLAG_ESTIMATE_MEAN
	| STAT_FLAG_ESTIMATE_SIGMA;

if(args_info.ks_test_arg){
	nstats.flag|=STAT_FLAG_ESTIMATE_KS_LEVEL
		| STAT_FLAG_COMPUTE_KS_TEST;
	}


if(pps->power_im_pc==NULL) {
	fprintf(stderr, "*** INTERNAL ERROR: %s requires pps->power_im_pc!=NULL\n", __FUNCTION__);
	exit(-1);
	}

if(pps->weight_arrays_non_zero) {
	max_weight=0;
	min_weight=1e50;
	
	v4_max_weight=_mm_load1_ps(&max_weight);
	v4_min_weight=_mm_load1_ps(&min_weight);

	if(!pps->collapsed_weight_arrays) {
		for(i=0;i<useful_bins;i++) {
			pps->weight_pppp[i]+=pps->c_weight_pppp;
			pps->weight_pppc[i]+=pps->c_weight_pppc;
			pps->weight_ppcc[i]+=pps->c_weight_ppcc;
			pps->weight_pccc[i]+=pps->c_weight_pccc;
			pps->weight_cccc[i]+=pps->c_weight_cccc;
			}
		pps->c_weight_pppp=0;
		pps->c_weight_pppc=0;
		pps->c_weight_ppcc=0;
		pps->c_weight_pccc=0;
		pps->c_weight_cccc=0;
		pps->collapsed_weight_arrays=1;
		}

	for(i=0;i<(useful_bins-3);i+=4) {
		/* compute weight */
		v4a=_mm_load_ps(&(pps->weight_pppp[i]));
		v4b=_mm_load1_ps(&ag->pppp);
		v4weight=_mm_mul_ps(v4a, v4b);

		v4a=_mm_load_ps(&(pps->weight_pppc[i]));
		v4b=_mm_load1_ps(&ag->pppc);
		v4c=_mm_mul_ps(v4a, v4b);

		v4weight=_mm_add_ps(v4weight, v4c);

		v4a=_mm_load_ps(&(pps->weight_ppcc[i]));
		v4b=_mm_load1_ps(&ag->ppcc);
		v4c=_mm_mul_ps(v4a, v4b);

		v4weight=_mm_add_ps(v4weight, v4c);

		v4a=_mm_load_ps(&(pps->weight_pccc[i]));
		v4b=_mm_load1_ps(&ag->pccc);
		v4c=_mm_mul_ps(v4a, v4b);

		v4weight=_mm_add_ps(v4weight, v4c);

		v4a=_mm_load_ps(&(pps->weight_cccc[i]));
		v4b=_mm_load1_ps(&ag->cccc);
		v4c=_mm_mul_ps(v4a, v4b);

		v4weight=_mm_add_ps(v4weight, v4c);

		/* update max and min weight variables */

		v4_max_weight=_mm_max_ps(v4_max_weight, v4weight);
		v4_min_weight=_mm_min_ps(v4_min_weight, v4weight);

		/* compute power sum */

		v4a=_mm_load_ps(&(pps->power_pp[i]));
		v4b=_mm_load1_ps(&ag->pp);
		v4tmp=_mm_mul_ps(v4a, v4b);

		v4a=_mm_load_ps(&(pps->power_pc[i]));
		v4b=_mm_load1_ps(&ag->pc);
		v4c=_mm_mul_ps(v4a, v4b);

		v4tmp=_mm_add_ps(v4tmp, v4c);

		v4a=_mm_load_ps(&(pps->power_cc[i]));
		v4b=_mm_load1_ps(&ag->cc);
		v4c=_mm_mul_ps(v4a, v4b);

		v4tmp=_mm_add_ps(v4tmp, v4c);

		v4a=_mm_load_ps(&(pps->power_im_pc[i]));
		v4b=_mm_load1_ps(&ag->im_pc);
		v4c=_mm_mul_ps(v4a, v4b);

		v4tmp=_mm_add_ps(v4tmp, v4c);

		v4tmp=_mm_div_ps(v4tmp, v4weight);

		_mm_store_ps(&(tmp[i]), v4tmp);

		}

	_mm_store_ps(tmp2, v4_max_weight);
	
	for(i=0;i<4;i++) {
		weight=tmp2[i];
		if(weight>max_weight)max_weight=weight;
		}	

	_mm_store_ps(tmp2, v4_min_weight);
	
	for(i=0;i<4;i++) {
		weight=tmp2[i];
		if(weight<min_weight)min_weight=weight;
		}	

	for(;i<useful_bins;i++) {
		weight=(pps->weight_pppp[i]*ag->pppp+
			pps->weight_pppc[i]*ag->pppc+
			pps->weight_ppcc[i]*ag->ppcc+
			pps->weight_pccc[i]*ag->pccc+
			pps->weight_cccc[i]*ag->cccc);
	
		if(weight>max_weight)max_weight=weight;
		if(weight<min_weight)min_weight=weight;

		tmp[i]=(pps->power_pp[i]*ag->pp+pps->power_pc[i]*ag->pc+pps->power_cc[i]*ag->cc+pps->power_im_pc[i]*ag->im_pc)/weight;
			
		}

	/* verify */
	#if 0
	if(0){
		float a1, a2, m1, m2;
		int *b1=&a1, *b2=&a2;
		m1=0;
		m2=1e50;
		for(i=0;i<useful_bins;i++) {
			weight=(pps->weight_pppp[i]*ag->pppp+
				pps->weight_pppc[i]*ag->pppc+
				pps->weight_ppcc[i]*ag->ppcc+
				pps->weight_pccc[i]*ag->pccc+
				pps->weight_cccc[i]*ag->cccc);
		
			if(weight>max_weight) fprintf(stderr, "*1*  %d %g %g %g\n", i, weight, max_weight, tmp[i]);
			if(weight<min_weight) fprintf(stderr, "*2*  %d %g %g %g\n", i, weight, min_weight, tmp[i]);
	
			if(weight>m1) m1=weight;
			if(weight<m2) m2=weight;
	
			a1=tmp[i];
			a2=(pps->power_pp[i]*ag->pp+pps->power_pc[i]*ag->pc+pps->power_cc[i]*ag->cc)/weight;
	
			if(*b1!=*b2){
				fprintf(stderr, " *3* %d %g %g %g %g %g\n", i, (pps->power_pp[i]*ag->pp+pps->power_pc[i]*ag->pc+pps->power_cc[i]*ag->cc)/weight, tmp[i], weight, min_weight, max_weight);
				}
				
			}
		a1=m1;
		a2=max_weight;
	
		if(*b1!=*b2){
			fprintf(stderr, " *4* %g %g %g %g\n", m2, m1, min_weight, max_weight);
			}
	
		a1=m2;
		a2=min_weight;
	
		if(*b1!=*b2) {
			fprintf(stderr, " *5* %g %g %g %g\n", m2, m1, min_weight, max_weight);
			}
		}
	#endif
	} else {
	weight=(pps->c_weight_pppp*ag->pppp+
		pps->c_weight_pppc*ag->pppc+
		pps->c_weight_ppcc*ag->ppcc+
		pps->c_weight_pccc*ag->pccc+
		pps->c_weight_cccc*ag->cccc+
		pps->c_weight_im_ppcc*ag->im_ppcc);
	max_weight=weight;
	min_weight=weight;

	inv_weight=1.0/weight;

	for(i=0;i<(useful_bins-3);i+=4) {
		/* compute power sum */

		v4a=_mm_load_ps(&(pps->power_pp[i]));
		v4b=_mm_load1_ps(&ag->pp);
		v4tmp=_mm_mul_ps(v4a, v4b);

		v4a=_mm_load_ps(&(pps->power_pc[i]));
		v4b=_mm_load1_ps(&ag->pc);
		v4c=_mm_mul_ps(v4a, v4b);

		v4tmp=_mm_add_ps(v4tmp, v4c);

		v4a=_mm_load_ps(&(pps->power_cc[i]));
		v4b=_mm_load1_ps(&ag->cc);
		v4c=_mm_mul_ps(v4a, v4b);

		v4tmp=_mm_add_ps(v4tmp, v4c);

		v4a=_mm_load_ps(&(pps->power_im_pc[i]));
		v4b=_mm_load1_ps(&ag->im_pc);
		v4c=_mm_mul_ps(v4a, v4b);

		v4tmp=_mm_add_ps(v4tmp, v4c);

		v4a=_mm_load1_ps(&inv_weight);

		v4tmp=_mm_mul_ps(v4tmp, v4a);

		_mm_store_ps(&(tmp[i]), v4tmp);
		}

	for(;i<useful_bins;i++) {
		tmp[i]=((float)pps->power_pp[i]*ag->pp+(float)pps->power_pc[i]*ag->pc+(float)pps->power_cc[i]*ag->cc+pps->power_im_pc[i]*ag->im_pc)*inv_weight;
		}
	}

/* 0 weight can happen due to extreme line veto at low frequencies and small spindowns */
if(min_weight<= args_info.small_weight_ratio_arg*max_weight) {
	set_missing_point_stats(pst);
	return;
	}
	
/* find highest bin */
compute_range_F(tmp, useful_bins, &max_dx, &min_val, &max_dx_bin);

/* doing everything in one pass and then subtracting does not work due to precision errors if we chance upon a very high max_dx and because float does not keep many digits */

/* there is also a possible issue with normalization, fortunately we have a ready constant now to normalize with: max_dx */


sum=0.0;

v4sum=_mm_setzero_ps();
for(i=0;i<max_dx_bin-half_window-3;i+=4) {
	v4sum=_mm_add_ps(v4sum, _mm_load_ps(&(tmp[i])));
	}

for(;i<max_dx_bin-half_window;i++) {
	sum+=tmp[i];
	}

count=i;
for(i=max_dx_bin+half_window+1;(i& 3) && (i<useful_bins);i++) {
	sum+=tmp[i];
	}

for(;i<useful_bins-3;i+=4) {
	v4sum=_mm_add_ps(v4sum, _mm_load_ps(&(tmp[i])));
	}

for(;i<useful_bins;i++) {
	sum+=tmp[i];
	}
_mm_store_ps(tmp2, v4sum);
sum+=tmp2[0]+tmp2[1]+tmp2[2]+tmp2[3];

count+=i-max_dx_bin-half_window-1;

inv_count=1.0/count;
M=sum*inv_count;

/* better normalization now that we have min_val */
inv_normalizer=max_dx-min_val;
if(inv_normalizer<=0.0)inv_normalizer=1.0; /* in case we have all 0 for missing data */
normalizer=1.0/inv_normalizer;

sum_sq=0.0;
sum1=0.0;
sum3=0.0;
sum4=0.0;
v4sum=_mm_setzero_ps();
v4sum_sq=_mm_setzero_ps();
v4sum3=_mm_setzero_ps();
v4sum4=_mm_setzero_ps();
v4zero=_mm_setzero_ps();

v4a=_mm_load1_ps(&M);
v4b=_mm_load1_ps(&normalizer);
for(i=0;i<max_dx_bin-half_window-3;i+=4) {
	v4tmp=_mm_mul_ps(_mm_sub_ps(_mm_load_ps(&(tmp[i])), v4a), v4b);
	v4c=_mm_mul_ps(v4tmp, v4tmp);
	/* collect negative first and second moment statistics - these would describe background behaviour */
	v4d=_mm_min_ps(v4tmp, v4zero);
	v4sum=_mm_sub_ps(v4sum, v4d);
	v4sum3=_mm_sub_ps(v4sum3, _mm_mul_ps(v4d, v4c));

	v4sum_sq=_mm_add_ps(v4sum_sq, v4c);
	v4sum4=_mm_add_ps(v4sum4, _mm_mul_ps(v4c, v4c));
	}

for(;i<max_dx_bin-half_window;i++) {
	a=(tmp[i]-M)*normalizer;
	b=a*a;
	/* collect negative first and second moment statistics - these would describe background behaviour */
	if(a<0) {
		sum1-=a;
		sum3-=a*b;
		}
	sum_sq+=b;
	sum4+=b*b;
	}

for(i=max_dx_bin+half_window+1;(i & 3) && (i<useful_bins);i++) {
	a=(tmp[i]-M)*normalizer;
	b=a*a;
	/* collect negative first and second moment statistics - these would describe background behaviour */
	if(a<0) {
		sum1-=a;
		sum3-=a*b;
		}
	sum_sq+=b;
	sum4+=b*b;
	}

for(;i<useful_bins-3;i+=4) {
	v4tmp=_mm_mul_ps(_mm_sub_ps(_mm_load_ps(&(tmp[i])), v4a), v4b);
	v4c=_mm_mul_ps(v4tmp, v4tmp);
	/* collect negative first and second moment statistics - these would describe background behaviour */
	v4d=_mm_min_ps(v4tmp, v4zero);
	v4sum=_mm_sub_ps(v4sum, v4d);
	v4sum3=_mm_sub_ps(v4sum3, _mm_mul_ps(v4d, v4c));

	v4sum_sq=_mm_add_ps(v4sum_sq, v4c);
	v4sum4=_mm_add_ps(v4sum4, _mm_mul_ps(v4c, v4c));
	}

for(;i<useful_bins;i++) {
	a=(tmp[i]-M)*normalizer;
	b=a*a;
	/* collect negative first and second moment statistics - these would describe background behaviour */
	if(a<0) {
		sum1-=a;
		sum3-=a*b;
		}
	sum_sq+=b;
	sum4+=b*b;
	}

_mm_store_ps(tmp2, v4sum);
sum1+=tmp2[0]+tmp2[1]+tmp2[2]+tmp2[3];

_mm_store_ps(tmp2, v4sum_sq);
sum_sq+=tmp2[0]+tmp2[1]+tmp2[2]+tmp2[3];

_mm_store_ps(tmp2, v4sum3);
sum3+=tmp2[0]+tmp2[1]+tmp2[2]+tmp2[3];

_mm_store_ps(tmp2, v4sum4);
sum4+=tmp2[0]+tmp2[1]+tmp2[2]+tmp2[3];

// S=sqrt(sum_sq/(count-1))*max_dx;
S=sum1*inv_normalizer*SQRT_2PI/count;
inv_S=1.0/S;

/* convert to SNR from the highest power */
max_dx=(max_dx-M)*inv_S;

if(max_dx<=0 || !isfinite(max_dx)) {
	/* In theory we could have max_dx=0 because the distribution is flat, but we really should not have this */
	fprintf(stderr, "***ERROR - irregular max_dx  max_dx=%g max_dx_bin=%d M=%g S=%g inv_S=%g tmp=%p tmp={%g %g ... %g %g ...}\n",
			max_dx,
			max_dx_bin,
			M,
			S,
			inv_S,
			tmp, 
			tmp[0], tmp[1], tmp[250], tmp[251]);
	/* this is not fatal - complain, but continue */
	}

/* Third pass - compute delta */

/* Original code */

// sum_c=0.0;
// for(i=0;i<max_dx_bin-half_window;i++) {
// 	a=(M-tmp[i])-x_epsilon*S;
// 	/* collect negative threshold statistics */
// 	if(a>=0) {
// 		sum_c+=a+UNIV_INV_B*S;
// 		}
// 	}
// 
// for(i=max_dx_bin+half_window+1;i<useful_bins;i++) {
// 	a=(M-tmp[i])-x_epsilon*S;
// 	/* collect negative threshold statistics */
// 	if(a>=0) {
// 		sum_c+=a+UNIV_INV_B*S;
// 		}
// 	}

a=M-x_epsilon*S;
b=UNIV_INV_B*S;

v4a=_mm_load1_ps(&a);
v4b=_mm_load1_ps(&b);
v4sum=_mm_setzero_ps();

sum_c=0;
for(i=0;i<max_dx_bin-half_window-3;i+=4) {
	v4tmp=_mm_sub_ps(v4a, _mm_load_ps(&(tmp[i])));
	v4sum=_mm_add_ps(v4sum, _mm_add_ps(_mm_max_ps(v4tmp, v4zero), _mm_and_ps(_mm_cmpge_ps(v4tmp, v4zero), v4b)));
	}

for(;i<max_dx_bin-half_window;i++) {
	a=(M-x_epsilon*S)-tmp[i];
	/* collect negative threshold statistics */
	if(a>=0) {
		sum_c+=a+UNIV_INV_B*S;
		}
	}

for(i=max_dx_bin+half_window+1;(i & 3) && (i<useful_bins);i++) {
	a=(M-x_epsilon*S)-tmp[i];
	/* collect negative threshold statistics */
	if(a>=0) {
		sum_c+=a+UNIV_INV_B*S;
		}
	}

for(;i<useful_bins-3;i+=4) {
	v4tmp=_mm_sub_ps(v4a, _mm_load_ps(&(tmp[i])));
	v4sum=_mm_add_ps(v4sum, _mm_add_ps(_mm_max_ps(v4tmp, v4zero), _mm_and_ps(_mm_cmpge_ps(v4tmp, v4zero), v4b)));
	}
	
for(;i<useful_bins;i++) {
	a=(M-x_epsilon*S)-tmp[i];
	/* collect negative threshold statistics */
	if(a>=0) {
		sum_c+=a+UNIV_INV_B*S;
		}
	}

_mm_store_ps(tmp2, v4sum);
//fprintf(stderr, "sum_c=%g tmp2 %g %g %g %g max_dx_bin=%d\n", sum_c, tmp2[0], tmp2[1], tmp2[2], tmp2[3], max_dx_bin);
sum_c+=tmp2[0]+tmp2[1]+tmp2[2]+tmp2[3];

//fprintf(stderr, "sse sum_c=%g inv_S=%g count=%d max_dx_bin=%d %f M=%g\n", sum_c*inv_S, inv_S, count, max_dx_bin, UNIV_INV_B, M);

sum_c=sum_c*inv_S/(UNIV_INV_B*count*(1.0-args_info.confidence_level_arg));

if(sum_c<=1)
	s3=x_epsilon;
	else
	s3=x_epsilon+(sum_c-1)*UNIV_INV_B;

pst->bin=max_dx_bin;
pst->iota=ag->iota;
pst->psi=ag->psi;

/* convert to upper limit units */
pst->S=sqrt(S)*strain_comp;
pst->M=sqrt(M)*strain_comp;

pst->ul=sqrt(S*(max_dx+s3))*strain_comp*upper_limit_comp;
/* for debugging store s3 in lower limit variable */
pst->ll=s3;
pst->centroid=sqrt(max_dx*S)*strain_comp*upper_limit_comp;
pst->snr=max_dx;

pst->max_weight=max_weight;
pst->weight_loss_fraction=(max_weight-min_weight)/max_weight;

/* Apply normalization */
S*=normalizer;
//pst->ks_value=(sum4*inv_count-4*sum3*inv_count*M+6*sum_sq*inv_count*M*M-3*M*M*M*M)/(S*S*S*S);
pst->m1_neg=sum_c;
pst->m3_neg=(sum3*inv_count)/(S*S*S);
pst->m4=(sum4*inv_count)/(S*S*S*S);
//fprintf(stderr, "%g = %g %g %g %g (%d %g %g)\n", pst->ks_value, M, sum_sq*inv_count, sum3*inv_count, sum4*inv_count, count, inv_count, S);
pst->ks_value=0;
pst->ks_count=0;
#else
fprintf(stderr, "**** MANUAL_SSE disabled in %s\n", __FUNCTION__);
exit(-2);
#endif
}

#if MANUAL_SSE
void (*point_power_sum_stats)(PARTIAL_POWER_SUM_F *pps, ALIGNMENT_COEFFS *ag, POINT_STATS *pst)=sse_point_power_sum_stats_linear;
#else
void (*point_power_sum_stats)(PARTIAL_POWER_SUM_F *pps, ALIGNMENT_COEFFS *ag, POINT_STATS *pst)=point_power_sum_stats_linear;
#endif

void prepare_power_sum_stats(POWER_SUM_STATS *stats)
{
stats->highest_ul.ul=-1;
stats->highest_circ_ul.ul=-1;
stats->highest_snr.snr=-1;
stats->highest_ks.ks_value=-1;
stats->highest_M.M=-1;
stats->highest_S.S=-1;
stats->max_weight_loss_fraction=-1;
stats->max_weight=-1;
stats->min_weight=1e50;
stats->max_m1_neg=-1;
stats->min_m1_neg=1e50;
stats->max_m3_neg=-1;
stats->min_m3_neg=1e50;
stats->max_m4=-1;
stats->min_m4=1e50;
stats->ntemplates=0;
}

void update_power_sum_stats(POINT_STATS *pst, ALIGNMENT_COEFFS *ag, POWER_SUM_STATS *stats)
{
stats->ntemplates++;

if(pst->snr>stats->highest_snr.snr) {
	memcpy(&(stats->highest_snr), pst, sizeof(*pst));
	}

if(pst->ul>stats->highest_ul.ul) {
	memcpy(&(stats->highest_ul), pst, sizeof(*pst));
	}

if(pst->ks_value>stats->highest_ks.ks_value) {
	memcpy(&(stats->highest_ks), pst, sizeof(*pst));
	}

if(pst->M>stats->highest_M.M) {
	memcpy(&(stats->highest_M), pst, sizeof(*pst));
	}

if(pst->S>stats->highest_S.S) {
	memcpy(&(stats->highest_S), pst, sizeof(*pst));
	}

/* Let us consider anything with iota < 1e-5 as circular. 
	In practice this should only be one point */
if(ag->iota<1e-5) {
	memcpy(&(stats->highest_circ_ul), pst, sizeof(*pst));
	}

if(pst->max_weight>stats->max_weight)stats->max_weight=pst->max_weight;
if(pst->max_weight<stats->min_weight)stats->min_weight=pst->max_weight;
if(pst->weight_loss_fraction>stats->max_weight_loss_fraction)stats->max_weight_loss_fraction=pst->weight_loss_fraction;

if(pst->m1_neg>stats->max_m1_neg)stats->max_m1_neg=pst->m1_neg;
if(pst->m1_neg<stats->min_m1_neg)stats->min_m1_neg=pst->m1_neg;
if(pst->m3_neg>stats->max_m3_neg)stats->max_m3_neg=pst->m3_neg;
if(pst->m3_neg<stats->min_m3_neg)stats->min_m3_neg=pst->m3_neg;
if(pst->m4>stats->max_m4)stats->max_m4=pst->m4;
if(pst->m4<stats->min_m4)stats->min_m4=pst->m4;
}

void power_sum_stats(PARTIAL_POWER_SUM_F *pps, POWER_SUM_STATS *stats)
{
int k;
POINT_STATS pst;

prepare_power_sum_stats(stats);

for(k=0;k<alignment_grid_free;k++) {
	memset(&pst, 0, sizeof(pst));

	point_power_sum_stats(pps, &(alignment_grid[k]), &(pst));

	if(pst.ul<0) {
		/* propagate masked points */
		stats->max_weight_loss_fraction=1.0;

		/* abort computation at this point. This has the benefit of reducing computation time for highly contaminated bands which would also produce a lot of high SNR outliers */
		return;
		continue;
		}
		
	update_power_sum_stats(&pst, &(alignment_grid[k]), stats);
	}
}

void power_sum_stats_selftest(void)
{
PARTIAL_POWER_SUM_F *ps1, *ps2;
POINT_STATS pst_ref, pst_test;
int k;
int result=0;

memset(&pst_ref, 0, sizeof(pst_ref));
memset(&pst_test, 0, sizeof(pst_test));

ps1=allocate_partial_power_sum_F(useful_bins, 1);
ps2=allocate_partial_power_sum_F(useful_bins, 1);

randomize_partial_power_sum_F(ps1);

/* make sure we don't get negative power sums */

ps1->c_weight_pppp+=1000;
ps1->c_weight_ppcc+= 300;
ps1->c_weight_cccc+=1000;

for(k=0;k<ps1->nbins;k++) {
	ps1->power_pp[k]+=100;
	ps1->power_cc[k]+=110;
	}


for(k=0;k<alignment_grid_free;k++) {

	/* reference */
	zero_partial_power_sum_F(ps2);
	accumulate_partial_power_sum_F(ps2, ps1);
	point_power_sum_stats_linear(ps2, &(alignment_grid[k]), &(pst_ref));

	/* cblas */
	zero_partial_power_sum_F(ps2);
	accumulate_partial_power_sum_F(ps2, ps1);
	cblas_point_power_sum_stats_linear(ps2, &(alignment_grid[k]), &(pst_test));
	result+=compare_point_stats("cblas1:", &pst_ref, &pst_test);

	/* sse */
#if MANUAL_SSE
	zero_partial_power_sum_F(ps2);
	accumulate_partial_power_sum_F(ps2, ps1);
	sse_point_power_sum_stats_linear(ps2, &(alignment_grid[k]), &(pst_test));
	result+=compare_point_stats("sse1:", &pst_ref, &pst_test);
#endif
	
	/* piecewise linear stats */
	zero_partial_power_sum_F(ps2);
	accumulate_partial_power_sum_F(ps2, ps1);
	point_power_sum_stats_universal_piecewise_linear(ps2, &(alignment_grid[k]), &(pst_ref));

#if MANUAL_SSE
	zero_partial_power_sum_F(ps2);
	accumulate_partial_power_sum_F(ps2, ps1);
	sse_point_power_sum_stats_universal_piecewise_linear(ps2, &(alignment_grid[k]), &(pst_test));
	result+=compare_point_stats_universal("universal sse1:", &pst_ref, &pst_test);
#endif
	
	}

ps1->weight_arrays_non_zero=0;

for(k=0;k<alignment_grid_free;k++) {
	zero_partial_power_sum_F(ps2);
	accumulate_partial_power_sum_F(ps2, ps1);

	/* reference */
	point_power_sum_stats_linear(ps2, &(alignment_grid[k]), &(pst_ref));

	/* cblas */
	zero_partial_power_sum_F(ps2);
	accumulate_partial_power_sum_F(ps2, ps1);
	cblas_point_power_sum_stats_linear(ps2, &(alignment_grid[k]), &(pst_test));
	result+=compare_point_stats("cblas2:", &pst_ref, &pst_test);

	/* sse */
#if MANUAL_SSE
	zero_partial_power_sum_F(ps2);
	accumulate_partial_power_sum_F(ps2, ps1);
	sse_point_power_sum_stats_linear(ps2, &(alignment_grid[k]), &(pst_test));
	result+=compare_point_stats("sse2:", &pst_ref, &pst_test);
#endif

	/* piecewise linear stats */
	zero_partial_power_sum_F(ps2);
	accumulate_partial_power_sum_F(ps2, ps1);
	point_power_sum_stats_universal_piecewise_linear(ps2, &(alignment_grid[k]), &(pst_ref));

	zero_partial_power_sum_F(ps2);
	accumulate_partial_power_sum_F(ps2, ps1);
	point_power_sum_stats_universal_piecewise_linearA(ps2, &(alignment_grid[k]), &(pst_test));
	result+=compare_point_stats_universal("universalA:", &pst_ref, &pst_test);
	
#if MANUAL_SSE
	zero_partial_power_sum_F(ps2);
	accumulate_partial_power_sum_F(ps2, ps1);
	sse_point_power_sum_stats_universal_piecewise_linear(ps2, &(alignment_grid[k]), &(pst_test));
	result+=compare_point_stats_universal("universal sse2:", &pst_ref, &pst_test);
	
	zero_partial_power_sum_F(ps2);
	accumulate_partial_power_sum_F(ps2, ps1);
	sse_point_power_sum_stats_universal_piecewise_linearA(ps2, &(alignment_grid[k]), &(pst_ref));
	result+=compare_point_stats_universal("universalA sse2:", &pst_ref, &pst_test);
#endif

	fprintf(stderr, "%d %f %f %d\n", k, alignment_grid[k].iota, alignment_grid[k].psi, result);
	}

if(result<0) {
	fprintf(stderr, "*** ERROR: internal inconsistency in power sum stats selftest\n");
	exit(-1);
	}

fprintf(stderr, "Power sum stats sefltest: passed\n");
fprintf(LOG, "Power sum stats sefltest: passed\n");

free_partial_power_sum_F(ps1);
free_partial_power_sum_F(ps2);
}

#if MANUAL_SSE
#define MODE(a)	(args_info.sse_arg ? (sse_ ## a) : (a) )
#else
#define MODE(a)	(a)
#endif

void init_power_sum_stats(void)
{
double eta;

if(!strcmp(args_info.statistics_function_arg , "linear")) {
	point_power_sum_stats=MODE(point_power_sum_stats_linear);
	fprintf(stderr, "point_power_sum_stats: linear\n");
	fprintf(LOG, "point_power_sum_stats: linear\n");
	} else
if(!strcmp(args_info.statistics_function_arg , "universal")) {
	point_power_sum_stats=MODE(point_power_sum_stats_universal_piecewise_linear);
	fprintf(stderr, "point_power_sum_stats: universal\n");
	fprintf(LOG, "point_power_sum_stats: universal\n");
	} else
if(!strcmp(args_info.statistics_function_arg , "sorted")) {
	point_power_sum_stats=point_power_sum_stats_sorted;
	fprintf(stderr, "point_power_sum_stats: sorted\n");
	fprintf(LOG, "point_power_sum_stats: sorted\n");
	} else {
	fprintf(stderr, "*** ERROR: Unknown statistics function requested\n");
	exit(-1);
	}

init_fc_ul();
init_fc_ll();
verify_limits();

/* Precompute x_epsilon for the universal upper limit function */
eta=0.04*(sqrt(log(args_info.nbins_arg*args_info.nbins_arg/(2*M_PI)))+gsl_cdf_gaussian_Pinv(args_info.confidence_level_arg, 1.0));

if(5.0/sqrt(args_info.nbins_arg)>eta)eta=5.0/sqrt(args_info.nbins_arg);

x_epsilon=gsl_cdf_gaussian_Pinv(args_info.confidence_level_arg, 1.0)+eta;


if(args_info.x_epsilon_given)x_epsilon=args_info.x_epsilon_arg;

fprintf(LOG, "confidence_level: %g\n", args_info.confidence_level_arg);
fprintf(LOG, "x_epsilon: %g\n", x_epsilon);

/* Account for power loss due to Hann windowing */
if(!strcasecmp("Hann", args_info.upper_limit_comp_arg)){
	if(!strcasecmp(args_info.averaging_mode_arg, "matched")) {
		/* Matched filter correctly reconstructs power in the bin */
		upper_limit_comp=1.0; 
		} else
	if(!strcasecmp(args_info.averaging_mode_arg, "single_bin_loose")) {
		/* 0.85 is a ratio between amplitude of 
		   half-bin centered signal and bin centered signal
		   *amplitude*

		   */
		/* Usual worst case single-bin correction for loss of power when not bin centered. */
		upper_limit_comp=1.0/0.85; 
		} else
	if(!strcasecmp(args_info.averaging_mode_arg, "matched_loose")) {
		/* Matched filter  correctly reconstructs power in the bin */
		upper_limit_comp=1.0; 
		} else
	if(!strcasecmp(args_info.averaging_mode_arg, "3") || !strcasecmp(args_info.averaging_mode_arg, "three")){
		/* 3 bins should contain the entire signal, regardless
		   of positioning */
		upper_limit_comp=sqrt(3.0);
		} else 
	if(!strcasecmp(args_info.averaging_mode_arg, "1") || !strcasecmp(args_info.averaging_mode_arg, "one")){
		/* 0.85 is a ratio between amplitude of 
		   half-bin centered signal and bin centered signal
		   *amplitude*

		   */
		upper_limit_comp=1.0/0.85;
		} else 
		{
		fprintf(stderr, "ERROR: do not know how to compensate upper limits for averaging mode \"%s\", try specifying upper_limit_comp option directly\n", args_info.averaging_mode_arg);
		}
	} else {
	upper_limit_comp=atof(args_info.upper_limit_comp_arg);
	}
fprintf(LOG, "upper limit compensation factor: %8f\n", upper_limit_comp);


// // /*	/* Extra factor to convert to amplitude from RMS power */
// // strain_comp=sqrt(2.0);*/
	/* New AM response correctly computes expected power from h0 */
strain_comp=1.0;
	/* Extra factor to convert to strain from raw SFT units */
strain_comp/=(args_info.sft_coherence_time_arg*16384.0);
	/* Extra factor to account for the fact that only half of SFT
	   coefficients is stored */
strain_comp*=sqrt(2.0);
	/* Revert strain normalization */
strain_comp*=args_info.strain_norm_factor_arg;

generate_alignment_grid();
}

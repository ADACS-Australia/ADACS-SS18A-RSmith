/*
*  Copyright (C) 2010, 2011 Evan Goetz
*
*  This program is free software; you can redistribute it and/or modify
*  it under the terms of the GNU General Public License as published by
*  the Free Software Foundation; either version 2 of the License, or
*  (at your option) any later version.
*
*  This program is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*  GNU General Public License for more details.
*
*  You should have received a copy of the GNU General Public License
*  along with with program; see the file COPYING. If not, write to the
*  Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston,
*  MA  02111-1307  USA
*/

#include <math.h>
#include <time.h>

#include <gsl/gsl_sf_trig.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_fit.h>
#include <gsl/gsl_sort.h>

#include <lal/LALConstants.h>
#include <lal/LALMalloc.h>
#include <lal/Window.h>

#include "templates.h"
#include "cdfwchisq.h"
#include "statistics.h"
#include "candidates.h"

//////////////////////////////////////////////////////////////
// Allocate memory for farStruct struct  -- done
farStruct * new_farStruct(void)
{
   
   const CHAR *fn = __func__;
   
   farStruct *farstruct = XLALMalloc(sizeof(*farstruct));
   if (farstruct==NULL) {
      fprintf(stderr,"%s: XLALMalloc(%zu) failed.\n", fn, sizeof(*farstruct));
      XLAL_ERROR_NULL(fn, XLAL_ENOMEM);
   }
   
   farstruct->far = 1.0;
   farstruct->topRvalues = NULL;

   return farstruct;

} /* new_farStruct() */


//////////////////////////////////////////////////////////////
// Destroy farStruct struct  -- done
void free_farStruct(farStruct *farstruct)
{
   
   XLALDestroyREAL4Vector(farstruct->topRvalues);
   farstruct->topRvalues = NULL;
   
   XLALFree((farStruct*)farstruct);

} /* free_farStruct() */


//////////////////////////////////////////////////////////////
// Estimate the FAR of the R statistic from the weights
void estimateFAR(farStruct *output, templateStruct *templatestruct, INT4 trials, REAL8 thresh, REAL4Vector *ffplanenoise, REAL4Vector *fbinaveratios)
{
   
   const CHAR *fn = __func__;
   
   INT4 ii, jj;
   INT4 numofweights = 0;
   for (ii=0; ii<(INT4)templatestruct->templatedata->length; ii++) if (templatestruct->templatedata->data[ii]!=0.0) numofweights++;
   
   REAL8 sumofsqweights = 0.0;
   for (ii=0; ii<numofweights; ii++) sumofsqweights += (templatestruct->templatedata->data[ii]*templatestruct->templatedata->data[ii]);
   REAL8 sumofsqweightsinv = 1.0/sumofsqweights;
   
   REAL4Vector *Rs = XLALCreateREAL4Vector((UINT4)trials);
   if (Rs==NULL) {
      fprintf(stderr,"%s: XLALCreateREAL4Vector(%d) failed.\n", fn, trials);
      XLAL_ERROR_VOID(fn, XLAL_EFUNC);
   }
   
   gsl_rng *rng = gsl_rng_alloc(gsl_rng_mt19937);
   if (rng==NULL) {
      fprintf(stderr,"%s: gsl_rng_alloc() failed.\n", fn);
      XLAL_ERROR_VOID(fn, XLAL_EFUNC);
   }
   //srand(time(NULL));
   //UINT8 randseed = rand();
   //gsl_rng_set(rng, randseed);
   gsl_rng_set(rng, 0);
   
   for (ii=0; ii<trials; ii++) {
      //Create noise value and R value
      REAL8 R = 0.0;
      for (jj=0; jj<numofweights; jj++) {
         REAL8 noise = expRandNum(ffplanenoise->data[ templatestruct->secondfftfrequencies->data[jj] ]*fbinaveratios->data[ templatestruct->firstfftfrequenciesofpixels->data[jj] ], rng);
         R += (noise - ffplanenoise->data[ templatestruct->secondfftfrequencies->data[jj] ]*fbinaveratios->data[ templatestruct->firstfftfrequenciesofpixels->data[jj] ])*templatestruct->templatedata->data[jj];
      }
      Rs->data[ii] = (REAL4)(R*sumofsqweightsinv);
   } /* for ii < trials */
   REAL4 mean = calcMean(Rs);
   REAL4 sigma = calcStddev(Rs);
   if (XLAL_IS_REAL4_FAIL_NAN(mean)) {
      fprintf(stderr,"%s: calcMean() failed.\n", fn);
      XLAL_ERROR_VOID(fn, XLAL_EFUNC);
   } else if (XLAL_IS_REAL4_FAIL_NAN(sigma)) {
      fprintf(stderr,"%s: calcStddev() failed.\n", fn);
      XLAL_ERROR_VOID(fn, XLAL_EFUNC);
   }
   
   //Do an insertion sort. At best this is O(thresh*trials), at worst this is O(thresh*trials*trials).
   if (output->topRvalues == NULL) {
      output->topRvalues = XLALCreateREAL4Vector((UINT4)roundf(thresh*trials)+1);
      if (output->topRvalues==NULL) {
         fprintf(stderr,"%s: XLALCreateREAL4Vector(%d) failed.\n", fn, (INT4)roundf(thresh*trials)+1);
         XLAL_ERROR_VOID(fn, XLAL_EFUNC);
      }
   }
   if ((gsl_sort_float_largest((float*)output->topRvalues->data, output->topRvalues->length, (float*)Rs->data, 1, Rs->length)) != 0) {
      fprintf(stderr,"%s: gsl_sort_float_largest() failed.\n", fn);
      XLAL_ERROR_VOID(fn, XLAL_EFUNC);
   }
   
   output->far = output->topRvalues->data[output->topRvalues->length - 1];
   output->distMean = mean;
   output->distSigma = sigma;
   
   //Destroy
   XLALDestroyREAL4Vector(Rs);
   gsl_rng_free(rng);

} /* estimateFAR() */


//////////////////////////////////////////////////////////////
// Numerically solve for the FAR of the R statistic from the weights
void numericFAR(farStruct *output, templateStruct *templatestruct, REAL8 thresh, REAL4Vector *ffplanenoise, REAL4Vector *fbinaveratios, INT4 method)
{
   
   const CHAR *fn = __func__;
   
   INT4 ii;
   
   INT4 errcode = 0;
   
   //Set up solver: method 0 is Brent's method, method 1 is Newton's method
   const gsl_root_fsolver_type *T1 = gsl_root_fsolver_brent;
   gsl_root_fsolver *s1 = gsl_root_fsolver_alloc(T1);
   if (s1==NULL) {
      fprintf(stderr,"%s: gsl_root_fsolver_alloc() failed.\n", fn);
      XLAL_ERROR_VOID(fn, XLAL_EFUNC);
   }
   gsl_function F;
   const gsl_root_fdfsolver_type *T0 = gsl_root_fdfsolver_newton;
   //const gsl_root_fdfsolver_type *T0 = gsl_root_fdfsolver_steffenson;
   gsl_root_fdfsolver *s0 = gsl_root_fdfsolver_alloc(T0);
   if (s0==NULL) {
      fprintf(stderr,"%s: gsl_root_fdfsolver_alloc() failed.\n", fn);
      XLAL_ERROR_VOID(fn, XLAL_EFUNC);
   }
   gsl_function_fdf FDF;
   
   
   //Include the various parameters in the struct required by GSL
   struct gsl_probR_pars params = {templatestruct, ffplanenoise, fbinaveratios, thresh, errcode};
   
   //Assign GSL function the necessary parts
   if (method != 0) {
      F.function = &gsl_probR;
      F.params = &params;
   } else {
      FDF.f = &gsl_probR;
      FDF.df = &gsl_dprobRdR;
      FDF.fdf = &gsl_probRandDprobRdR;
      FDF.params = &params;
   }
   
   //Start off with an initial guess and set the solver at the beginning
   REAL8 Rlow = 0.0, Rhigh = 10000.0, root = 400.0;
   if (method != 0) {
      if ( (gsl_root_fsolver_set(s1, &F, Rlow, Rhigh)) != 0 ) {
         fprintf(stderr,"%s: Unable to initialize root solver to bracketed positions.\n", fn);
         XLAL_ERROR_VOID(fn, XLAL_EFUNC);
      }
   } else {
      if ( (gsl_root_fdfsolver_set(s0, &FDF, root)) != 0 ) {
         fprintf(stderr,"%s: Unable to initialize root solver to first guess.\n", fn);
         XLAL_ERROR_VOID(fn, XLAL_EFUNC);
      } 
   }
   
   
   gsl_rng *rng = gsl_rng_alloc(gsl_rng_mt19937);
   if (rng==NULL) {
      fprintf(stderr,"%s: gsl_rng_alloc() failed.\n", fn);
      XLAL_ERROR_VOID(fn, XLAL_ENOMEM);
   }
   srand(time(NULL));
   UINT8 randseed = rand();
   gsl_rng_set(rng, randseed);
   
   
   //And now find the root
   ii = 0;
   INT4 max_iter = 100, jj = 0, max_retries = 10;
   INT4 status = GSL_CONTINUE;
   REAL8 prevroot = 0.0;
   while (status==GSL_CONTINUE && ii<max_iter) {
      
      ii++;
      
      if (method != 0) {
         status = gsl_root_fsolver_iterate(s1);
         if (status!=GSL_CONTINUE && status!=GSL_SUCCESS) {
            fprintf(stderr,"%s: gsl_root_fsolver_iterate() failed with code %d.\n", fn, status);
            XLAL_ERROR_VOID(fn, XLAL_EFUNC);
         }
         if (ii>0) prevroot = root;
         root = gsl_root_fsolver_root(s1);
         Rlow = gsl_root_fsolver_x_lower(s1);
         Rhigh = gsl_root_fsolver_x_upper(s1);
         status = gsl_root_test_interval(Rlow, Rhigh, 0.0, 0.001);
         if (status!=GSL_CONTINUE && status!=GSL_SUCCESS) {
            fprintf(stderr,"%s: gsl_root_test_interval() failed with code %d.\n", fn, status);
            XLAL_ERROR_VOID(fn, XLAL_EFUNC);
         }
      } else {
         status = gsl_root_fdfsolver_iterate(s0);
         if (status!=GSL_CONTINUE && status!=GSL_SUCCESS) {
            fprintf(stderr,"%s: gsl_root_fdfsolver_iterate() failed with code %d.\n", fn, status);
            XLAL_ERROR_VOID(fn, XLAL_EFUNC);
         }
         prevroot = root;
         root = gsl_root_fdfsolver_root(s0);
         status = gsl_root_test_delta(prevroot, root, 0.0, 0.001);
         if (status!=GSL_CONTINUE && status!=GSL_SUCCESS) {
            fprintf(stderr,"%s: gsl_root_test_delta() failed with code %d.\n", fn, status);
            XLAL_ERROR_VOID(fn, XLAL_EFUNC);
         }
         
         //If there is an issue that the root is negative, try a new initial guess
         if (root<0.0 && jj<max_retries) {
            ii = 0;
            jj++;
            status = GSL_CONTINUE;
            if ( (gsl_root_fdfsolver_set(s0, &FDF, gsl_rng_uniform_pos(rng)*Rhigh)) != 0 ) {
               fprintf(stderr,"%s: Unable to initialize root solver to first guess.\n", fn);
               XLAL_ERROR_VOID(fn, XLAL_EFUNC);
            }
         } else if (root<0.0 && jj==max_retries) {
            status = GSL_FAILURE;
         } //Up to here
         
      }
      
   } /* while status==GSL_CONTINUE && ii < max_iter */
   
   if (method != 0) {
      if (status != GSL_SUCCESS) {
         fprintf(stderr,"%s: Root finding iteration (%d/%d) failed with failure code %d. Previous root = %f, current root = %f\n", fn, ii, max_iter, status, prevroot, root);
         XLAL_ERROR_VOID(fn, XLAL_FAILURE);
      } else if (ii==max_iter) {
         fprintf(stderr,"%s: Root finding iteration (%d/%d) failed with failure code %d. Previous root = %f, current root = %f\n", fn, ii, max_iter, status, prevroot, root);
         XLAL_ERROR_VOID(fn, XLAL_EMAXITER);
      } else if (root == 0.0) {
         fprintf(stderr,"%s: Root finding iteration (%d/%d) converged to 0.0.\n", fn, ii, max_iter);
         XLAL_ERROR_VOID(fn, XLAL_ERANGE);
      } else if (root == 1000.0) {
         fprintf(stderr,"%s: Root finding iteration (%d/%d) converged to 1000.0.\n", fn, ii, max_iter);
         XLAL_ERROR_VOID(fn, XLAL_ERANGE);
      }
   } else {
      if (status != GSL_SUCCESS) {
         fprintf(stderr,"%s: Root finding iteration (%d/%d) failed with failure code %d. Previous root = %f, current root = %f\n", fn, ii, max_iter, status, prevroot, root);
         XLAL_ERROR_VOID(fn, XLAL_FAILURE);
      } else if (ii==max_iter) {
         fprintf(stderr,"%s: Root finding iteration (%d/%d) failed with failure code %d. Previous root = %f, current root = %f\n", fn, ii, max_iter, status, prevroot, root);
         XLAL_ERROR_VOID(fn, XLAL_EMAXITER);
      } else if (root<=0.0) {
         fprintf(stderr,"%s: Threshold value found (%f) is less than 0.0!\n", fn, root);
         XLAL_ERROR_VOID(fn, XLAL_ERANGE);
      }
   }
   
   
   output->far = root;
   output->distMean = 0.0;
   output->distSigma = 0.0; //Fake the value of sigma
   output->farerrcode = errcode;
   
   //Cleanup
   gsl_root_fsolver_free(s1);
   gsl_root_fdfsolver_free(s0);
   gsl_rng_free(rng);
   
   
} /* numericFAR() */
REAL8 gsl_probR(REAL8 R, void *param)
{
   
   struct gsl_probR_pars *pars = (struct gsl_probR_pars*)param;
   
   REAL8 dR = 0.005;
   REAL8 R1 = (1.0+dR)*R;
   REAL8 R2 = (1.0-dR)*R;
   INT4 errcode1 = 0, errcode2 = 0, errcode3 = 0;
   
   REAL8 prob = (probR(pars->templatestruct, pars->ffplanenoise, pars->fbinaveratios, R, &errcode1) + probR(pars->templatestruct, pars->ffplanenoise, pars->fbinaveratios, R1, &errcode2) + probR(pars->templatestruct, pars->ffplanenoise, pars->fbinaveratios, R2, &errcode3))/3.0;
   
   if (errcode1!=0) {
      pars->errcode = errcode1;
   } else if (errcode2!=0) {
      pars->errcode = errcode2;
   } else if (errcode3!=0) {
      pars->errcode = errcode3;
   }
   
   REAL8 returnval = prob - log10(pars->threshold);
   
   return returnval;
   
} /* gsl_probR() */
REAL8 gsl_dprobRdR(REAL8 R, void *param)
{
   
   struct gsl_probR_pars *pars = (struct gsl_probR_pars*)param;
   
   REAL8 dR = 0.005;
   
   INT4 errcode1 = 0, errcode2 = 0;
   
   //Explicit computation of slope
   REAL8 R1 = (1.0+dR)*R;
   REAL8 R2 = (1.0-dR)*R;
   //REAL8 prob1 = probR(pars->templatestruct, pars->ffplanenoise, pars->fbinaveratios, R1, &errcode1);
   //REAL8 prob2 = probR(pars->templatestruct, pars->ffplanenoise, pars->fbinaveratios, R2, &errcode2);
   REAL8 prob1 = gsl_probR(R1, pars);
   REAL8 prob2 = gsl_probR(R2, pars);
   while (fabs(prob1-prob2)<100.0*LAL_REAL8_EPS) {
      dR *= 2.0;
      R1 = (1.0+dR)*R;
      R2 = (1.0-dR)*R;
      //prob1 = probR(pars->templatestruct, pars->ffplanenoise, pars->fbinaveratios, R1, &errcode1);
      //prob2 = probR(pars->templatestruct, pars->ffplanenoise, pars->fbinaveratios, R2, &errcode2);
      prob1 = gsl_probR(R1, pars);
      prob2 = gsl_probR(R2, pars);
   }
   REAL8 diffR = R1 - R2;
   REAL8 slope = (prob1-prob2)/diffR;
   //fprintf(stderr,"GSL derivative = %g\n", slope);
   
   //Added for improved resolution:
   /* REAL8 R3 = R-2.0*diffR;
   REAL8 prob3 = probR(pars->templatestruct, pars->ffplanenoise, pars->fbinaveratios, R3, &pars->errcode);
   REAL8 R4 = R+2.0*diffR;
   REAL8 prob4 = probR(pars->templatestruct, pars->ffplanenoise, pars->fbinaveratios, R4, &pars->errcode);
   
   slope = (8.0*(prob1-prob2)+prob3-prob4)/(12.0*diffR); */
   
   if (errcode1!=0) {
      pars->errcode = errcode1;
   } else if (errcode2!=0) {
      pars->errcode = errcode2;
   }
   
   return slope;
   
} /* gsl_dprobRdR() */
void gsl_probRandDprobRdR(REAL8 R, void *param, REAL8 *probabilityR, REAL8 *dprobRdR)
{
   
   struct gsl_probR_pars *pars = (struct gsl_probR_pars*)param;
   
   *probabilityR = gsl_probR(R, pars);
   
   *dprobRdR = gsl_dprobRdR(R, pars);
   
} /* gsl_probRandDprobRdR() */


//////////////////////////////////////////////////////////////
// Analytically calculate the probability of a true signal output is log10(prob)
REAL8 probR(templateStruct *templatestruct, REAL4Vector *ffplanenoise, REAL4Vector *fbinaveratios, REAL8 R, INT4 *errcode)
{
   
   const CHAR *fn = __func__;
   
   INT4 ii = 0;
   REAL8 prob = 0.0;
   REAL8 sumwsq = 0.0;
   INT4 numweights = 0;
   for (ii=0; ii<(INT4)templatestruct->templatedata->length; ii++) {
      if (templatestruct->templatedata->data[ii]!=0.0) {
         numweights++;
         sumwsq += templatestruct->templatedata->data[ii]*templatestruct->templatedata->data[ii];
      }
   }
   
   REAL8Vector *newweights = XLALCreateREAL8Vector((UINT4)numweights);
   INT4Vector *sorting = XLALCreateINT4Vector((UINT4)numweights);
   if (newweights==NULL) {
      fprintf(stderr,"%s: XLALCreateREAL8Vector(%d) failed.\n", fn, numweights);
      XLAL_ERROR_REAL8(fn, XLAL_EFUNC);
   } else if (sorting==NULL) {
      fprintf(stderr,"%s: XLALCreateINT4Vector(%d) failed.\n", fn, numweights);
      XLAL_ERROR_REAL8(fn, XLAL_EFUNC);
   }
   
   REAL8 Rpr = R;
   for (ii=0; ii<(INT4)newweights->length; ii++) {
      newweights->data[ii] = 0.5*templatestruct->templatedata->data[ii]*ffplanenoise->data[ templatestruct->secondfftfrequencies->data[ii] ]*fbinaveratios->data[ templatestruct->firstfftfrequenciesofpixels->data[ii] ]/sumwsq;
      Rpr += templatestruct->templatedata->data[ii]*ffplanenoise->data[ templatestruct->secondfftfrequencies->data[ii] ]*fbinaveratios->data[ templatestruct->firstfftfrequenciesofpixels->data[ii] ]/sumwsq;
      sorting->data[ii] = ii;  //This is for the fact that a few steps later (before using Davies' algorithm, we sort the weights)
   }
   
   qfvars vars;
   vars.weights = newweights;
   vars.sorting = sorting;
   vars.dofs = NULL;
   vars.noncentrality = NULL;
   vars.ndtsrt = 0;           //Set because we do the sorting outside of Davies' algorithm with qsort
   vars.lim = 1000000;
   vars.c = Rpr;
   REAL8 sigma = 0.0;
   REAL8 accuracy = 1.0e-13;   //(1e-5) old value
   
   //sort the weights here so we don't have to do it later (qsort)
   sort_double_ascend(newweights);
   
   //cdfwchisq(algorithm variables, sigma, accuracy, error code)
   prob = 1.0 - cdfwchisq_twospect(&vars, sigma, accuracy, errcode);
   
   //Large R values can cause a problem when computing the probability. We run out of accuracy quickly even using double precision
   //Potential fix: compute log10(prob) for smaller values of R, for when slope is linear between log10 probabilities
   //Use slope to extend the computation and then compute the exponential of the found log10 probability.
   REAL8 logprobest = 0.0;
   INT4 estimatedTheProb = 0;
   if (prob<=1.0e-9) {
      estimatedTheProb = 1;
      
      INT4 errcode1 = 0, errcode2 = 0;
      REAL8 probslope=0.0, tempprob, tempprob2, c1, c2, c = 0.0, logprobave = 0.0;
      
      gsl_rng *rng = gsl_rng_alloc(gsl_rng_mt19937);
      if (rng==NULL) {
         fprintf(stderr,"%s: gsl_rng_alloc() failed.\n", fn);
         XLAL_ERROR_REAL8(fn, XLAL_EFUNC);
      }
      gsl_rng_set(rng, 0);
      
      REAL8Vector *slopes = XLALCreateREAL8Vector(50);
      if (slopes==NULL) {
         fprintf(stderr,"%s: XLALCreateREAL8Vector(%d) failed.\n", fn, 50);
         XLAL_ERROR_REAL8(fn, XLAL_EFUNC);
      }
      
      REAL8 lowerend = 0.0;
      REAL8 upperend = Rpr;
      
      for (ii=0; ii<(INT4)slopes->length; ii++) {
         c1 = gsl_rng_uniform_pos(rng)*(upperend-lowerend)+lowerend;
         vars.c = c1;
         tempprob = 1.0-cdfwchisq_twospect(&vars, sigma, accuracy, &errcode1);
         while (tempprob<=1.0e-11 || tempprob>=1.0e-9) {
            if (tempprob<=1.0e-11) upperend = c1;
            else if (tempprob>=1.0e-9) lowerend = c1;
            c1 = gsl_rng_uniform_pos(rng)*(upperend-lowerend)+lowerend;
            vars.c = c1;
            tempprob = 1.0-cdfwchisq_twospect(&vars, sigma, accuracy, &errcode1);
         }
         logprobave += log10(tempprob);
         c += c1;
         c2 = gsl_rng_uniform_pos(rng)*(upperend-lowerend)+lowerend;
         vars.c = c2;
         tempprob2 = 1.0 - cdfwchisq_twospect(&vars, sigma, accuracy, &errcode2);
         while (tempprob2<=1.0e-11 || tempprob2>=1.0e-9 || fabs(c1-c2)<=100.0*LAL_REAL8_EPS) {
            if (tempprob2<=1.0e-11) upperend = c2;
            else if (tempprob2>=1.0e-9) lowerend = c2;
            c2 = gsl_rng_uniform_pos(rng)*(upperend-lowerend)+lowerend;
            vars.c = c2;
            tempprob2 = 1.0-cdfwchisq_twospect(&vars, sigma, accuracy, &errcode2);
         }
         logprobave += log10(tempprob2);
         c += c2;
         
         //If either point along the slope had a problem at the end, then better fail.
         //Otherwise, set errcode = 0;
         if (errcode1!=0 || errcode2!=0) {
            fprintf(stderr,"%s: cdfwchisq() failed.\n", fn);
            XLAL_ERROR_REAL8(fn, XLAL_EFUNC);
         } else {
            *errcode = errcode1;
         }
         
         slopes->data[ii] = (log10(tempprob)-log10(tempprob2))/(c1-c2);
         if (slopes->data[ii]>=0.0) {
            fprintf(stderr, "%s: Slope calculation failed. Non-negative slope: %f", fn, probslope);
            XLAL_ERROR_REAL8(fn, XLAL_EDIVERGE);
         }
      }
      REAL8 cave = .5*c/(REAL8)slopes->length;
      logprobave /= 2.0*slopes->length;
      probslope = calcMeanD(slopes);
      logprobest = probslope*(Rpr-cave) + logprobave;
      if (logprobest>-0.5) {
         fprintf(stderr, "%s: Failure calculating accurate interpolated value.\n", fn);
         XLAL_ERROR_REAL8(fn, XLAL_ERANGE);
      }
      XLALDestroyREAL8Vector(slopes);
      
      lowerend = 0.0;
      upperend = Rpr;
      REAL8Vector *probvals = XLALCreateREAL8Vector(10);
      REAL8Vector *cvals = XLALCreateREAL8Vector(probvals->length);
      if (probvals==NULL) {
         fprintf(stderr,"%s: XLALCreateREAL8Vector(%d) failed.\n", fn, 10);
         XLAL_ERROR_REAL8(fn, XLAL_EFUNC);
      } else if (cvals==NULL) {
         fprintf(stderr,"%s: XLALCreateREAL8Vector(%d) failed.\n", fn, 10);
         XLAL_ERROR_REAL8(fn, XLAL_EFUNC);
      }
      for (ii=0; ii<(INT4)probvals->length; ii++) {
         c1 = gsl_rng_uniform_pos(rng)*(upperend-lowerend)+lowerend;
         vars.c = c1;
         tempprob = 1.0-cdfwchisq_twospect(&vars, sigma, accuracy, &errcode1);
         while (tempprob<=1.0e-11 || tempprob>=1.0e-9) {
            if (tempprob<=1.0e-11) upperend = c1;
            else if (tempprob>=1.0e-9) lowerend = c1;
            c1 = gsl_rng_uniform_pos(rng)*(upperend-lowerend)+lowerend;
            vars.c = c1;
            tempprob = 1.0-cdfwchisq_twospect(&vars, sigma, accuracy, &errcode1);
         }
         probvals->data[ii] = log10(tempprob);
         cvals->data[ii] = c1;
      }
      REAL8 yintercept, cov00, cov01, cov11, sumsq;
      if (gsl_fit_linear(cvals->data, 1, probvals->data, 1, cvals->length, &yintercept, &probslope, &cov00, &cov01, &cov11, &sumsq)!=GSL_SUCCESS) {
         fprintf(stderr,"%s: gsl_fit_linear() failed.\n", fn);
         XLAL_ERROR_REAL8(fn, XLAL_EFUNC);
      }
      logprobest = probslope*Rpr + yintercept;
      XLALDestroyREAL8Vector(probvals);
      XLALDestroyREAL8Vector(cvals);
      
      gsl_rng_free(rng);
      
   }
   
   //If errcode is still != 0, better fail
   if (*errcode!=0) {
      fprintf(stderr,"%s: cdfwchisq() failed.\n", fn);
      XLAL_ERROR_REAL8(fn, XLAL_EFUNC);
   }
   
   //Cleanup
   XLALDestroyREAL8Vector(newweights);
   XLALDestroyINT4Vector(sorting);
   
   if (estimatedTheProb==1) {
      return logprobest;
   } else {
      return log10(prob);
   }
   
} /* probR() */


templateStruct * new_templateStruct(INT4 length)
{
   
   const CHAR *fn = __func__;
   
   INT4 ii;
   
   templateStruct *templatestruct = XLALMalloc(sizeof(*templatestruct));
   if (templatestruct==NULL) {
      fprintf(stderr,"%s: XLALMalloc(%zu) failed.\n", fn, sizeof(*templatestruct));
      XLAL_ERROR_NULL(fn, XLAL_ENOMEM);
   }
   
   templatestruct->templatedata = XLALCreateREAL4Vector((UINT4)length);
   templatestruct->pixellocations = XLALCreateINT4Vector((UINT4)length);
   templatestruct->firstfftfrequenciesofpixels = XLALCreateINT4Vector((UINT4)length);
   templatestruct->secondfftfrequencies = XLALCreateINT4Vector((UINT4)length);
   if (templatestruct->templatedata==NULL) {
      fprintf(stderr,"%s: XLALCreateREAL4Vector(%d) failed.\n", fn, length);
      XLAL_ERROR_NULL(fn, XLAL_EFUNC);
   } else if (templatestruct->pixellocations==NULL) {
      fprintf(stderr,"%s: XLALCreateINT4Vector(%d) failed.\n", fn, length);
      XLAL_ERROR_NULL(fn, XLAL_EFUNC);
   } else if (templatestruct->firstfftfrequenciesofpixels==NULL) {
      fprintf(stderr,"%s: XLALCreateINT4Vector(%d) failed.\n", fn, length);
      XLAL_ERROR_NULL(fn, XLAL_EFUNC);
   } else if (templatestruct->secondfftfrequencies==NULL) {
      fprintf(stderr,"%s: XLALCreateINT4Vector(%d) failed.\n", fn, length);
      XLAL_ERROR_NULL(fn, XLAL_EFUNC);
   } 
   
   for (ii=0; ii<length; ii++) {
      templatestruct->templatedata->data[ii] = 0.0;
      templatestruct->pixellocations->data[ii] = 0;
      templatestruct->firstfftfrequenciesofpixels->data[ii] = 0;
      templatestruct->secondfftfrequencies->data[ii] = 0;
   }
   
   templatestruct->f0 = 0.0;
   templatestruct->period = 0.0;
   templatestruct->moddepth = 0.0;
   
   return templatestruct;
   
} /* new_templateStruct() */


void free_templateStruct(templateStruct *nameoftemplate)
{
   
   XLALDestroyREAL4Vector(nameoftemplate->templatedata);
   XLALDestroyINT4Vector(nameoftemplate->pixellocations);
   XLALDestroyINT4Vector(nameoftemplate->firstfftfrequenciesofpixels);
   XLALDestroyINT4Vector(nameoftemplate->secondfftfrequencies);
   
   XLALFree((templateStruct*)nameoftemplate);
   
} /* free_templateStruct() */


//////////////////////////////////////////////////////////////
// Make an estimated template based on FFT of train of Gaussians
//void makeTemplateGaussians(ffdataStruct *out, candidate *in)
void makeTemplateGaussians(templateStruct *output, candidate input, inputParamsStruct *params)
{
   
   const CHAR *fn = __func__;
   
   //Set data for output template
   output->f0 = input.fsig;
   output->period = input.period;
   output->moddepth = input.moddepth;
   
   INT4 ii, jj, numfbins, numffts, N;
   
   //Reset the data values to zero, just in case
   for (ii=0; ii<(INT4)output->templatedata->length; ii++) output->templatedata->data[ii] = 0.0;
   
   numfbins = (INT4)(round(params->fspan*params->Tcoh)+1);   //Number of frequency bins
   numffts = (INT4)floor(params->Tobs/(params->Tcoh-params->SFToverlap)-1);   //Number of FFTs
   N = (INT4)floor(params->Tobs/input.period);     //Number of Gaussians
   
   REAL8 periodf = 1.0/input.period;
   
   //Determine separation in time of peaks for each frequency
   REAL8Vector *phi_actual = XLALCreateREAL8Vector((UINT4)numfbins);
   if (phi_actual==NULL) {
      fprintf(stderr,"%s: XLALCreateREAL8Vector(%d) failed.\n", fn, numfbins);
      XLAL_ERROR_VOID(fn, XLAL_EFUNC);
   }
   for (ii=0; ii<(INT4)phi_actual->length; ii++) {
      if ( fabs(params->fmin + ii/params->Tcoh - input.fsig)/input.moddepth <= 1.0 ) {
         phi_actual->data[ii] = 0.5*input.period - asin(fabs(params->fmin + ii/params->Tcoh - input.fsig)/
            input.moddepth)*LAL_1_PI*input.period;
      } else {
         phi_actual->data[ii] = 0.0;
      }
   } /* for ii < phi_actual->length */
   
   //Create second FFT frequencies
   REAL8Vector *fpr = XLALCreateREAL8Vector((UINT4)floor(numffts*0.5)+1);
   if (fpr==NULL) {
      fprintf(stderr,"%s: XLALCreateREAL8Vector(%d) failed.\n", fn, (INT4)floor(numffts*0.5)+1);
      XLAL_ERROR_VOID(fn, XLAL_EFUNC);
   }
   for (ii=0; ii<(INT4)fpr->length; ii++) fpr->data[ii] = (REAL8)ii*(1.0/params->Tobs);
   
   //Scale used for "spillover" into bins outside of phi_actual
   REAL8 k = input.moddepth*params->Tcoh;    //amplitude of modulation in units of bins
   REAL8Vector *scale = XLALCreateREAL8Vector((UINT4)numfbins);      //the scaling factor
   if (scale==NULL) {
      fprintf(stderr,"%s: XLALCreateREAL8Vector(%d) failed.\n", fn, numfbins);
      XLAL_ERROR_VOID(fn, XLAL_EFUNC);
   }
   INT4 m0 = (INT4)round(input.fsig*params->Tcoh) - (INT4)round(params->fmin*params->Tcoh);   //central frequency bin
   INT4 mextent = (INT4)floor(input.moddepth*params->Tcoh);   //Bins filled by modulation
   REAL8 overage = (k-(REAL8)mextent)-1.0;
   //REAL8 overage = (k-(REAL8)mextent);
   INT4 fnumstart = -1;
   INT4 fnumend = -1;
   for (ii=0; ii<numfbins; ii++) {
      if (ii < m0-mextent-2 || ii > m0+mextent+2) {
         scale->data[ii] = 0.0;
      } else if (ii == m0-mextent-2 || ii == m0+mextent+2) {
         //scale->data[ii] = sincxoverxsqminusone(overage-1.0)*sincxoverxsqminusone(overage-1.0);
         
         //scale->data[ii] = sincxoverxsqminusone(overage-1.0);
         //scale->data[ii] *= scale->data[ii];
         
         scale->data[ii] = sqsincxoverxsqminusone(overage-1.0);
      } else if (ii == m0-mextent-1 || ii == m0+mextent+1) {
         //scale->data[ii] = sincxoverxsqminusone(overage)*sincxoverxsqminusone(overage);
         
         //scale->data[ii] = sincxoverxsqminusone(overage);
         //scale->data[ii] *= scale->data[ii];
         
         scale->data[ii] = sqsincxoverxsqminusone(overage);
      } else {
         scale->data[ii] = 1.0;
      }
   } /* for ii < scale->length */
   for (ii=0; ii<(INT4)scale->length; ii++) {
      if (scale->data[ii] != 0.0 && fnumstart == -1) fnumstart = ii;
      if (scale->data[ii] == 0.0 && fnumstart != -1 && fnumend==-1) fnumend = ii-1;
   }
   
   //Exit with failure if there is a problem
   if (fnumend==-1) {
      fprintf(stderr, "%s: Failed because fnumend was -1.\n", fn);
      XLAL_ERROR_VOID(fn, XLAL_EFAILED);
   }
   
   //Make sigmas for each frequency
   REAL8Vector *sigmas = XLALCreateREAL8Vector((UINT4)(fnumend-fnumstart+1));
   REAL8Vector *wvals = XLALCreateREAL8Vector((UINT4)floor(2.0*input.period/params->Tcoh));
   REAL8Vector *allsigmas = XLALCreateREAL8Vector(wvals->length * sigmas->length);
   if (sigmas==NULL) {
      fprintf(stderr,"%s: XLALCreateREAL8Vector(%d) failed.\n", fn, (UINT4)(fnumend-fnumstart+1));
      XLAL_ERROR_VOID(fn, XLAL_EFUNC);
   } else if (wvals==NULL) {
      fprintf(stderr,"%s: XLALCreateREAL8Vector(%d) failed.\n", fn, (UINT4)floor(2.0*input.period/params->Tcoh));
      XLAL_ERROR_VOID(fn, XLAL_EFUNC);
   } else if (allsigmas==NULL) {
      fprintf(stderr,"%s: XLALCreateREAL8Vector(%d) failed.\n", fn, wvals->length * sigmas->length);
      XLAL_ERROR_VOID(fn, XLAL_EFUNC);
   }
   for (ii=0; ii<(INT4)wvals->length; ii++) {         //t = (ii+1)*in->Tcoh*0.5
      REAL8 sigbin = (input.moddepth*cos(LAL_TWOPI*periodf*((ii+1)*params->Tcoh*0.5))+input.fsig)*params->Tcoh;
      REAL8 sigbinvelocity = fabs(-input.moddepth*sin(LAL_TWOPI*periodf*((ii+1)*params->Tcoh*0.5))*params->Tcoh*0.5*params->Tcoh*LAL_TWOPI*periodf);
      REAL8 sigma = 0.5 * params->Tcoh * ((383.85*LAL_1_PI)*(0.5*6.1e-3) / ((sigbinvelocity+0.1769)*(sigbinvelocity+0.1769)+(0.5*6.1e-3)*(0.5*6.1e-3)) + 0.3736);   //Derived fit from simulation
      for (jj=0; jj<(INT4)sigmas->length; jj++) {
         //allsigmas->data[ii*sigmas->length + jj] = sincxoverxsqminusone(sigbin-round(params->fmin*params->Tcoh+jj+fnumstart))*sincxoverxsqminusone(sigbin-round(params->fmin*params->Tcoh+jj+fnumstart))*sigma;
         
         //allsigmas->data[ii*sigmas->length + jj] = sincxoverxsqminusone(sigbin-round(params->fmin*params->Tcoh+jj+fnumstart));
         //allsigmas->data[ii*sigmas->length + jj] *= allsigmas->data[ii*sigmas->length + jj];
         //allsigmas->data[ii*sigmas->length + jj] *= sigma;
         
         allsigmas->data[ii*sigmas->length + jj] = sqsincxoverxsqminusone(sigbin-round(params->fmin*params->Tcoh+jj+fnumstart))*sigma;
      }
   } /* for ii < wvals->length */
   for (ii=0; ii<(INT4)sigmas->length; ii++) {
      for (jj=0; jj<(INT4)wvals->length; jj++) wvals->data[jj] = allsigmas->data[ii + jj*sigmas->length]*allsigmas->data[ii + jj*sigmas->length];
      sigmas->data[ii] = sqrt(calcMeanD(wvals));
      //for (jj=0; jj<(INT4)wvals->length; jj++) wvals->data[jj] = allsigmas->data[ii + jj*sigmas->length];
      //sigmas->data[ii] = (calcMeanD(wvals));
   }
   
   
   //Create template
   REAL8 sum = 0.0, dataval = 0.0;
   REAL8 sin2pix = 0.0, cos2pix = 0.0, sinx = 0.0, cosx = 0.0;
   for (ii=0; ii<(INT4)sigmas->length; ii++) {
      REAL8 s = sigmas->data[ii];
      REAL8 scale1 = 1.0/(1.0+exp(-phi_actual->data[ii+fnumstart]*phi_actual->data[ii+fnumstart]*0.5/(s*s)));
      for (jj=0; jj<(INT4)fpr->length; jj++) {
         
         REAL8 omega = LAL_TWOPI*fpr->data[jj];
         twospect_sin_cos_LUT(&sinx, &cosx, input.period*omega);
         if (jj==0 || jj==1 || jj==2 || jj==3) {
            dataval = 0.0;
         } else if (fabs(cosx-1.0)<1e-5) {
         //} else if (fabs(cos(input.period*omega)-1.0)<1e-5) {
            //dataval = scale->data[ii+fnumstart] * scale1 * 2.0 * LAL_TWOPI * s * s * exp(-s * s * omega * omega) * (cos(phi_actual->data[ii+fnumstart] * omega) + 1.0) * N * N;
            twospect_sin_cos_2PI_LUT(&sin2pix, &cos2pix, phi_actual->data[ii+fnumstart]*fpr->data[jj]);
            dataval = scale->data[ii+fnumstart] * scale1 * 2.0 * LAL_TWOPI * s * s * exp(-s * s * omega * omega) * (cos2pix + 1.0) * N * N;
         } else {
            //dataval = scale->data[ii+fnumstart] * scale1 * 2.0 * LAL_TWOPI * s * s * exp(-s * s * omega * omega) * (cos(N * input.period * omega) - 1.0) * (cos(phi_actual->data[ii+fnumstart] * omega) + 1.0) / (cos(input.period * omega) - 1.0);
            twospect_sin_cos_2PI_LUT(&sin2pix, &cos2pix, N*input.period*fpr->data[jj]);
            dataval = scale->data[ii+fnumstart] * scale1 * 2.0 * LAL_TWOPI * s * s * exp(-s * s * omega * omega) * (cos2pix - 1.0);
            twospect_sin_cos_2PI_LUT(&sin2pix, &cos2pix, phi_actual->data[ii+fnumstart]*fpr->data[jj]);
            dataval *= (cos2pix + 1.0);
            twospect_sin_cos_2PI_LUT(&sin2pix, &cos2pix, input.period*fpr->data[jj]);
            dataval /= (cos2pix - 1.0);
         }
         
         //Sum up the weights in total
         sum += dataval;
         
         //Compare with weakest top bins and if larger, launch a search to find insertion spot (insertion sort)
         if (jj>1 && dataval > output->templatedata->data[output->templatedata->length-1]) {
            insertionSort_template(output, (REAL4)dataval, (ii+fnumstart)*fpr->length+jj, ii+fnumstart, jj);
         }
      } /* for jj < fpr->legnth */
   } /* for ii < sigmas->length */
   
   //Normalize
   for (ii=0; ii<(INT4)output->templatedata->length; ii++) if (output->templatedata->data[ii]!=0.0) output->templatedata->data[ii] /= (REAL4)sum;
   
   //Destroy variables
   XLALDestroyREAL8Vector(phi_actual);
   XLALDestroyREAL8Vector(scale);
   XLALDestroyREAL8Vector(sigmas);
   XLALDestroyREAL8Vector(allsigmas);
   XLALDestroyREAL8Vector(wvals);
   XLALDestroyREAL8Vector(fpr);

} /* mateTemplateGaussians() */


//////////////////////////////////////////////////////////////
// Make an template based on FFT of sinc squared functions  -- done
void makeTemplate(templateStruct *output, candidate input, inputParamsStruct *params, INT4Vector *sftexist, REAL4FFTPlan *plan)
{
   
   const CHAR *fn = __func__;
   
   //Set data for output template
   output->f0 = input.fsig;
   output->period = input.period;
   output->moddepth = input.moddepth;
   
   INT4 ii, jj, numfbins, numffts;
   
   for (ii=0; ii<(INT4)output->templatedata->length; ii++) output->templatedata->data[ii] = 0.0;
   
   numfbins = (INT4)(round(params->fspan*params->Tcoh)+1);   //Number of frequency bins
   numffts = (INT4)floor(params->Tobs/(params->Tcoh-params->SFToverlap)-1);   //Number of FFTs
   
   REAL4Vector *psd1 = XLALCreateREAL4Vector((UINT4)(numfbins*numffts));
   INT4Vector *freqbins = XLALCreateINT4Vector((UINT4)numfbins);
   if (psd1==NULL) {
      fprintf(stderr,"%s: XLALCreateREAL4Vector(%d) failed.\n", fn, numfbins*numffts);
      XLAL_ERROR_VOID(fn, XLAL_EFUNC);
   } else if (freqbins==NULL) {
      fprintf(stderr,"%s: XLALCreateINT4Vector(%d) failed.\n", fn, numfbins);
      XLAL_ERROR_VOID(fn, XLAL_EFUNC);
   }
   
   REAL4 periodf = 1.0/input.period;
   REAL4 B = input.moddepth*params->Tcoh;
   
   //Bin numbers of the frequencies
   for (ii=0; ii<numfbins; ii++) freqbins->data[ii] = (INT4)roundf(params->fmin*params->Tcoh) + ii;
   
   //Determine the signal modulation in bins with time at center of coherence time and create
   //Hann windowed PSDs
   REAL8 sin2pix = 0.0, cos2pix = 0.0;
   for (ii=0; ii<numffts; ii++) {
      REAL4 t = 0.5*params->Tcoh*ii;  //Assumed 50% overlapping SFTs
      //REAL4 n0 = B*sin(LAL_TWOPI*periodf*t) + input.fsig*params->Tcoh;
      twospect_sin_cos_2PI_LUT(&sin2pix, &cos2pix, periodf*t);
      REAL4 n0 = B*sin2pix + input.fsig*params->Tcoh;
      if (sftexist->data[ii]==1) {
         for (jj=0; jj<numfbins; jj++) {
            //Create windowed PSD values
            //if ( fabs(n0-freqbins->data[jj]) <= 5.0 ) psd1->data[ii*numfbins + jj] = 2.0/3.0*params->Tcoh*sincxoverxsqminusone(n0-freqbins->data[jj])*sincxoverxsqminusone(n0-freqbins->data[jj]);
            
            //if ( fabs(n0-freqbins->data[jj]) <= 5.0 ) {
               //psd1->data[ii*numfbins + jj] = sincxoverxsqminusone(n0-freqbins->data[jj]);
               //psd1->data[ii*numfbins + jj] *= psd1->data[ii*numfbins + jj];
               //psd1->data[ii*numfbins + jj] *= (2.0/3.0)*params->Tcoh;
            //}
            
            if ( fabs(n0-freqbins->data[jj]) <= 5.0 ) psd1->data[ii*numfbins + jj] = sqsincxoverxsqminusone(n0-freqbins->data[jj])*(2.0/3.0)*params->Tcoh;
            else psd1->data[ii*numfbins + jj] = 0.0;
         } /* for jj < numfbins */
      } else {
         for (jj=0; jj<numfbins; jj++) psd1->data[ii*numfbins + jj] = 0.0;
      }

   } /* for ii < numffts */
   
   //Do the second FFT
   REAL4Vector *x = XLALCreateREAL4Vector((UINT4)numffts);
   REAL4Window *win = XLALCreateHannREAL4Window(x->length);
   REAL4Vector *psd = XLALCreateREAL4Vector((UINT4)floor(x->length*0.5)+1);
   if (x==NULL) {
      fprintf(stderr,"%s: XLALCreateREAL4Vector(%d) failed.\n", fn, numffts);
      XLAL_ERROR_VOID(fn, XLAL_EFUNC);
   } else if (win==NULL) {
      fprintf(stderr,"%s: XLALCreateHannREAL4Window(%d) failed.\n", fn, x->length);
      XLAL_ERROR_VOID(fn, XLAL_EFUNC);
   } else if (psd==NULL) {
      fprintf(stderr,"%s: XLALCreateREAL4Vector(%d) failed.\n", fn, (UINT4)floor(x->length*0.5)+1);
      XLAL_ERROR_VOID(fn, XLAL_EFUNC);
   }
   REAL4 winFactor = 8.0/3.0;
   REAL8 sum = 0.0;
   INT4 doSecondFFT;
   //First loop over frequencies
   for (ii=0; ii<numfbins; ii++) {
      //Set doSecondFFT check flag to 0. Value becomes 1 if at least one element in frequency row is non-zero
      doSecondFFT = 0;
   
      //Next, loop over times
      for (jj=0; jj<(INT4)x->length; jj++) {
         //Check, do we need to do the second FFT...?
         if (doSecondFFT==0 && psd1->data[ii+jj*numfbins]>0.0) {
            doSecondFFT = 1;
            jj = (INT4)x->length;  //End the search for bins with power
         }
      }
      
      //If there was power in the frequency bin of the template, then do the FFT
      if (doSecondFFT==1) {
         //Obtain and window the time series
         x = SSVectorMultiply_with_stride_and_offset(x, psd1, win->data, numfbins, 1, ii, 0);
         if (xlalErrno!=0) {
            fprintf(stderr,"%s, SSVectorMultiply_with_stride_and_offset() failed.\n", fn);
            XLAL_ERROR_VOID(fn, XLAL_EFUNC);
         }
         
         //Do the FFT
         if (XLALREAL4PowerSpectrum(psd,x,plan) != 0) {
            fprintf(stderr,"%s: XLALREAL4PowerSpectrum() failed.\n", fn);
            XLAL_ERROR_VOID(fn, XLAL_EFUNC);
         }
      } /* if doSecondFFT */
      
      //Scale the data points by 1/N and window factor and (1/fs)
      //Order of vector is by second frequency then first frequency
      //Ignore the DC to 3rd frequency bins
      if (doSecondFFT==1) {
         for (jj=4; jj<(INT4)psd->length; jj++) {
            REAL4 correctedValue = psd->data[jj]*winFactor/x->length*0.5*params->Tcoh;
            
            //Sum the total weights
            sum += correctedValue;
            
            //Sort the weights, insertion sort technique
            if (correctedValue > output->templatedata->data[output->templatedata->length-1]) {
               insertionSort_template(output, correctedValue, ii*psd->length+jj, ii, jj);
            }
         } /* for jj < psd->length */
      } /* if(doSecondFFT) */
      
   } /* if ii < numfbins */
   
   //Normalize
   for (ii=0; ii<(INT4)output->templatedata->length; ii++) if (output->templatedata->data[ii]!=0.0) output->templatedata->data[ii] /= (REAL4)sum;
   
   //Destroy
   XLALDestroyREAL4Vector(psd1);
   XLALDestroyINT4Vector(freqbins);
   XLALDestroyREAL4Vector(x);
   XLALDestroyREAL4Window(win);
   XLALDestroyREAL4Vector(psd);
   
}


void bruteForceTemplateSearch(candidate *output, candidate input, REAL8 fminimum, REAL8 fmaximum, INT4 numfsteps, INT4 numperiods, REAL8 dfmin, REAL8 dfmax, INT4 numdfsteps, inputParamsStruct *params, REAL4Vector *ffdata, INT4Vector *sftexist, REAL4Vector *aveNoise, REAL4Vector *aveTFnoisePerFbinRatio, REAL4FFTPlan *secondFFTplan, INT4 useExactTemplates)
{
   
   const CHAR *fn = __func__;
   
   INT4 ii, jj, kk;
   REAL8Vector *trialf, *trialb, *trialp;
   REAL8 fstepsize, dfstepsize;
   REAL4 tcohfactor = 1.49e-3*params->Tcoh + 1.76;
   
   //Set up parameters of modulation depth search
   if (dfmin<(0.5/params->Tcoh-1.0e-9)) dfmin = 0.5/params->Tcoh;
   trialb = XLALCreateREAL8Vector(numdfsteps);
   if (trialb==NULL) {
      fprintf(stderr,"%s: XLALCreateREAL8Vector(%d) failed.\n", fn, numdfsteps);
      XLAL_ERROR_VOID(fn, XLAL_EFUNC);
   }
   dfstepsize = (dfmax-dfmin)/(REAL8)(numdfsteps-1);
   for (ii=0; ii<numdfsteps; ii++) trialb->data[ii] = dfmin + dfstepsize*ii;
   
   //Set up parameters of signal frequency search
   if (fminimum<params->fmin) fminimum = params->fmin;
   if (fmaximum>params->fmin+params->fspan) fmaximum = params->fmin+params->fspan;
   trialf = XLALCreateREAL8Vector(numfsteps);
   if (trialf==NULL) {
      fprintf(stderr,"%s: XLALCreateREAL8Vector(%d) failed.\n", fn, numfsteps);
      XLAL_ERROR_VOID(fn, XLAL_EFUNC);
   }
   fstepsize = (fmaximum-fminimum)/(REAL8)(numfsteps-1);
   for (ii=0; ii<numfsteps; ii++) trialf->data[ii] = fminimum + fstepsize*ii;
   
   //Search over numperiods different periods
   trialp = XLALCreateREAL8Vector(numperiods);
   if (trialp==NULL) {
      fprintf(stderr,"%s: XLALCreateREAL8Vector(%d) failed.\n", fn, numperiods);
      XLAL_ERROR_VOID(fn, XLAL_EFUNC);
   }
   
   //Now search over the parameter space. Frequency, then modulation depth, then period
   //Initialze best values as the initial point we are searching around
   INT4 bestproberrcode = 0;
   REAL8 bestf = 0.0, bestp = 0.0, bestdf = 0.0, bestR = 0.0, besth0 = 0.0, bestProb = 0.0;
   candidate cand;
   templateStruct *template = new_templateStruct(params->templatelength);
   if (template==NULL) {
      fprintf(stderr,"%s: new_templateStruct(%d) failed.\n", fn, params->templatelength);
      XLAL_ERROR_VOID(fn, XLAL_EFUNC); 
   }
   farStruct *farval = NULL;
   if (params->calcRthreshold) {
      farval = new_farStruct();
      if (farval==NULL) {
         fprintf(stderr,"%s: new_farStruct() failed.\n", fn);
         XLAL_ERROR_VOID(fn, XLAL_EFUNC); 
      }
   }
   
   INT4 midposition = (INT4)roundf((numperiods-1)*0.5), proberrcode = 0;
   for (ii=0; ii<(INT4)trialf->length; ii++) {
      for (jj=0; jj<(INT4)trialb->length; jj++) {
         //Start with period of the first guess, then determine nearest neighbor from the
         //modulation depth amplitude to find the other period guesses. These parameters 
         //are determined from simulation to scale the N.N. distance w.r.t. mod. depth with
         //20% mismatch parameter
         trialp->data[midposition] = input.period;
         for (kk=0; kk<midposition; kk++) {
            REAL8 nnp = trialp->data[midposition+kk]*trialp->data[midposition+kk]*(1+trialp->data[midposition+kk]/tcohfactor/params->Tobs)/tcohfactor/params->Tobs*sqrt(3.6e-3/trialb->data[jj]);
            trialp->data[midposition+(kk+1)] = trialp->data[midposition+kk] + nnp;
            nnp = trialp->data[midposition-kk]*trialp->data[midposition-kk]*(1+trialp->data[midposition-kk]/tcohfactor/params->Tobs)/tcohfactor/params->Tobs*sqrt(3.6e-3/trialb->data[jj]);
            trialp->data[midposition-(kk+1)] = trialp->data[midposition-kk] - nnp;
         }
         
         for (kk=0; kk<(INT4)trialp->length; kk++) {
            if ( (trialf->data[ii]-trialb->data[jj]-6.0/params->Tcoh)>params->fmin && 
                (trialf->data[ii]+trialb->data[jj]+6.0/params->Tcoh)<(params->fmin+params->fspan) && 
                trialb->data[jj]<maxModDepth(trialp->data[kk], params->Tcoh) && 
                trialp->data[kk]>minPeriod(trialb->data[jj], params->Tcoh) && 
                trialp->data[kk]<=(0.2*params->Tobs) && 
                trialp->data[kk]>=(2.0*3600.0) && 
                trialb->data[jj]>=params->dfmin && 
                trialb->data[jj]<=params->dfmax && 
                trialp->data[kk]<=params->Pmax && 
                trialp->data[kk]>=params->Pmin ) {
               
               loadCandidateData(&cand, trialf->data[ii], trialp->data[kk], trialb->data[jj], input.ra, input.dec, 0, 0, 0.0, 0, 0.0);
               
               if (useExactTemplates!=0) {
                  makeTemplate(template, cand, params, sftexist, secondFFTplan);
                  if (xlalErrno!=0) {
                     fprintf(stderr,"%s: makeTemplate() failed.\n", fn);
                     XLAL_ERROR_VOID(fn, XLAL_EFUNC);
                  }
               } else {
                  makeTemplateGaussians(template, cand, params);
                  if (xlalErrno!=0) {
                     fprintf(stderr,"%s: makeTemplateGaussians() failed.\n", fn);
                     XLAL_ERROR_VOID(fn, XLAL_EFUNC);
                  }
               }
               
               if (params->calcRthreshold && bestProb==0.0) {
                  numericFAR(farval, template, params->templatefar, aveNoise, aveTFnoisePerFbinRatio, params->rootFindingMethod);
                  if (xlalErrno!=0) {
                     fprintf(stderr,"%s: numericFAR() failed.\n", fn);
                     XLAL_ERROR_VOID(fn, XLAL_EFUNC);
                  }
               }
               
               REAL8 R = calculateR(ffdata, template, aveNoise, aveTFnoisePerFbinRatio);
               if (XLAL_IS_REAL8_FAIL_NAN(R)) {
                  fprintf(stderr,"%s: calculateR() failed.\n", fn);
                  XLAL_ERROR_VOID(fn, XLAL_EFUNC);
               }
               REAL8 prob = probR(template, aveNoise, aveTFnoisePerFbinRatio, R, &proberrcode);
               if (XLAL_IS_REAL8_FAIL_NAN(prob)) {
                  fprintf(stderr,"%s: probR() failed.\n", fn);
                  XLAL_ERROR_VOID(fn, XLAL_EFUNC);
               }
               
               REAL8 h0 = 2.7426*pow(R/(params->Tcoh*params->Tobs),0.25);
               
               if ( (bestProb!=0.0 && prob < bestProb) || (bestProb==0.0 && !params->calcRthreshold && prob<log10(params->templatefar)) || (bestProb==0.0 && params->calcRthreshold && R > farval->far) ) {
                  bestf = trialf->data[ii];
                  bestp = trialp->data[kk];
                  bestdf = trialb->data[jj];
                  bestR = R;
                  besth0 = h0;
                  bestProb = prob;
                  bestproberrcode = proberrcode;
               }
               
            } /* if within boundaries */
         } /* for kk < trialp */
      } /* for jj < trialb */
   } /* for ii < trialf */
   free_templateStruct(template);
   template = NULL;
   if (params->calcRthreshold) {
      free_farStruct(farval);
      farval = NULL;
   }
   XLALDestroyREAL8Vector(trialf);
   XLALDestroyREAL8Vector(trialb);
   XLALDestroyREAL8Vector(trialp);
   trialf = NULL;
   trialb = NULL;
   trialp = NULL;
   
   if (bestProb==0.0) {
      loadCandidateData(output, input.fsig, input.period, input.moddepth, input.ra, input.dec, input.stat, input.h0, input.prob, input.proberrcode, input.normalization);
   } else {
      loadCandidateData(output, bestf, bestp, bestdf, input.ra, input.dec, bestR, besth0, bestProb, bestproberrcode, input.normalization);
   }
   
}


void efficientTemplateSearch(candidate *output, candidate input, REAL8 fminimum, REAL8 fmaximum, REAL8 minfstep, INT4 numperiods, REAL8 dfmin, REAL8 dfmax, REAL8 minDfstep, inputParamsStruct *params, REAL4Vector *ffdata, INT4Vector *sftexist, REAL4Vector *aveNoise, REAL4Vector *aveTFnoisePerFbinRatio, REAL4FFTPlan *secondFFTplan, INT4 useExactTemplates)
{
   
   const CHAR *fn = __func__;
   
   INT4 bestproberrcode = 0, ii, jj, kk;
   REAL8 bestf = input.fsig, bestp = input.period, bestdf = input.moddepth, bestR = input.stat, besth0 = input.h0, bestProb = input.prob, fstepsize = 0.25*(fmaximum-fminimum), dfstepsize = 0.25*(dfmax-dfmin);
   REAL4 tcohfactor = 1.49e-3*params->Tcoh + 1.76;
   candidate cand;
   
   templateStruct *template = new_templateStruct(params->templatelength);
   if (template==NULL) {
      fprintf(stderr,"%s: new_templateStruct(%d) failed.\n", fn, params->templatelength);
      XLAL_ERROR_VOID(fn, XLAL_EFUNC); 
   }
   farStruct *farval = NULL;
   if (params->calcRthreshold) {
      farval = new_farStruct();
      if (farval==NULL) {
         fprintf(stderr,"%s: new_farStruct() failed.\n", fn);
         XLAL_ERROR_VOID(fn, XLAL_EFUNC);
      }
   }
   
   //Search over numperiods different periods
   REAL8Vector *trialp = XLALCreateREAL8Vector(numperiods);
   if (trialp==NULL) {
      fprintf(stderr,"%s: XLALCreateREAL8Vector(%d) failed.\n", fn, numperiods);
      XLAL_ERROR_VOID(fn, XLAL_EFUNC);
   }
   INT4 midposition = (INT4)roundf((numperiods-1)*0.5), proberrcode = 0;
   trialp->data[midposition] = input.period;
   
   REAL8Vector *trialf = XLALCreateREAL8Vector(2);
   if (trialf==NULL) {
      fprintf(stderr,"%s: XLALCreateREAL8Vector(%d) failed.\n", fn, 2);
      XLAL_ERROR_VOID(fn, XLAL_EFUNC);
   }
   REAL8Vector *trialb = XLALCreateREAL8Vector(2);
   if (trialb==NULL) {
      fprintf(stderr,"%s: XLALCreateREAL8Vector(%d) failed.\n", fn, 2);
      XLAL_ERROR_VOID(fn, XLAL_EFUNC);
   }
   
   while (fstepsize-minfstep>1.0e-8 && dfstepsize-minDfstep>1.0e-8) {
      //Initial point with template
      loadCandidateData(&cand, bestf, bestp, bestdf, input.ra, input.dec, 0, 0, 0.0, 0, 0.0);
      if (useExactTemplates!=0) {
         makeTemplate(template, cand, params, sftexist, secondFFTplan);
         if (xlalErrno!=0) {
            fprintf(stderr,"%s: makeTemplate() failed.\n", fn);
            XLAL_ERROR_VOID(fn, XLAL_EFUNC);
         }
      } else {
         makeTemplateGaussians(template, cand, params);
         if (xlalErrno!=0) {
            fprintf(stderr,"%s: makeTemplateGaussians() failed.\n", fn);
            XLAL_ERROR_VOID(fn, XLAL_EFUNC);
         }
      }
      if (params->calcRthreshold && bestProb==0.0) {
         numericFAR(farval, template, params->templatefar, aveNoise, aveTFnoisePerFbinRatio, params->rootFindingMethod);
         if (xlalErrno!=0) {
            fprintf(stderr,"%s: numericFAR() failed.\n", fn);
            XLAL_ERROR_VOID(fn, XLAL_EFUNC);
         }
      }
      bestR = calculateR(ffdata, template, aveNoise, aveTFnoisePerFbinRatio);
      if (XLAL_IS_REAL8_FAIL_NAN(bestR)) {
         fprintf(stderr,"%s: calculateR() failed.\n", fn);
         XLAL_ERROR_VOID(fn, XLAL_EFUNC);
      }
      bestProb = probR(template, aveNoise, aveTFnoisePerFbinRatio, bestR, &bestproberrcode);
      if (XLAL_IS_REAL8_FAIL_NAN(bestProb)) {
         fprintf(stderr,"%s: probR() failed.\n", fn);
         XLAL_ERROR_VOID(fn, XLAL_EFUNC);
      }
      besth0 = 2.7426*pow(bestR/(params->Tcoh*params->Tobs),0.25);
      
      for (ii=0; ii<(INT4)trialf->length; ii++) {
         if (ii==0) trialf->data[ii] = bestf - fstepsize;
         else trialf->data[ii] = bestf + fstepsize;
      }
      for (ii=0; ii<(INT4)trialb->length; ii++) {
         if (ii==0) trialb->data[ii] = bestdf - dfstepsize;
         else trialb->data[ii] = bestdf + dfstepsize;
      }
      
      INT4 movedtoabetterpoint = 0;
      for (ii=0; ii<(INT4)trialf->length; ii++) {
         for (jj=0; jj<(INT4)trialb->length; jj++) {
            for (kk=0; kk<midposition; kk++) {
               REAL8 nnp = trialp->data[midposition+kk]*trialp->data[midposition+kk]*(1+trialp->data[midposition+kk]/tcohfactor/params->Tobs)/tcohfactor/params->Tobs*sqrt(3.6e-3/trialb->data[jj]);
               trialp->data[midposition+(kk+1)] = trialp->data[midposition+kk] + nnp;
               nnp = trialp->data[midposition-kk]*trialp->data[midposition-kk]*(1+trialp->data[midposition-kk]/tcohfactor/params->Tobs)/tcohfactor/params->Tobs*sqrt(3.6e-3/trialb->data[jj]);
               trialp->data[midposition-(kk+1)] = trialp->data[midposition-kk] - nnp;
            }
            for (kk=0; kk<(INT4)trialp->length; kk++) {
               if ( (trialf->data[ii]-trialb->data[jj]-6.0/params->Tcoh)>params->fmin && 
                   (trialf->data[ii]+trialb->data[jj]+6.0/params->Tcoh)<(params->fmin+params->fspan) && 
                   trialb->data[jj]<maxModDepth(trialp->data[kk], params->Tcoh) && 
                   trialp->data[kk]>minPeriod(trialb->data[jj], params->Tcoh) && 
                   trialp->data[kk]<=(0.2*params->Tobs) && 
                   trialp->data[kk]>=(2.0*3600.0) && 
                   trialb->data[jj]>=params->dfmin && 
                   trialb->data[jj]<=params->dfmax && 
                   trialp->data[kk]<=params->Pmax && 
                   trialp->data[kk]>=params->Pmin && 
                   trialf->data[ii]>=fminimum && 
                   trialf->data[ii]<=fmaximum && 
                   trialb->data[jj]>=dfmin && 
                   trialb->data[jj]<=dfmax ) {
                  
                  loadCandidateData(&cand, trialf->data[ii], trialp->data[kk], trialb->data[jj], input.ra, input.dec, 0, 0, 0.0, 0, 0.0);
                  
                  if (useExactTemplates!=0) {
                     makeTemplate(template, cand, params, sftexist, secondFFTplan);
                     if (xlalErrno!=0) {
                        fprintf(stderr,"%s: makeTemplate() failed.\n", fn);
                        XLAL_ERROR_VOID(fn, XLAL_EFUNC);
                     }
                  } else {
                     makeTemplateGaussians(template, cand, params);
                     if (xlalErrno!=0) {
                        fprintf(stderr,"%s: makeTemplateGaussians() failed.\n", fn);
                        XLAL_ERROR_VOID(fn, XLAL_EFUNC);
                     }
                  }
                  
                  if (params->calcRthreshold && bestProb==0.0) {
                     numericFAR(farval, template, params->templatefar, aveNoise, aveTFnoisePerFbinRatio, params->rootFindingMethod);
                     if (xlalErrno!=0) {
                        fprintf(stderr,"%s: numericFAR() failed.\n", fn);
                        XLAL_ERROR_VOID(fn, XLAL_EFUNC);
                     }
                  }
                  
                  REAL8 R = calculateR(ffdata, template, aveNoise, aveTFnoisePerFbinRatio);
                  if (XLAL_IS_REAL8_FAIL_NAN(R)) {
                     fprintf(stderr,"%s: calculateR() failed.\n", fn);
                     XLAL_ERROR_VOID(fn, XLAL_EFUNC);
                  }
                  REAL8 prob = probR(template, aveNoise, aveTFnoisePerFbinRatio, R, &proberrcode);
                  if (XLAL_IS_REAL8_FAIL_NAN(prob)) {
                     fprintf(stderr,"%s: probR() failed.\n", fn);
                     XLAL_ERROR_VOID(fn, XLAL_EFUNC);
                  }
                  
                  REAL8 h0 = 2.7426*pow(R/(params->Tcoh*params->Tobs),0.25);
                  
                  if ( prob < bestProb ) {
                     bestf = trialf->data[ii];
                     bestp = trialp->data[kk];
                     bestdf = trialb->data[jj];
                     bestR = R;
                     besth0 = h0;
                     bestProb = prob;
                     bestproberrcode = proberrcode;
                     movedtoabetterpoint = 1;
                  }
                  
               } // if within boundaries
            } // for kk < trialp->length
         } // for jj < trialb->length
      } // for ii < trialf->length
      
      if (movedtoabetterpoint==0) {
         fstepsize *= 0.5;
         dfstepsize *= 0.5;
      }
      
   } // while
   
   
   loadCandidateData(output, bestf, bestp, bestdf, input.ra, input.dec, bestR, besth0, bestProb, bestproberrcode, input.normalization);
   
   
   free_templateStruct(template);
   template = NULL;
   if (params->calcRthreshold) {
      free_farStruct(farval);
      farval = NULL;
   }
   XLALDestroyREAL8Vector(trialp);
   XLALDestroyREAL8Vector(trialf);
   XLALDestroyREAL8Vector(trialb);
   
}


//////////////////////////////////////////////////////////////
// Does the insertion sort for the template weights
void insertionSort_template(templateStruct *output, REAL4 weight, INT4 pixelloc, INT4 firstfftfreq, INT4 secfftfreq)
{
   
   INT4 ii;
   
   INT4 insertionpoint = (INT4)output->templatedata->length-1;
   while (insertionpoint > 0 && weight > output->templatedata->data[insertionpoint-1]) insertionpoint--;
   
   for (ii=output->templatedata->length-1; ii>insertionpoint; ii--) {
      output->templatedata->data[ii] = output->templatedata->data[ii-1];
      output->pixellocations->data[ii] = output->pixellocations->data[ii-1];
      output->firstfftfrequenciesofpixels->data[ii] = output->firstfftfrequenciesofpixels->data[ii-1];
      output->secondfftfrequencies->data[ii] = output->secondfftfrequencies->data[ii-1];
   }
   
   output->templatedata->data[insertionpoint] = weight;
   output->pixellocations->data[insertionpoint] = pixelloc;
   output->firstfftfrequenciesofpixels->data[insertionpoint] = firstfftfreq;
   output->secondfftfrequencies->data[insertionpoint] = secfftfreq;
   
} /* insertionSort_template() */



//////////////////////////////////////////////////////////////
// Calculates y = sin(pi*x)/(pi*x)/(x^2-1)
REAL8 sincxoverxsqminusone(REAL8 x)
{
   
   if (fabs(x*x-1.0)<1.0e-8) return -0.5;
   else return gsl_sf_sinc(x)/(x*x-1.0);
   
} /* sincxoverxsqminusone() */
REAL8 sqsincxoverxsqminusone(REAL8 x)
{
   
   REAL8 val = sincxoverxsqminusone(x);
   return val*val;
   
} /* sqsincxoverxsqminusone() */


//Stolen from computeFstat.c with higher resolution
#define OOTWOPI         (1.0 / LAL_TWOPI)
int twospect_sin_cos_LUT(REAL8 *sinx, REAL8 *cosx, REAL8 x)
{
   return twospect_sin_cos_2PI_LUT( sinx, cosx, x * OOTWOPI );
} /* twospect_sin_cos_LUT() */
#define LUT_RES         1024      /* resolution of lookup-table */
#define LUT_RES_F       (1.0 * LUT_RES)
#define OO_LUT_RES      (1.0 / LUT_RES)
#define X_TO_IND        (1.0 * LUT_RES * OOTWOPI )
#define IND_TO_X        (LAL_TWOPI * OO_LUT_RES)
#define TRUE (1==1)
#define FALSE (1==0)
int twospect_sin_cos_2PI_LUT(REAL8 *sin2pix, REAL8 *cos2pix, REAL8 x)
{
   REAL8 xt;
   INT4 i0;
   REAL8 d, d2;
   REAL8 ts, tc;
   REAL8 dummy;
   
   static BOOLEAN firstCall = TRUE;
   static REAL4 sinVal[LUT_RES+1], cosVal[LUT_RES+1];
   
   /* the first time we get called, we set up the lookup-table */
   if ( firstCall ) {
      UINT4 k;
      for (k=0; k <= LUT_RES; k++) {
         sinVal[k] = sin( LAL_TWOPI * k * OO_LUT_RES );
         cosVal[k] = cos( LAL_TWOPI * k * OO_LUT_RES );
      }
      firstCall = FALSE;
   }

   /* we only need the fractional part of 'x', which is number of cylces,
   * this was previously done using
   *   xt = x - (INT4)x;
   * which is numerically unsafe for x > LAL_INT4_MAX ~ 2e9
   * for saftey we therefore rather use modf(), even if that
   * will be somewhat slower...
   */
   //xt = modf(x, &dummy);/* xt in (-1, 1) */
   if (x<9.2233720368547e18) xt = x - (INT8)x;  // if x < roughly LAL_REAL8_MAX
   else xt = modf(x, &dummy);
   
   if ( xt < 0.0 ) xt += 1.0;                  /* xt in [0, 1 ) */
   #ifndef LAL_NDEBUG
      if ( xt < 0.0 || xt > 1.0 ) {
         XLALPrintError("\nFailed numerica in twospect_sin_cos_2PI_LUT(): xt = %f not in [0,1)\n\n", xt );
         return XLAL_FAILURE;
      }
   #endif
   
   i0 = (INT4)( xt * LUT_RES_F + 0.5 );  /* i0 in [0, LUT_RES ] */
   d = d2 = LAL_TWOPI * (xt - OO_LUT_RES * i0);
   d2 *= 0.5 * d;
   
   ts = sinVal[i0];
   tc = cosVal[i0];
   
   /* use Taylor-expansions for sin/cos around LUT-points */
   (*sin2pix) = ts + d * tc - d2 * ts;
   (*cos2pix) = tc - d * ts - d2 * tc;
   
   return XLAL_SUCCESS;
} /* twospect_sin_cos_2PI_LUT() */




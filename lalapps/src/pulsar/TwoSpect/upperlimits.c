/*
 *  Copyright (C) 2011 Evan Goetz
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

#include <lal/LALMalloc.h>

#include <gsl/gsl_roots.h>

#include "upperlimits.h"
#include "statistics.h"
#include "IHS.h"

UpperLimitVector * new_UpperLimitVector(UINT4 length)
{
   
   INT4 ii;
   
   UpperLimitVector *vector = XLALMalloc(sizeof(*vector));
   if (vector==NULL) {
      fprintf(stderr,"%s: XLALMalloc(%zu) failed.\n", __func__, sizeof(*vector));
      XLAL_ERROR_NULL(XLAL_ENOMEM);
   }
   
   vector->length = length;
   if (length==0) vector->data = NULL;
   else {
      vector->data = XLALMalloc( length*sizeof(*vector->data) );
      if (vector->data==NULL) {
         XLALFree((UpperLimitVector*)vector);
         fprintf(stderr,"%s: XLALMalloc(%zu) failed.\n", __func__, length*sizeof(*vector->data));
         XLAL_ERROR_NULL(XLAL_ENOMEM);
      }
      for (ii=0; ii<(INT4)length; ii++) reset_UpperLimitStruct(&(vector->data[ii]));
   }
   
   return vector;
   
} /* new_UpperLimitVector() */



UpperLimitVector * resize_UpperLimitVector(UpperLimitVector *vector, UINT4 length)
{
   
   if (vector==NULL) return new_UpperLimitVector(length);
   if (length==0) {
      free_UpperLimitVector(vector);
      return NULL;
   }
   
   UINT4 oldlength = vector->length;
   INT4 ii;
   
   vector->data = XLALRealloc(vector->data, length*sizeof(*vector->data));
   if (vector->data==NULL) {
      vector->length = 0;
      fprintf(stderr,"%s: XLALRealloc(%zu) failed.\n", __func__, length*sizeof(*vector->data));
      XLAL_ERROR_NULL(XLAL_ENOMEM);
   }
   vector->length = length;
   for (ii=(INT4)oldlength; ii<(INT4)length; ii++) reset_UpperLimitStruct(&(vector->data[ii]));
   
   return vector;
   
} /* resize_UpperLimitVector() */



void free_UpperLimitVector(UpperLimitVector *vector)
{
   
   INT4 ii;
   
   if (vector==NULL) return;
   if ((!vector->length || !vector->data) && (vector->length || vector->data)) XLAL_ERROR_VOID(XLAL_EINVAL);
   if (vector->data) {
      for (ii=0; ii<(INT4)vector->length; ii++) free_UpperLimitStruct(&(vector->data[ii]));
      XLALFree((UpperLimit*)vector->data);
   }
   vector->data = NULL;
   XLALFree((UpperLimitVector*)vector);
   return;
   
} /* free_UpperLimitVector() */



void reset_UpperLimitStruct(UpperLimit *ul)
{
   
   ul->fsig = NULL;
   ul->period = NULL;
   ul->moddepth = NULL;
   ul->ULval = NULL;
   ul->effSNRval = NULL;
   
}
void free_UpperLimitStruct(UpperLimit *ul)
{
   
   if (ul->fsig) {
      XLALDestroyREAL8Vector(ul->fsig);
      ul->fsig = NULL;
   }
   if (ul->period) {
      XLALDestroyREAL8Vector(ul->period);
      ul->period = NULL;
   }
   if (ul->moddepth) {
      XLALDestroyREAL8Vector(ul->moddepth);
      ul->moddepth = NULL;
   }
   if (ul->ULval) {
      XLALDestroyREAL8Vector(ul->ULval);
      ul->ULval = NULL;
   }
   if (ul->effSNRval) {
      XLALDestroyREAL8Vector(ul->effSNRval);
      ul->effSNRval = NULL;
   }
   
}



void skypoint95UL(UpperLimit *ul, inputParamsStruct *params, ffdataStruct *ffdata, ihsMaximaStruct *ihsmaxima, ihsfarStruct *ihsfar, REAL4Vector *fbinavgs)
{
   
   INT4 ii, jj, kk;
   
   INT4 minrows = (INT4)round(2.0*params->dfmin*params->Tcoh)+1;
   
   ul->fsig = XLALCreateREAL8Vector((ihsmaxima->rows-minrows)+1);
   ul->period = XLALCreateREAL8Vector((ihsmaxima->rows-minrows)+1);
   ul->moddepth = XLALCreateREAL8Vector((ihsmaxima->rows-minrows)+1);
   ul->ULval = XLALCreateREAL8Vector((ihsmaxima->rows-minrows)+1);
   ul->effSNRval = XLALCreateREAL8Vector((ihsmaxima->rows-minrows)+1);
   if (ul->fsig==NULL) {
      fprintf(stderr, "%s: XLALCreateREAL8Vector(%d) failed.\n", __func__, (ihsmaxima->rows-minrows)+1);
      XLAL_ERROR_VOID(XLAL_EFUNC);
   } else if (ul->period==NULL) {
      fprintf(stderr, "%s: XLALCreateREAL8Vector(%d) failed.\n", __func__, (ihsmaxima->rows-minrows)+1);
      XLAL_ERROR_VOID(XLAL_EFUNC);
   } else if (ul->moddepth==NULL) {
      fprintf(stderr, "%s: XLALCreateREAL8Vector(%d) failed.\n", __func__, (ihsmaxima->rows-minrows)+1);
      XLAL_ERROR_VOID(XLAL_EFUNC);
   } else if (ul->ULval==NULL) {
      fprintf(stderr, "%s: XLALCreateREAL8Vector(%d) failed.\n", __func__, (ihsmaxima->rows-minrows)+1);
      XLAL_ERROR_VOID(XLAL_EFUNC);
   } else if (ul->effSNRval==NULL) {
      fprintf(stderr, "%s: XLALCreateREAL8Vector(%d) failed.\n", __func__, (ihsmaxima->rows-minrows)+1);
      XLAL_ERROR_VOID(XLAL_EFUNC);
   }
   
   //Initialize solver
   const gsl_root_fsolver_type *T = gsl_root_fsolver_brent;
   gsl_root_fsolver *s = gsl_root_fsolver_alloc (T);
   gsl_function F;
   struct ncx2cdf_solver_params pars;
   
   //loop over modulation depths
   for (ii=minrows; ii<=ihsmaxima->rows; ii++) {
      REAL8 loudestoutlier = 0.0, loudestoutlierminusnoise = 0.0, loudestoutliernoise = 0.0;
      INT4 jjbinofloudestoutlier = 0, locationofloudestoutlier = -1;
      REAL8 noise = 0.0, totalnoise = 0.0, ihsminusnoise = 0.0;
      INT4 startpositioninmaximavector = (ii-2)*ffdata->numfbins - ((ii-1)*(ii-1)-(ii-1))/2;
      //loop over frequency bins
      for (jj=0; jj<ffdata->numfbins-(ii-1); jj++) {
         
         INT4 locationinmaximavector = startpositioninmaximavector + jj;
         
         INT4 location = ihsmaxima->locations->data[locationinmaximavector];
         
         noise = ihsfar->expectedIHSVector->data[location-5];
         totalnoise = 0.0;
         for (kk=0; kk<ii; kk++) totalnoise += noise*fbinavgs->data[jj+kk];
         ihsminusnoise = ihsmaxima->maxima->data[locationinmaximavector] - totalnoise;
         
         REAL8 fsig = params->fmin + (0.5*(ii-1.0) + jj)/params->Tcoh;
         REAL8 moddepth = 0.5*(ii-1.0)/params->Tcoh;
         
         if (ihsminusnoise>loudestoutlierminusnoise && 
             (fsig>=params->ULfmin && fsig<=params->ULfmin+params->ULfspan) && 
             (moddepth>=params->ULmindf && moddepth<=params->ULmaxdf)) {
            loudestoutlier = ihsmaxima->maxima->data[locationinmaximavector];
            loudestoutliernoise = totalnoise;
            loudestoutlierminusnoise = ihsminusnoise;
            locationofloudestoutlier = ihsmaxima->locations->data[locationinmaximavector];
            jjbinofloudestoutlier = jj;
         }
      } /* for jj < ffdata->numfbins-(ii-1) */
      
      if (locationofloudestoutlier==-1) {
         fprintf(stderr, "%s: Failed to reach a louder outlier minus noise greater than 0\n", __func__);
         XLAL_ERROR_VOID(XLAL_EFUNC);
      }
      
      //comment or remove this
      //fprintf(stderr, "%f %f %.6f %.6f %d %d\n", params->fmin + (0.5*(ii-1.0) + jjbinofloudestoutlier)/params->Tcoh, 0.5*(ii-1.0)/params->Tcoh, loudestoutliernoise, loudestoutlierminusnoise, locationofloudestoutlier, jjbinofloudestoutlier);
      
      REAL8 initialguess = ncx2inv(0.95, 2.0*loudestoutliernoise, 2.0*loudestoutlierminusnoise);
      if (XLAL_IS_REAL8_FAIL_NAN(initialguess)) {
         fprintf(stderr, "%s: ncx2inv(%f,%f,%f) failed.\n", __func__, 0.95, 2.0*loudestoutliernoise, 2.0*loudestoutlierminusnoise);
         XLAL_ERROR_VOID(XLAL_EFUNC);
      }
      REAL8 lo = 0.001*initialguess, hi = 10.0*initialguess;
      pars.val = 2.0*loudestoutlier;
      pars.dof = 2.0*loudestoutliernoise;
      pars.ULpercent = 0.95;
      F.function = &gsl_ncx2cdf_float_withouttinyprob_solver;
      F.params = &pars;
      if (gsl_root_fsolver_set(s, &F, lo, hi) != 0) {
         fprintf(stderr,"%s: gsl_root_fsolver_set() failed.\n", __func__);
         XLAL_ERROR_VOID(XLAL_EFUNC);
      }
      
      INT4 status = GSL_CONTINUE;
      INT4 max_iter = 100;
      REAL8 root = 0.0;
      jj = 0;
      while (status==GSL_CONTINUE && jj<max_iter) {
         jj++;
         status = gsl_root_fsolver_iterate(s);
         if (status!=GSL_CONTINUE && status!=GSL_SUCCESS) {
            fprintf(stderr,"%s: gsl_root_fsolver_iterate() failed with code %d.\n", __func__, status);
            XLAL_ERROR_VOID(XLAL_EFUNC);
         }
         root = gsl_root_fsolver_root(s);
         lo = gsl_root_fsolver_x_lower(s);
         hi = gsl_root_fsolver_x_upper(s);
         status = gsl_root_test_interval(lo, hi, 0.0, 0.001);
         if (status!=GSL_CONTINUE && status!=GSL_SUCCESS) {
            fprintf(stderr,"%s: gsl_root_test_interval() failed with code %d.\n", __func__, status);
            XLAL_ERROR_VOID(XLAL_EFUNC);
         }
      }
      if (status != GSL_SUCCESS) {
         fprintf(stderr, "%s: Root finding iteration (%d/%d) failed with code %d. Current root = %f\n", __func__, jj, max_iter, status, root);
         XLAL_ERROR_VOID(XLAL_FAILURE);
      } else if (jj==max_iter) {
         fprintf(stderr, "%s: Root finding failed to converge after %d iterations", __func__, jj);
         XLAL_ERROR_VOID(XLAL_FAILURE);
      }
      
      REAL8 h0 = ihs2h0(root, params);
      if (XLAL_IS_REAL8_FAIL_NAN(h0)) {
         fprintf(stderr, "%s: ihs2h0() failed.\n", __func__);
         XLAL_ERROR_VOID(XLAL_EFUNC);
      }
      ul->fsig->data[ii-minrows] = params->fmin + (0.5*(ii-1.0) + jjbinofloudestoutlier)/params->Tcoh;
      ul->period->data[ii-minrows] = params->Tobs/locationofloudestoutlier;
      ul->moddepth->data[ii-minrows] = 0.5*(ii-1.0)/params->Tcoh;
      ul->ULval->data[ii-minrows] = h0;
      ul->effSNRval->data[ii-minrows] = unitGaussianSNR(root+pars.dof, pars.dof);
   } /* for ii=minrows --> maximum rows */
   
   gsl_root_fsolver_free(s);
   
}
REAL8 gsl_ncx2cdf_solver(REAL8 x, void *p)
{
   
   struct ncx2cdf_solver_params *params = (struct ncx2cdf_solver_params*)p;
   REAL8 val = ncx2cdf(params->val, params->dof, x);
   if (XLAL_IS_REAL8_FAIL_NAN(val)) {
      fprintf(stderr, "%s: ncx2cdf(%f, %f, %f) failed.\n", __func__, params->val, params->dof, x);
      XLAL_ERROR_REAL8(XLAL_EFUNC);
   }
   return val - (1.0-params->ULpercent);
   
}
REAL8 gsl_ncx2cdf_float_solver(REAL8 x, void *p)
{
   
   struct ncx2cdf_solver_params *params = (struct ncx2cdf_solver_params*)p;
   REAL4 val = ncx2cdf_float((REAL4)params->val, (REAL4)params->dof, (REAL4)x);
   if (XLAL_IS_REAL4_FAIL_NAN(val)) {
      fprintf(stderr, "%s: ncx2cdf_float(%f, %f, %f) failed.\n", __func__, params->val, params->dof, x);
      XLAL_ERROR_REAL8(XLAL_EFUNC);
   }
   return (REAL8)val - (1.0-params->ULpercent);
   
}
REAL8 gsl_ncx2cdf_withouttinyprob_solver(REAL8 x, void *p)
{
   
   struct ncx2cdf_solver_params *params = (struct ncx2cdf_solver_params*)p;
   REAL8 val = ncx2cdf_withouttinyprob(params->val, params->dof, x);
   if (XLAL_IS_REAL8_FAIL_NAN(val)) {
      fprintf(stderr, "%s: ncx2cdf_withouttinyprob(%f, %f, %f) failed.\n", __func__, params->val, params->dof, x);
      XLAL_ERROR_REAL8(XLAL_EFUNC);
   }
   else return val - (1.0-params->ULpercent);
   
}
REAL8 gsl_ncx2cdf_float_withouttinyprob_solver(REAL8 x, void *p)
{
   
   struct ncx2cdf_solver_params *params = (struct ncx2cdf_solver_params*)p;
   REAL4 val = ncx2cdf_float_withouttinyprob((REAL4)params->val, (REAL4)params->dof, (REAL4)x);
   if (XLAL_IS_REAL4_FAIL_NAN(val)) {
      fprintf(stderr, "%s: ncx2cdf_float_withouttinyprob(%f, %f, %f) failed.\n", __func__, params->val, params->dof, x);
      XLAL_ERROR_REAL8(XLAL_EFUNC);
   }
   else return (REAL8)val - (1.0-params->ULpercent);
   
}



void outputUpperLimitToFile(FILE *outputfile, UpperLimit ul, INT4 printAllULvalues)
{
   
   INT4 ii;
   REAL8 highesth0 = 0.0, snr = 0.0, fsig = 0.0, period = 0.0, moddepth = 0.0;
   for (ii=0; ii<(INT4)ul.moddepth->length; ii++) {
      if (printAllULvalues==1) {
         fprintf(outputfile, "%.6f %.6f %.6g %.6f %.6f %.6f %.6f %.6g\n", ul.alpha, ul.delta, ul.ULval->data[ii], ul.effSNRval->data[ii], ul.fsig->data[ii], ul.period->data[ii], ul.moddepth->data[ii], ul.normalization);
      } else if (ul.ULval->data[ii]>highesth0) {
         highesth0 = ul.ULval->data[ii];
         snr = ul.effSNRval->data[ii];
         fsig = ul.fsig->data[ii];
         period = ul.period->data[ii];
         moddepth = ul.moddepth->data[ii];
      }
   }
   if (printAllULvalues==0) {
      fprintf(outputfile, "%.6f %.6f %.6g %.6f %.6f %.6f %.6f %.6g\n", ul.alpha, ul.delta, highesth0, snr, fsig, period, moddepth, ul.normalization);
   }
   
}




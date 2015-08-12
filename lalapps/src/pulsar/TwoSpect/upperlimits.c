/*
 *  Copyright (C) 2011, 2014 Evan Goetz
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

#include <gsl/gsl_roots.h>
#include "upperlimits.h"
#include "cdfdist.h"
#include "IHS.h"


/**
 * Allocate a new UpperLimitVector
 * \param [in] length Length of the vector
 * \return Pointer to newly allocated UpperLimitVector
 */
UpperLimitVector * createUpperLimitVector(const UINT4 length)
{

   UpperLimitVector *vector = NULL;
   XLAL_CHECK_NULL( (vector = XLALMalloc(sizeof(*vector))) != NULL, XLAL_ENOMEM );

   vector->length = length;
   if (length==0) vector->data = NULL;
   else {
      XLAL_CHECK_NULL( (vector->data = XLALMalloc( length*sizeof(*vector->data) )) != NULL, XLAL_ENOMEM );
      for (UINT4 ii=0; ii<length; ii++) resetUpperLimitStruct(&(vector->data[ii]));
   }

   return vector;

} // createUpperLimitVector()


/**
 * Resize an UpperLimitVector
 * \param [in,out] vector Pointer to UpperLimitVector to resize
 * \param [in]     length New length of the vector
 * \return Pointer to reallocated UpperLimitVector
 */
UpperLimitVector * resizeUpperLimitVector(UpperLimitVector *vector, const UINT4 length)
{

   if (vector==NULL) return createUpperLimitVector(length);
   if (length==0) {
      destroyUpperLimitVector(vector);
      return NULL;
   }

   UINT4 oldlength = vector->length;

   XLAL_CHECK_NULL( (vector->data = XLALRealloc(vector->data, length*sizeof(*vector->data))) != NULL, XLAL_ENOMEM );
   vector->length = length;
   for (UINT4 ii=oldlength; ii<length; ii++) resetUpperLimitStruct(&(vector->data[ii]));

   return vector;

} // resizeUpperLimitVector()


/**
 * Free an UpperLimitVector
 * \param [in] vector Pointer to an UpperLimitVector
 */
void destroyUpperLimitVector(UpperLimitVector *vector)
{
   if (vector==NULL) return;
   if ((!vector->length || !vector->data) && (vector->length || vector->data)) XLAL_ERROR_VOID(XLAL_EINVAL);
   if (vector->data) {
      for (UINT4 ii=0; ii<vector->length; ii++) destroyUpperLimitStruct(&(vector->data[ii]));
      XLALFree((UpperLimit*)vector->data);
   }
   vector->data = NULL;
   XLALFree((UpperLimitVector*)vector);
   return;
} // destroyUpperLimitVector()


/**
 * Reset an UpperLimitStruct
 * \param [in] ul Pointer to an UpperLimit structure
 */
void resetUpperLimitStruct(UpperLimit *ul)
{
   ul->fsig = NULL;
   ul->period = NULL;
   ul->moddepth = NULL;
   ul->ULval = NULL;
   ul->effSNRval = NULL;
} // resetUpperLimitStruct()


/**
 * Free an UpperLimit structure
 * \param [in] ul Pointer to an UpperLimit structure
 */
void destroyUpperLimitStruct(UpperLimit *ul)
{
   if (ul->fsig) XLALDestroyREAL8Vector(ul->fsig);
   if (ul->period) XLALDestroyREAL8Vector(ul->period);
   if (ul->moddepth) XLALDestroyREAL8Vector(ul->moddepth);
   if (ul->ULval) XLALDestroyREAL8Vector(ul->ULval);
   if (ul->effSNRval) XLALDestroyREAL8Vector(ul->effSNRval);
} // destroyUpperLimitStruct()


/**
 * Determine the 95% confidence level upper limit at a particular sky location from the loudest IHS value
 * \param [out] ul        Pointer to an UpperLimit struct
 * \param [in]  params    Pointer to UserInput_t
 * \param [in]  ffdata    Pointer to ffdataStruct
 * \param [in]  ihsmaxima Pointer to an ihsMaximaStruct
 * \param [in]  ihsfar    Pointer to an ihsfarStruct
 * \param [in]  fbinavgs  Pointer to a REAL4VectorAligned of the 2nd FFT background powers
 * \return Status value
 */
INT4 skypoint95UL(UpperLimit *ul, const UserInput_t *params, const ffdataStruct *ffdata, const ihsMaximaStruct *ihsmaxima, const ihsfarStruct *ihsfar, const REAL4VectorAligned *fbinavgs)
{

   XLAL_CHECK( ul != NULL && params != NULL && ffdata != NULL && ihsmaxima != NULL && ihsfar!= NULL && fbinavgs != NULL, XLAL_EINVAL );

   BOOLEAN ULdetermined = 0;

   INT4 minrows = (INT4)round(2.0*params->dfmin*params->Tsft)+1;

   //Allocate vectors
   XLAL_CHECK( (ul->fsig = XLALCreateREAL8Vector((ihsmaxima->rows-minrows)+1)) != NULL, XLAL_EFUNC );
   XLAL_CHECK( (ul->period = XLALCreateREAL8Vector((ihsmaxima->rows-minrows)+1)) != NULL, XLAL_EFUNC );
   XLAL_CHECK( (ul->moddepth = XLALCreateREAL8Vector((ihsmaxima->rows-minrows)+1)) != NULL, XLAL_EFUNC );
   XLAL_CHECK( (ul->ULval = XLALCreateREAL8Vector((ihsmaxima->rows-minrows)+1)) != NULL, XLAL_EFUNC );
   XLAL_CHECK( (ul->effSNRval = XLALCreateREAL8Vector((ihsmaxima->rows-minrows)+1)) != NULL, XLAL_EFUNC );

   //Initialize solver
   const gsl_root_fsolver_type *T = gsl_root_fsolver_brent;
   gsl_root_fsolver *s = NULL;
   XLAL_CHECK( (s = gsl_root_fsolver_alloc (T)) != NULL, XLAL_EFUNC );
   gsl_function F;
   switch (params->ULsolver) {
      case 1:
         F.function = &gsl_ncx2cdf_withouttinyprob_solver;           //double precision, without the extremely tiny probability part
         break;
      case 2:
         F.function = &gsl_ncx2cdf_float_solver;   //single precision
         break;
      case 3:
         F.function = &gsl_ncx2cdf_solver;         //double precision
         break;
      case 4:
         F.function = &ncx2cdf_float_withouttinyprob_withmatlabchi2cdf_solver;   //single precision, w/ Matlab-based chi2cdf function
         break;
      case 5:
         F.function = &ncx2cdf_withouttinyprob_withmatlabchi2cdf_solver;         //double precision, w/ Matlab-based chi2cdf function
         break;
      default:
         F.function = &gsl_ncx2cdf_float_withouttinyprob_solver;     //single precision, without the extremely tiny probability part
         break;
   }
   struct ncx2cdf_solver_params pars;

   //loop over modulation depths
   for (INT4 ii=minrows; ii<=ihsmaxima->rows; ii++) {
      REAL8 loudestoutlier = 0.0, loudestoutlierminusnoise = 0.0, loudestoutliernoise = 0.0;
      INT4 jjbinofloudestoutlier = 0, locationofloudestoutlier = -1;
      INT4 startpositioninmaximavector = (ii-2)*ffdata->numfbins - ((ii-1)*(ii-1)-(ii-1))/2;
      REAL8 moddepth = 0.5*(ii-1.0)/params->Tsft;                             //"Signal" modulation depth

      //loop over frequency bins
      for (INT4 jj=0; jj<ffdata->numfbins-(ii-1); jj++) {
         INT4 locationinmaximavector = startpositioninmaximavector + jj;      //Current location in IHS maxima vector
         REAL8 noise = ihsfar->expectedIHSVector->data[ihsmaxima->locations->data[locationinmaximavector] - 5];  //Expected noise

         //Sum across multiple frequency bins scaling noise each time with average noise floor
         REAL8 totalnoise = 0.0;
         for (INT4 kk=0; kk<ii; kk++) totalnoise += fbinavgs->data[jj+kk];
         totalnoise = noise*totalnoise;

         REAL8 ihsminusnoise = ihsmaxima->maxima->data[locationinmaximavector] - totalnoise;    //IHS value minus noise

         REAL8 fsig = params->fmin - params->dfmax + (0.5*(ii-1.0) + jj - 6.0)/params->Tsft;        //"Signal" frequency

         if (ihsminusnoise>loudestoutlierminusnoise &&
             (fsig>=params->ULfmin && fsig<params->ULfmin+params->ULfspan) &&
             (moddepth>=params->ULminimumDeltaf && moddepth<=params->ULmaximumDeltaf)) {
            loudestoutlier = ihsmaxima->maxima->data[locationinmaximavector];
            loudestoutliernoise = totalnoise;
            loudestoutlierminusnoise = ihsminusnoise;
            locationofloudestoutlier = ihsmaxima->locations->data[locationinmaximavector];
            jjbinofloudestoutlier = jj;
         }
      } /* for jj < ffdata->numfbins-(ii-1) */

      if (locationofloudestoutlier!=-1) {
         //We do a root finding algorithm to find the delta value required so that only 5% of a non-central chi-square
         //distribution lies below the maximum value.
         REAL8 initialguess = ncx2inv_float(0.95, 2.0*loudestoutliernoise, 2.0*loudestoutlierminusnoise);
         XLAL_CHECK( xlalErrno == 0, XLAL_EFUNC );

         REAL8 lo = 0.001*initialguess, hi = 10.0*initialguess;
         pars.val = 2.0*loudestoutlier;
         pars.dof = 2.0*loudestoutliernoise;
         pars.ULpercent = 0.95;
         F.params = &pars;
         XLAL_CHECK( gsl_root_fsolver_set(s, &F, lo, hi) == GSL_SUCCESS, XLAL_EFUNC );

         INT4 status = GSL_CONTINUE;
         INT4 max_iter = 100;
         REAL8 root = 0.0;
         INT4 jj = 0;
         while (status==GSL_CONTINUE && jj<max_iter) {
            jj++;
            status = gsl_root_fsolver_iterate(s);
            XLAL_CHECK( status == GSL_CONTINUE || status == GSL_SUCCESS, XLAL_EFUNC, "gsl_root_fsolver_iterate() failed with code %d\n", status );
            root = gsl_root_fsolver_root(s);
            lo = gsl_root_fsolver_x_lower(s);
            hi = gsl_root_fsolver_x_upper(s);
            status = gsl_root_test_interval(lo, hi, 0.0, 0.001);
            XLAL_CHECK( status == GSL_CONTINUE || status == GSL_SUCCESS, XLAL_EFUNC, "gsl_root_test_interval() failed with code %d\n", status );
         } /* while status==GSL_CONTINUE and jj<max_iter */
         XLAL_CHECK( status == GSL_SUCCESS, XLAL_EFUNC, "Root finding iteration (%d/%d) failed with code %d\n", jj, max_iter, status );

         //Convert the root value to an h0 value
         REAL8 h0 = ihs2h0(root, params);

         //Store values in the upper limit struct
         ul->fsig->data[ii-minrows] = params->fmin - params->dfmax + (0.5*(ii-1.0) + jjbinofloudestoutlier - 6.0)/params->Tsft;
         ul->period->data[ii-minrows] = params->Tobs/locationofloudestoutlier;
         ul->moddepth->data[ii-minrows] = 0.5*(ii-1.0)/params->Tsft;
         ul->ULval->data[ii-minrows] = h0;
         ul->effSNRval->data[ii-minrows] = unitGaussianSNR(root, pars.dof);
         ULdetermined++;
      } // if locationofloudestoutlier != -1
   } // for ii=minrows --> maximum rows

   //Signal an error if we didn't find something above the noise level
   XLAL_CHECK( ULdetermined != 0, XLAL_EFUNC, "Failed to reach a louder outlier minus noise greater than 0\n" );

   gsl_root_fsolver_free(s);

   return XLAL_SUCCESS;

}


//The non-central chi-square CDF solver used in the GSL root finding algorithm
//Double precision
REAL8 gsl_ncx2cdf_solver(const REAL8 x, void *p)
{

   struct ncx2cdf_solver_params *params = (struct ncx2cdf_solver_params*)p;
   REAL8 val = ncx2cdf(params->val, params->dof, x);
   XLAL_CHECK_REAL8( xlalErrno == 0, XLAL_EFUNC );
   return val - (1.0-params->ULpercent);

}

//The non-central chi-square CDF solver used in the GSL root finding algorithm
//Float precision (although output is in double precision for GSL)
REAL8 gsl_ncx2cdf_float_solver(const REAL8 x, void *p)
{

   struct ncx2cdf_solver_params *params = (struct ncx2cdf_solver_params*)p;
   REAL4 val = ncx2cdf_float((REAL4)params->val, (REAL4)params->dof, (REAL4)x);
   XLAL_CHECK_REAL8( xlalErrno == 0, XLAL_EFUNC );
   return (REAL8)val - (1.0-params->ULpercent);

}

//The non-central chi-square CDF solver used in the GSL root finding algorithm
//Double precision, without the tiny probability
REAL8 gsl_ncx2cdf_withouttinyprob_solver(const REAL8 x, void *p)
{

   struct ncx2cdf_solver_params *params = (struct ncx2cdf_solver_params*)p;
   REAL8 val = ncx2cdf_withouttinyprob(params->val, params->dof, x);
   XLAL_CHECK_REAL8( xlalErrno == 0, XLAL_EFUNC );
   return val - (1.0-params->ULpercent);

}

//The non-central chi-square CDF solver used in the GSL root finding algorithm
//Float precision (although output is in double precision for GSL), without the tiny probability
REAL8 gsl_ncx2cdf_float_withouttinyprob_solver(const REAL8 x, void *p)
{

   struct ncx2cdf_solver_params *params = (struct ncx2cdf_solver_params*)p;
   REAL4 val = ncx2cdf_float_withouttinyprob((REAL4)params->val, (REAL4)params->dof, (REAL4)x);
   XLAL_CHECK_REAL8( xlalErrno == 0, XLAL_EFUNC );
   return (REAL8)val - (1.0-params->ULpercent);

}

//The non-central chi-square CDF solver used in the GSL root finding algorithm, using a Matlab-based chi2cdf function
//Double precision, without the tiny probability
REAL8 ncx2cdf_withouttinyprob_withmatlabchi2cdf_solver(const REAL8 x, void *p)
{

   struct ncx2cdf_solver_params *params = (struct ncx2cdf_solver_params*)p;
   REAL8 val = ncx2cdf_withouttinyprob_withmatlabchi2cdf(params->val, params->dof, x);
   XLAL_CHECK_REAL8( xlalErrno == 0, XLAL_EFUNC );
   return val - (1.0-params->ULpercent);

}

//The non-central chi-square CDF solver used in the GSL root finding algorithm, using a Matlab-based chi2cdf function
//Float precision (although output is in double precision for GSL), without the tiny probability
REAL8 ncx2cdf_float_withouttinyprob_withmatlabchi2cdf_solver(REAL8 x, void *p)
{

   struct ncx2cdf_solver_params *params = (struct ncx2cdf_solver_params*)p;
   REAL4 val = ncx2cdf_float_withouttinyprob_withmatlabchi2cdf((REAL4)params->val, (REAL4)params->dof, (REAL4)x);
   XLAL_CHECK_REAL8( xlalErrno == 0, XLAL_EFUNC );
   return (REAL8)val - (1.0-params->ULpercent);

}


/**
 * Output the highest upper limit to a file unless printAllULvalues==1 in which case, all UL values are printed to a file
 * \param [in] outputfile       String of the filename
 * \param [in] ul               UpperLimit structure to print to file
 * \param [in] printAllULvalues Option flag to print all UL values from a sky location (1) or only the largest (0)
 * \return Status value
 */
INT4 outputUpperLimitToFile(const CHAR *outputfile, const UpperLimit ul, const BOOLEAN printAllULvalues)
{
   XLAL_CHECK( outputfile!=NULL, XLAL_EINVAL );

   FILE *ULFILE = NULL;
   XLAL_CHECK( (ULFILE = fopen(outputfile, "a")) != NULL, XLAL_EIO, "Couldn't fopen file %s to output upper limits\n", outputfile );

   REAL8 highesth0 = 0.0, snr = 0.0, fsig = 0.0, period = 0.0, moddepth = 0.0;
   for (UINT4 ii=0; ii<ul.moddepth->length; ii++) {
      if (printAllULvalues==1) {
         fprintf(ULFILE, "%.6f %.6f %.6g %.6f %.6f %.6f %.6f %.6g\n", ul.alpha, ul.delta, ul.ULval->data[ii], ul.effSNRval->data[ii], ul.fsig->data[ii], ul.period->data[ii], ul.moddepth->data[ii], ul.normalization);
      } else if (printAllULvalues==0 && ul.ULval->data[ii]>highesth0) {
         highesth0 = ul.ULval->data[ii];
         snr = ul.effSNRval->data[ii];
         fsig = ul.fsig->data[ii];
         period = ul.period->data[ii];
         moddepth = ul.moddepth->data[ii];
      }
   }
   if (printAllULvalues==0) {
      fprintf(ULFILE, "%.6f %.6f %.6g %.6f %.6f %.6f %.6f %.6g\n", ul.alpha, ul.delta, highesth0, snr, fsig, period, moddepth, ul.normalization);
   }

   fclose(ULFILE);

   return XLAL_SUCCESS;

}

/*
*  Copyright (C) 2010 Evan Goetz
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
#include "IHS.h"
#include "candidates.h"

//////////////////////////////////////////////////////////////
// Create vectors for IHS maxima struct  -- done
ihsMaximaStruct * new_ihsMaxima(ffdataStruct *ffdata, INT4 columns)
{

   ihsMaximaStruct *ihsmaxima = (ihsMaximaStruct*)XLALMalloc(sizeof(ihsMaximaStruct));
   
   ihsmaxima->maxima = XLALCreateREAL4Vector(ffdata->f->length*(UINT4)columns);
   ihsmaxima->locations = XLALCreateINT4Vector(ffdata->f->length);
   ihsmaxima->columns = columns;
   
   return ihsmaxima;

}

//////////////////////////////////////////////////////////////
// Destroy vectors for IHS maxima struct  -- done
void free_ihsMaxima(ihsMaximaStruct *data)
{

   XLALDestroyREAL4Vector(data->maxima);
   XLALDestroyINT4Vector(data->locations);
   XLALFree((ihsMaximaStruct*)data);

}


//////////////////////////////////////////////////////////////
// Run the IHS algorithm  -- done
void runIHS(ihsMaximaStruct *out, ffdataStruct *in, INT4 columns)
{

   INT4 ii, jj;
   
   REAL4Vector *column = XLALCreateREAL4Vector(in->fpr->length);
   REAL4Vector *ihss = XLALCreateREAL4Vector(in->f->length);
   ihsVals *ihsvals = new_ihsVals();
   
   //Loop through the columns, 1 frequency at a time
   for (ii=0; ii<(INT4)in->f->length; ii++) {
   
      //For each column, populate it with the data for that frequency bin
      for (jj=0; jj<(INT4)column->length; jj++) column->data[jj] = in->ffdata->data[ii*in->fpr->length + jj];
      
      //Run the IHS algorithm on the column
      incHarmSum(ihsvals, column);
      
      //Temporarily save the IHS value
      ihss->data[ii] = ihsvals->ihs;
      
      //Save the IHS maximum location value for each column
      out->locations->data[ii] = ihsvals->loc;
   }
   
   //Save the maxima for all the column sums
   out->maxima = ihsSums(ihss, columns);
   
   //Save the column widths
   out->columns = columns;
   
   //Destroy variables
   XLALDestroyREAL4Vector(column);
   XLALDestroyREAL4Vector(ihss);
   free_ihsVals(ihsvals);

}


//////////////////////////////////////////////////////////////
// Allocate memory for ihsVals struct  -- done
ihsVals * new_ihsVals(void)
{

   ihsVals *ihsvals = (ihsVals*)XLALMalloc(sizeof(ihsVals));

   return ihsvals;

}

//////////////////////////////////////////////////////////////
// Destroy ihsVals struct  -- done
void free_ihsVals(ihsVals *ihsvals)
{

   XLALFree((ihsVals*)ihsvals);

}


//////////////////////////////////////////////////////////////
// Compute the IHS sum  -- Done
void incHarmSum(ihsVals *out, REAL4Vector *vector)
{
   
   INT4 ii, jj, kk, ll, mm, loc;
   REAL4 ihs;
   
   //Load the stretched spectra
   jj = -1;
   kk = -1;
   ll = -1;
   mm = -1;
   ihs = 0.0;
   loc = 0;
   for (ii=0; ii<(INT4)vector->length; ii++) {
      if (ii % 2 == 0) jj++;
      if (ii % 3 == 0) kk++;
      if (ii % 4 == 0) ll++;
      if (ii % 5 == 0) mm++;
      REAL4 sum = vector->data[ii] + 0.5*vector->data[jj] + vector->data[kk]/3.0 + 0.25*vector->data[ll] + 0.2*vector->data[mm];
      if (sum > ihs && ii > 3) {
         ihs = sum;
         loc = ii;
      }
   }
   
   //Load the outputs into the structure
   out->ihs = ihs;
   out->loc = loc;

}



//////////////////////////////////////////////////////////////
// Allocate memory for ihsfarStruct struct  -- done
ihsfarStruct * new_ihsfarStruct(INT4 columns)
{

   ihsfarStruct *ihsfarstruct = (ihsfarStruct*)XLALMalloc(sizeof(ihsfarStruct));
   
   ihsfarstruct->ihsfar = XLALCreateREAL4Vector((UINT4)columns);
   ihsfarstruct->ihsdistMean = XLALCreateREAL4Vector((UINT4)columns);
   ihsfarstruct->ihsdistSigma = XLALCreateREAL4Vector((UINT4)columns);

   return ihsfarstruct;

}

//////////////////////////////////////////////////////////////
// Destroy ihsfarStruct struct  -- done
void free_ihsfarStruct(ihsfarStruct *ihsfarstruct)
{

   XLALDestroyREAL4Vector(ihsfarstruct->ihsfar);
   XLALDestroyREAL4Vector(ihsfarstruct->ihsdistMean);
   XLALDestroyREAL4Vector(ihsfarstruct->ihsdistSigma);
   XLALFree((ihsfarStruct*)ihsfarstruct);

}


//////////////////////////////////////////////////////////////
// Compute the IHS FAR for a sum of a number of columns  -- done
void genIhsFar(ihsfarStruct *out, ffdataStruct *ffdata, INT4 columns, REAL4 threshold)
{
   
   INT4 ii, jj, kk, loc, length;
   REAL4Vector *noise = NULL;
   REAL4 max = 0.0;
   
   length = ffdata->fpr->length;
   
   INT4 trials = (INT4)10000*roundf(0.01/threshold);    //Number of trials to determine FAR value
   trials += columns;
   REAL4Vector *ihss = XLALCreateREAL4Vector((UINT4)trials);
   
   //Initialize random number generator
   gsl_rng *rng = gsl_rng_alloc(gsl_rng_mt19937);
   srand(time(NULL));
   UINT8 randseed = rand();
   gsl_rng_set(rng, randseed);
   
   ihsVals *ihsvals = new_ihsVals();
   
   //Determine IHS values for the number of trials
   noise = XLALCreateREAL4Vector((UINT4)length);
   for (ii=0; ii<trials; ii++) {
      //Make exponential noise
      for (jj=0; jj<length; jj++) noise->data[jj] = expRandNum(1.0, rng);
      
      //Compute IHS value on exponential noise
      incHarmSum(ihsvals, noise);
      ihss->data[ii] = ihsvals->ihs;
      
   }
   XLALDestroyREAL4Vector(noise);
   
   //Calculate the IHS sum values for the IHS trials
   REAL4Vector *ihssumvals = ihsSums(ihss, columns);
   
   //Now determine distribution values and FAR for the different IHS sum values for each set of columns
   REAL4Vector *tempihsvals = NULL;
   for (ii=0; ii<columns; ii++) {
      //Temporary vector to hold the trial values of IHS column sums
      tempihsvals = XLALCreateREAL4Vector((UINT4)(trials-ii));
      for (jj=0; jj<(INT4)tempihsvals->length; jj++) {
         if (ii==0) tempihsvals->data[jj] = ihssumvals->data[jj];
         else tempihsvals->data[jj] = ihssumvals->data[ii*trials-(ii-1)+jj];
      }
      
      //Mean and sigma of the various trials
      out->ihsdistMean->data[ii] = calcMean(tempihsvals);
      out->ihsdistSigma->data[ii] = calcStddev(tempihsvals);
      
      //Find the threshold value to get the FAR
      for (jj=0; jj<(INT4)roundf((trials-ii)*threshold)+1; jj++) {
         max = 0.0;
         loc = 0;
         for (kk=0; kk<(INT4)tempihsvals->length; kk++) {
            if (tempihsvals->data[kk]>max) {
               max = tempihsvals->data[kk];
               loc = kk;
            }
         }
         tempihsvals->data[loc] = 0.0;
      }
      
      //FAR value
      out->ihsfar->data[ii] = max;
      
      //Reset temporary vector
      XLALDestroyREAL4Vector(tempihsvals);
      tempihsvals = NULL;
   }
   
   
   //Destroy variables
   XLALDestroyREAL4Vector(ihssumvals);
   XLALDestroyREAL4Vector(ihss);
   free_ihsVals(ihsvals);
   gsl_rng_free(rng);
   

}



//////////////////////////////////////////////////////////////
// Compute the IHS sums for a number of columns  -- done
REAL4Vector * ihsSums(REAL4Vector *ihss, INT4 cols)
{
   
   INT4 ii, jj, locToAdd, locInMaximaVector;
   INT4 startPosition = 0;
   
   UINT4 numToRemove = 0;
   for (ii=1; ii<cols; ii++) numToRemove += (UINT4)(ii-1);
   
   //Initialize maxima vector and set all elements to 0
   REAL4Vector *maxima = XLALCreateREAL4Vector((UINT4)(ihss->length * cols)-numToRemove);
   for (ii=0; ii<(INT4)maxima->length; ii++) maxima->data[ii] = 0;
   
   //Start with the vector of single column IHS values
   for (ii=0; ii<(INT4)ihss->length; ii++) maxima->data[ii] = ihss->data[ii];
   
   locInMaximaVector = ihss->length;
   for (ii=1; ii<cols; ii++) {
      locToAdd = ii;
      for (jj=0; jj<(INT4)ihss->length-ii; jj++) {
         if (jj==0) startPosition = locInMaximaVector-((INT4)ihss->length-(ii-1));
         maxima->data[locInMaximaVector] = maxima->data[startPosition+jj] + maxima->data[locToAdd];
         locToAdd++;
         locInMaximaVector++;
      }
   }
   
   return maxima;

}


//////////////////////////////////////////////////////////////
// Calculate the IHS FOM for a number of columns  -- done
REAL4 ihsFOM(REAL4Vector *ihss, INT4Vector *locs, REAL4Vector *expect)
{

   INT4 ii, maxsnrloc;
   REAL4 maxsnr, fom;
   
   //Create normalized SNR of IHS values
   REAL4Vector *snrs = XLALCreateREAL4Vector(ihss->length);
   for (ii=0; ii<(INT4)snrs->length; ii++) snrs->data[ii] = ihss->data[ii]/expect->data[ii];
   
   //Find which pair has the best combined SNR (RMS) and the location
   maxsnr = sqrt(snrs->data[0]*snrs->data[0] + snrs->data[snrs->length-1]*snrs->data[snrs->length-1]);
   maxsnrloc = 0;
   for (ii=1; ii<(INT4)floor(snrs->length*0.5); ii++) {
      if (sqrt(snrs->data[ii]*snrs->data[ii] + 
            snrs->data[snrs->length-ii-1]*snrs->data[snrs->length-ii-1])>maxsnr) {
         maxsnr = sqrt(snrs->data[ii]*snrs->data[ii] +
            snrs->data[snrs->length-ii-1]*snrs->data[snrs->length-ii-1]);
         maxsnrloc = ii;
      }
   }
   
   //For the highest SNR pair, compute the FOM
   fom = 6.0*(locs->data[maxsnrloc] - locs->data[locs->length-maxsnrloc-1])*
         (locs->data[maxsnrloc] - locs->data[locs->length-maxsnrloc-1]);
   
   //Destroy used variables
   XLALDestroyREAL4Vector(snrs);
   
   return fom;

}


//////////////////////////////////////////////////////////////
// Calculate a guess for the location of the brightest pixels
REAL4 ihsLoc(REAL4Vector *ihss, INT4Vector *locs, REAL4Vector *expect)
{

   INT4 ii, maxsnrloc;
   REAL4 maxsnr;
   
   //Create normalized SNR of IHS values
   REAL4Vector *snrs = XLALCreateREAL4Vector(ihss->length);
   for (ii=0; ii<(INT4)snrs->length; ii++) snrs->data[ii] = ihss->data[ii]/expect->data[ii];
   
   //Find which pair has the best combined SNR (RMS) and the location
   maxsnr = sqrt(snrs->data[0]*snrs->data[0] + snrs->data[snrs->length-1]*snrs->data[snrs->length-1]);
   maxsnrloc = 0;
   for (ii=1; ii<(INT4)floor(snrs->length*0.5); ii++) {
      if (sqrt(snrs->data[ii]*snrs->data[ii] + 
            snrs->data[snrs->length-ii-1]*snrs->data[snrs->length-ii-1])>maxsnr) {
         maxsnr = sqrt(snrs->data[ii]*snrs->data[ii] +
            snrs->data[snrs->length-ii-1]*snrs->data[snrs->length-ii-1]);
         maxsnrloc = ii;
      }
   }
   
   //For the highest SNR pair, compute the location
   REAL4 location = 0.5*(locs->data[maxsnrloc] + locs->data[locs->length-maxsnrloc-1]);
   
   //Destroy used variables
   XLALDestroyREAL4Vector(snrs);
   
   return location;

}



void findIHScandidates(candidate *candlist[], INT4 *numofcandidates, ihsfarStruct *ihsfarstruct, REAL4Vector *aveFFnoise, inputParamsStruct *inputParams, ffdataStruct *ffdata, ihsMaximaStruct *ihsmaxima, REAL4 ra, REAL4 dec)
{
   
   INT4 ii, jj, kk, ll;
   REAL4 fsig, per0, B;
   REAL4 ihsfomfar = 6.0;
   
   REAL4Vector *ihss, *noiseinrange, *ihsexpect;
   INT4Vector *locs;
   ffdataStruct *ffdata_temp;
   ihsMaximaStruct *ihsmaxima_temp;
   //Check the IHS values against the FAR, checking for >1 column widths
   for (ii=1; ii<(INT4)ihsfarstruct->ihsfar->length; ii++) {
      ihss = XLALCreateREAL4Vector((UINT4)(ii+1));
      locs = XLALCreateINT4Vector((UINT4)(ii+1));
      noiseinrange = XLALCreateREAL4Vector((UINT4)(ii+1));
      ihsexpect = XLALCreateREAL4Vector((UINT4)(ii+1));
      for (jj=0; jj<(INT4)ffdata->f->length-ii; jj++) {
      
         //Noise in the range of the columns, mean and rms values for IHS
         for (kk=0; kk<=ii; kk++) {
            noiseinrange->data[kk] = aveFFnoise->data[jj + kk];
            ihsexpect->data[kk] = aveFFnoise->data[jj + kk]*ihsfarstruct->ihsdistMean->data[ii];
         }
         REAL4 meanNoise = calcMean(noiseinrange);
         REAL4 rmsNoise = calcRms(noiseinrange);
         
         //Check the IHS sum against the FAR (scaling FAR with mean of the noise in the range of columns)
         if (ihsmaxima->maxima->data[ii*ihsmaxima->locations->length + jj] > ihsfarstruct->ihsfar->data[ii]*meanNoise) {
         
            //Load temporary vectors for determining the FOM
            for (kk=0; kk<=ii; kk++) {
               ihss->data[kk] = ihsmaxima->maxima->data[jj + kk];
               locs->data[kk] = ihsmaxima->locations->data[jj + kk];
            }
            
            //Compute the IHS FOM
            REAL4 fom = ihsFOM(ihss, locs, ihsexpect);
         
            //Check the IHS FOM against the FAR
            if  (fom<=ihsfomfar) {
               
               //Candidate frequency
               fsig = inputParams->fmin + (0.5*ii + jj)/inputParams->Tcoh;
               
               //Allocate memory for temporary ffdata structure
               ffdata_temp = new_ffdata(inputParams, 1);
            
               //Create vector to hold ffdata with second frequency >5/Tobs to make sure
               //candidates have >5 periods in the observation time
               for (kk=0; kk<(INT4)ffdata_temp->f->length; kk++) {
                  for (ll=5; ll<(INT4)ffdata->fpr->length; ll++) ffdata_temp->ffdata->data[kk*(ffdata->fpr->length-5) + (ll-5)] = ffdata->ffdata->data[kk*ffdata->fpr->length + ll];
               }
            
               //Run the IHS maxima algorithm again, so we need memory allocated
               ihsmaxima_temp = new_ihsMaxima(ffdata_temp, ii);
               runIHS(ihsmaxima_temp, ffdata_temp, ii);
               
               //Determine locations. Load temporary vectors of IHS values and locations.
               //Then compute the best location
               REAL4 loc = 0.0;
               for (kk=0; kk<=ii; kk++) {
                  ihss->data[kk] = ihsmaxima_temp->maxima->data[jj + kk];
                  locs->data[kk] = ihsmaxima_temp->locations->data[jj + kk] + 5;
               }
               loc = ihsLoc(ihss, locs, ihsexpect);
               
               //Candidate modulation depth
               B = 0.5*ii/inputParams->Tcoh;
               
               //If the location is not zero, then let's load it as a candidate!
               if (loc!=0.0 && inputParams->Tobs/loc>=2.0*3600.0) {
                  per0 = inputParams->Tobs/loc;
                  
                  REAL4 ihs_sum = ihsmaxima->maxima->data[ii*ihsmaxima->locations->length + jj];
                  //REAL4 ihsSnr = ihs_sum/sqrt(ii+1)/ihsStd;
                  REAL4 ihsSnr = (ihs_sum - meanNoise*ihsfarstruct->ihsdistMean->data[ii])/(rmsNoise*ihsfarstruct->ihsdistSigma->data[ii]);
                  candlist[(*numofcandidates)] = new_candidate();
                  loadCandidateData(candlist[(*numofcandidates)], fsig, per0, B, ra, dec, ihs_sum, ihsSnr);
                  (*numofcandidates)++;
               }
               
               //Destroy unneeded variables
               free_ffdata(ffdata_temp);
               free_ihsMaxima(ihsmaxima_temp);
               ihsmaxima_temp = NULL;
               ffdata_temp = NULL;
            }
         }
      }
      XLALDestroyREAL4Vector(ihss);
      XLALDestroyREAL4Vector(ihsexpect);
      XLALDestroyREAL4Vector(noiseinrange);
      XLALDestroyINT4Vector(locs);
      ihss = NULL;
      locs = NULL;
      ihsexpect = NULL;
      noiseinrange = NULL;
   }
   
}





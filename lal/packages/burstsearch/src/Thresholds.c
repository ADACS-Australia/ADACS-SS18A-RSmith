/*----------------------------------------------------------------------- 
 * 
 * File Name: Thresholds.c 
 * 
 * Author: Eanna Flanagan
 * 
 * Revision: $Id$
 * 
 *----------------------------------------------------------------------- 
 * 
 * NAME 
 * Threshold
 * 
 * SYNOPSIS 
 * 
 * DESCRIPTION 
 *  A set of functions to compute cumulative distribution functions
 *  and thresholds for the chi-squared and non-central chi-squared
 *  distributions.
 * 
 * 
 * DIAGNOSTICS 
 *
 * CALLS
 * 
 * NOTES
 * 
 *-----------------------------------------------------------------------
 */


#include <lal/LALRCSID.h>


NRCSID (THRESHOLDSC, "$Id$");

#include <stdlib.h>
#include <math.h>
#include <lal/LALStdlib.h>
#include <lal/LALConstants.h>
#include <lal/Thresholds.h>
#include <lal/FindRoot.h>




/*
 *
 * Private functions not accessible outside this file Thresholds.c 
 *
 */


static REAL8 logGamma(REAL8 z)
{
  /* 
   *  natural log of Gamma function
   */

  INT4 j;
  REAL8 lngamma;
  REAL8 tmp;
  const REAL8 ff[9]={0.9999999999995183, 676.5203681218835,
              -1259.139216722289, 771.3234287757674,
              -176.6150291498386, 12.50734324009056,
              -0.1385710331296526, 0.9934937113930748e-05,
		    0.1659470187408462e-06};
  const REAL8 lnsqrt2pi=0.9189385332046727;

  lngamma = 0.0;
  tmp = z + 7.0;
  for(j=8;j>=1;j--)
    {
      lngamma += ff[j]/tmp;
      tmp -= 1.0;
    }
  lngamma += ff[0];
  lngamma = log(lngamma) + lnsqrt2pi - (z+6.5) + (z-0.5)*log(z+6.5);
  return lngamma;
}



static REAL8 Factorial(INT4 n)
{
  /* returns n!  */
  
  REAL8 temp = 1.0;
  INT4 i;
  if(n==0) return(1.0);
  for(i=1; i<=n;i++) temp *= (REAL8)(i);
  return(temp);
}


static void
ChisqCdf1 (
    LALStatus                *status,
    REAL8                 *prob,
    REAL8                 lnchi2,
    void                  *params
    )
{  
  /* 
   *  This is just the LALChisqCdf() function plus an added constant. 
   *  Its used in LALChi2Threshold() in the call to DFindRoot().
   *
   */



  ChisqCdfIn         localparams;
  Chi2ThresholdIn    *input;

  INITSTATUS (status, "ChisqCdf1", THRESHOLDSC);
  ATTATCHSTATUSPTR (status);





  ASSERT (prob, status, THRESHOLDS_ENULLP, THRESHOLDS_MSGENULLP);
  ASSERT (params, status, THRESHOLDS_ENULLP, THRESHOLDS_MSGENULLP);

  /* 
   * 
   * params is actually a pointer to a structure Chi2ThresholdIn
   * It passed to ChisqCdf1 as a pointer to void so that ChisqCdf1 will
   * be a REAL8LALFunction and be passable to DFindRoot();
   * see header file FindRoot.h
   *
   */
  input = (Chi2ThresholdIn *)params;

  /* set up parameters to be passed to LALChisqCdf()  */
  localparams.dof = input->dof;
  localparams.chi2 = exp(lnchi2);


  LALOneMinusChisqCdf( status->statusPtr, prob, &localparams );
  /* we can ignore the error where probability is 1 or 0 */
  if(status->statusPtr->statusCode==THRESHOLDS_ERANGE)
    {
      status->statusPtr->statusCode=0;
    }
  /* check for any other errors */
  CHECKSTATUSPTR (status);


  *prob -= (input->falseAlarm);
  
  DETATCHSTATUSPTR (status);
  RETURN(status);
}



static void
NoncChisqCdf1 (
    LALStatus                *status,
    REAL8                 *prob,
    REAL8                 lnrho,
    void                  *params
    )
{  
  /* 
   *  This is just the LALNoncChisqCdf() function plus an added constant. 
   *  Its used in LALRhoThreshold() in the call to DFindRoot().
   *
   */

  ChisqCdfIn         localparams;
  RhoThresholdIn     *input;

  INITSTATUS (status, "NoncChisqCdf1", THRESHOLDSC);
  ATTATCHSTATUSPTR (status);

  ASSERT (prob, status, THRESHOLDS_ENULLP, THRESHOLDS_MSGENULLP);
  ASSERT (params, status, THRESHOLDS_ENULLP, THRESHOLDS_MSGENULLP);

  /* 
   * 
   * params is actually a pointer to a structure RhoThresholdIn
   * It passed to NoncChisqCdf1 as a pointer to void so that NoncChisqCdf1 will
   * be a REAL8LALFunction and be passable to DFindRoot();
   * see header file FindRoot.h
   *
   */
  input = (RhoThresholdIn *)params;

  /* set up parameters to be passed to LALNoncChisqCdf()  */
  localparams.dof = input->dof;
  localparams.chi2 = input->chi2;
  localparams.nonCentral = exp(2.0*lnrho);

  LALNoncChisqCdf( status->statusPtr, prob, &localparams );
  /* we can ignore the error where probability is 1 or 0 */
  if(status->statusPtr->statusCode==THRESHOLDS_ERANGE)
    {
      status->statusPtr->statusCode=0;
    }
  /* check for any other errors */
  CHECKSTATUSPTR (status);

  *prob -= input->falseDismissal;
  
  DETATCHSTATUSPTR (status);
  RETURN(status);
}




/*
 *
 *  Public functions
 *
 */


void
LALChisqCdf (
    LALStatus        *status,
    REAL8         *prob,
    ChisqCdfIn    *input
    )
{  

  /*
   *  Cumulative Probability Distribution for Chi Squared distribution.
   *  
   *  returns probability that x_1^2 + .. x_dof^2 <= chi2, where
   *  x_1, ..  x_dof are independent Gaussians of zero mean and
   *  unit variance.
   *  The integral expression is
   *  prob = int_0^{chi^2/2} dx  x^((n/2)-1) e^(-x) / Gamma(n/2), where 
   *  n = dof = number of degrees of freedom.
   *  note chi2 = 2 * cal E, calE = variable used in paper
   *  (also dof = input->dof;   chi2 = input->chi2)
   *  
   *  The parameter input->nonCentral is not used by ChisqCdf.
   *
   */

  REAL8 temp1;
  REAL8 temp2;
  REAL8 ww;
  REAL8 a;
  REAL8 x;
  const INT2 maxloop=100;         /* maximum allowed number of iterations */
  const REAL8 small = 3.0e-7;
  const REAL8 small1 = 1.0e-30;

  INITSTATUS (status, "LALChisqCdf", THRESHOLDSC);

  /* check that arguments are reasonable */
  ASSERT (prob, status, THRESHOLDS_ENULLP, THRESHOLDS_MSGENULLP);
  ASSERT (input, status, THRESHOLDS_ENULLP, THRESHOLDS_MSGENULLP);

  /* Arguments chi2 and dof must be non-negative */
  x = input->chi2/2.0;
  a = input->dof/2.0;
  ASSERT ( (x >= 0.0) && (a > 0.0), status, THRESHOLDS_EPOSARG,
           THRESHOLDS_MSGEPOSARG);

  if (x < (a+1.0))
    {
      INT4 n;
      REAL8 sum;
      REAL8 del;
      REAL8 temp3;
      ww=logGamma(a);
      if (x <= 0.0)
	{
	  temp1=0.0;
	}
      else 
	{
	  temp3=a;
	  del=sum=1.0/a;
	  n=1;
	  do
	    {
	      ++temp3;
	      del *= x/temp3;
	      sum += del;
	      n++;
	    }
	  while( (n<=maxloop) && (fabs(del) > fabs(sum)*small) );

	  ASSERT(n < maxloop, status, THRESHOLDS_EMXIT, THRESHOLDS_MSGEMXIT);
	  temp1=sum*exp(-x+a*log(x)-(ww));
	}
      *prob=temp1;
    } 
  else 
    {
      INT4 i;
      REAL8 an,b,c,d,del,h;
      ww=logGamma(a);
      b=x+1.0-a;
      c=1.0/small1;
      d=1.0/b;
      h=d;
      i=1;
      do
	{
	  an = -i*(i-a);
	  b += 2.0;
	  d=an*d+b;
	  if (fabs(d) < small1) d=small1;
	  c=b+an/c;
	  if (fabs(c) < small1) c=small1;
	  d=1.0/d;
	  del=d*c;
	  h *= del;
	  if (fabs(del-1.0) < small) break;
	  i++;
	}
      while( (i<=maxloop) && (fabs(del-1.0) > small) );

      ASSERT(i < maxloop, status, THRESHOLDS_EMXIT, THRESHOLDS_MSGEMXIT);
      temp2=exp(-x+a*log(x)-(ww))*h;
      *prob= 1.0-temp2;
    }

  /* 
   *  check that final answer is a legal probability
   *  third test is necessary since there are some numbers x for 
   *  which (x>0.0) evaluates as TRUE but for which 1/x evaluates to inf
   */
  ASSERT( (*prob > 0.0) && (*prob < 1.0) && ( 1.0/(*prob) < LAL_REAL8_MAX ), 
          status, THRESHOLDS_ERANGE, THRESHOLDS_MSGERANGE);

  RETURN (status);
}






void
LALOneMinusChisqCdf (
    LALStatus        *status,
    REAL8         *prob,
    ChisqCdfIn    *input
    )
{  

  /*
   *  Cumulative Probability Distribution for Chi Squared distribution.
   *   Alternative version which is more accurate for large rho.
   *  
   *  returns probability that x_1^2 + .. x_dof^2 >= chi2, where
   *  x_1, ..  x_dof are independent Gaussians of zero mean and
   *  unit variance.
   *  The integral expression is
   *  prob = int_{chi^2/2}^\infty dx  x^((n/2)-1) e^(-x) / Gamma(n/2), where 
   *  n = dof = number of degrees of freedom.
   *  note chi2 = 2 * cal E, calE = variable used in paper
   *  (also dof = input->dof;   chi2 = input->chi2)
   *  
   *  The parameter input->nonCentral is not used by OneMinusChisqCdf.
   *
   *  This function's code is the same as
   *  the function LALChisqCdf() except for the very end.
   */

  REAL8 temp1;
  REAL8 temp2;
  REAL8 ww;
  REAL8 a;
  REAL8 x;
  const INT2 maxloop=100;         /* maximum allowed number of iterations */
  const REAL8 small = 3.0e-7;
  const REAL8 small1 = 1.0e-30;

  INITSTATUS (status, "LALOneMinusChisqCdf", THRESHOLDSC);

  /* check that arguments are reasonable */
  ASSERT (prob, status, THRESHOLDS_ENULLP, THRESHOLDS_MSGENULLP);
  ASSERT (input, status, THRESHOLDS_ENULLP, THRESHOLDS_MSGENULLP);

  /* Arguments chi2 and dof must be non-negative */
  x = input->chi2/2.0;
  a = input->dof/2.0;
  ASSERT ( (x >= 0.0) && (a > 0.0), status, THRESHOLDS_EPOSARG,
           THRESHOLDS_MSGEPOSARG);

  if (x < (a+1.0))
    {
      INT4 n;
      REAL8 sum;
      REAL8 del;
      REAL8 temp3;
      ww=logGamma(a);
      if (x <= 0.0)
	{
	  temp1=0.0;
	}
      else 
	{
	  temp3=a;
	  del=sum=1.0/a;
	  n=1;
	  do
	    {
	      ++temp3;
	      del *= x/temp3;
	      sum += del;
	      n++;
	    }
	  while( (n<=maxloop) && (fabs(del) > fabs(sum)*small) );

	  ASSERT(n < maxloop, status, THRESHOLDS_EMXIT, THRESHOLDS_MSGEMXIT);
	  temp1=sum*exp(-x+a*log(x)-(ww));
	}
      *prob=1.0-temp1;
    } 
  else 
    {
      INT4 i;
      REAL8 an,b,c,d,del,h;
      ww=logGamma(a);
      b=x+1.0-a;
      c=1.0/small1;
      d=1.0/b;
      h=d;
      i=1;
      do
	{
	  an = -i*(i-a);
	  b += 2.0;
	  d=an*d+b;
	  if (fabs(d) < small1) d=small1;
	  c=b+an/c;
	  if (fabs(c) < small1) c=small1;
	  d=1.0/d;
	  del=d*c;
	  h *= del;
	  if (fabs(del-1.0) < small) break;
	  i++;
	}
      while( (i<=maxloop) && (fabs(del-1.0) > small) );

      ASSERT(i < maxloop, status, THRESHOLDS_EMXIT, THRESHOLDS_MSGEMXIT);
      temp2=exp(-x+a*log(x)-(ww))*h;
      *prob= temp2;
    }

  /* 
   *  check that final answer is a legal probability
   *  third test is necessary since there are some numbers x for 
   *  which (x>0.0) evaluates as TRUE but for which 1/x evaluates to inf
   */
  ASSERT( (*prob > 0.0) && (*prob < 1.0) && ( 1.0/(*prob) < LAL_REAL8_MAX ), 
          status, THRESHOLDS_ERANGE, THRESHOLDS_MSGERANGE);


  RETURN (status);
}





void
LALNoncChisqCdf (
	      LALStatus            *status,
	      REAL8             *prob,
	      ChisqCdfIn        *input
	      )
{
  /*
   *  Cumulative distribution function for noncentral chi-squared distribution
   *  
   *  returns probability that (x_1+rho)^2 + x_2^2 + .. x_dof^2 \le chi2, where
   *  x_1, ..  x_dof are independent Gaussians of zero mean and
   *  unit variance, and where nonCentral = rho^2 and 
   *  dof = number of degrees of freedom
   *
   *  We use the series formula to evaluate the probability.  Each term in the
   *  series involves a call to LALChisqCdf().
   */

  ChisqCdfIn  localparams;  
  /* 
   *  local copies of input parameters.  localparams.nonCentral and 
   *  localparams.chi2 will not change, but localparams.dof will change
   *  from one call to the next of LALChisqCdf().
   *
   */

  REAL8 temp;             /* temporary variables used in calculations */
  REAL8 current;
  REAL8 sum;
  INT4 n;

  const REAL8 fractionalAccuracy = 1.0e-10;
  const INT4 maxloop=1000;  /* maximum number of terms in series to be used */


  INITSTATUS (status, "LALNoncChisqCdf", THRESHOLDSC);
  ATTATCHSTATUSPTR (status);

  /* check that pointers are not null */
  ASSERT (prob, status, THRESHOLDS_ENULLP, THRESHOLDS_MSGENULLP);
  ASSERT (input, status, THRESHOLDS_ENULLP, THRESHOLDS_MSGENULLP);


  /* Arguments chi2, dof and nonCentral must be non-negative */
  localparams = *input;
  ASSERT ( (localparams.dof > 0.0)   &&
           (localparams.chi2 >= 0.0) && 
           (localparams.nonCentral >=0.0 ),
	   status, THRESHOLDS_EPOSARG, THRESHOLDS_MSGEPOSARG);


   /* Evaluate the first term in the series   */
  LALChisqCdf (status->statusPtr, &temp, &localparams);
  CHECKSTATUSPTR (status);
  sum = temp * exp(- localparams.nonCentral/2.0);

  /* 
   *  Now add successive higher terms in the series until either sufficient
   *  accuracy is achieved, or we exceed the maximum allowed number of terms
   */

  n=0;
  do
    {
      n++;
      localparams.dof = input->dof + 2.0 * (REAL8)(n);
      LALChisqCdf (status->statusPtr, &temp, &localparams );
      CHECKSTATUSPTR (status);
      current = exp(- localparams.nonCentral/2.0 + ((REAL8)(n)) * log(localparams.nonCentral/2.0)) * temp / Factorial(n);  
      sum += current;
    }
  while( ((current*current)/(sum*sum) > fractionalAccuracy) && (n < maxloop) );

  ASSERT(n < maxloop, status, THRESHOLDS_EMXIT, THRESHOLDS_MSGEMXIT);

  *prob = sum;
  
  /* 
   *  check that final answer is a legal probability
   *  third test is necessary since there are some numbers x for 
   *  which (x>0.0) evaluates as TRUE but for which 1/x evaluates to inf
   */
  ASSERT( (*prob > 0.0) && (*prob < 1.0) && ( 1.0/(*prob) < LAL_REAL8_MAX ), 
          status, THRESHOLDS_ERANGE, THRESHOLDS_MSGERANGE);

  DETATCHSTATUSPTR (status);
  RETURN (status);
}



void
LALChi2Threshold (
	      LALStatus            *status,
	      REAL8             *chi2,
	      Chi2ThresholdIn   *input
	      )
{
  /*
   *  threshold for chi2:  returns value of chi2 such that
   *  falseAlarm = 1 - chisqCdf(chi2,dof)
   */  

  REAL8              lnchi2Ans;
  DFindRootIn        frInput; 

  INITSTATUS (status, "LALChi2Threshold", THRESHOLDSC);
  ATTATCHSTATUSPTR (status);

  /* check that pointers are not null */
  ASSERT (chi2, status, THRESHOLDS_ENULLP, THRESHOLDS_MSGENULLP);
  ASSERT (input, status, THRESHOLDS_ENULLP, THRESHOLDS_MSGENULLP);

  /* Argument dof must be positive */
  ASSERT ( input->dof > 0.0 , status,
           THRESHOLDS_EPOSARG, THRESHOLDS_MSGEPOSARG);


  /* Supplied false alarm probability must be between 0 and 1 */
  ASSERT ( (input->falseAlarm > 0.0) && (input->falseAlarm < 1.0) , status,
           THRESHOLDS_EBADPROB, THRESHOLDS_MSGEBADPROB);


  /* Initialize input structure for DFindRoot() */
  frInput.function = ChisqCdf1;
  frInput.xmin = -2.0;
  frInput.xmax = 2.0;
  frInput.xacc = 1e-5;
 
  /* Now bracket and find the root */
  LALDBracketRoot (status->statusPtr, &frInput, input );
  CHECKSTATUSPTR (status);
  
  LALDBisectionFindRoot (status->statusPtr, &lnchi2Ans, 
			      &frInput, input );
  CHECKSTATUSPTR (status);

  *chi2 = exp(lnchi2Ans);
  
  DETATCHSTATUSPTR (status);
  RETURN (status);

}






void
LALRhoThreshold (
	      LALStatus            *status,
	      REAL8             *rho,
	      RhoThresholdIn    *input
	      )
{
  /*
   *  threshold for rho:  returns value of rho such that
   *  falseAlarm = noncChisqCdf(chi2,dof,rho^2)
   *  note that rho^2 is the same as nonCentral
   */
  
  REAL8              lnrhoAns;
  DFindRootIn        frInput; 

  INITSTATUS (status, "LALRhoThreshold", THRESHOLDSC);
  ATTATCHSTATUSPTR (status);

  /* check that pointers are not null */
  ASSERT (rho, status, THRESHOLDS_ENULLP, THRESHOLDS_MSGENULLP);
  ASSERT (input, status, THRESHOLDS_ENULLP, THRESHOLDS_MSGENULLP);

  /* Arguments dof and chi2 must be positive */
  ASSERT ( (input->dof > 0.0 ) && (input->chi2 >= 0.0), status,
           THRESHOLDS_EPOSARG, THRESHOLDS_MSGEPOSARG);

  /* Supplied false dismissal probability must be between 0 and 1 */
  ASSERT ( (input->falseDismissal > 0.0) && (input->falseDismissal < 1.0),
           status, THRESHOLDS_EBADPROB, THRESHOLDS_MSGEBADPROB);


  /* Initialize input structure for DFindRoot() */
  frInput.function = NoncChisqCdf1;
  frInput.xmin = -2.0;
  frInput.xmax = 2.0;
  frInput.xacc = 1e-5;
 

  /* Now bracket and find the root */
  LALDBracketRoot (status->statusPtr, &frInput, input );
  CHECKSTATUSPTR (status);
  
  LALDBisectionFindRoot (status->statusPtr, &lnrhoAns, 
                          &frInput, input );
  CHECKSTATUSPTR (status);

  *rho = exp(lnrhoAns);
  
  DETATCHSTATUSPTR (status);
  RETURN (status);

}








/********************************** <lalVerbatim file="ThresholdsHV">
Author: Flanagan, E
$Id$
**************************************************** </lalVerbatim> */


#ifndef _THRESHOLDS_H
#define _THRESHOLDS_H


#include <lal/LALStdlib.h>
#include <lal/LALDatatypes.h>
#include <lal/LALRCSID.h>

#ifdef  __cplusplus   /* C++ protection. */
extern "C" {
#endif


NRCSID (THRESHOLDSH, "$Id$");


/******** <lalErrTable file="ThresholdsHErrTab"> ********/
#define THRESHOLDSH_ENULLP    1
#define THRESHOLDSH_EPOSARG   2
#define THRESHOLDSH_EMXIT     4
#define THRESHOLDSH_EBADPROB  8
#define THRESHOLDSH_ERANGE    16

#define THRESHOLDSH_MSGENULLP    "Null pointer"
#define THRESHOLDSH_MSGEPOSARG   "Arguments must be non-negative"
#define THRESHOLDSH_MSGEMXIT     "Maximum iterations exceeded"
#define THRESHOLDSH_MSGEBADPROB  "Supplied probability must be between 0 and 1"
#define THRESHOLDSH_MSGERANGE    "Arguments too large, cannot obtain finite probability"
/******** </lalErrTable> ********/


typedef struct tagChisqCdfIn
{
  REAL8     chi2;               /* value of chi squared          */
  REAL8     dof;                /* number of degrees of freedom  */
  REAL8     nonCentral;         /* non-centrality parameter      */
}
ChisqCdfIn;

typedef struct tagChi2ThresholdIn
{
  REAL8     dof;                /* number of degrees of freedom  */
  REAL8     falseAlarm;         /* false alarm probability       */
}
Chi2ThresholdIn;

typedef struct tagRhoThresholdIn
{
  REAL8     chi2;               /* value of chi squared          */
  REAL8     dof;                /* number of degrees of freedom  */
  REAL8     falseDismissal;     /* false dismissal probability   */
}
RhoThresholdIn;



void
LALChisqCdf (
          LALStatus                        *status,
          REAL8                         *prob,
          ChisqCdfIn                    *input
          );

void
LALOneMinusChisqCdf (
                  LALStatus                *status,
                  REAL8                 *prob,
                  ChisqCdfIn            *input
                  );

void
LALNoncChisqCdf (
              LALStatus                    *status,
              REAL8                     *prob,
              ChisqCdfIn                *input
              );

void
LALChi2Threshold (
              LALStatus                    *status,
              REAL8                     *chi2,
              Chi2ThresholdIn           *input
               );

void
LALRhoThreshold (
              LALStatus                    *status,
              REAL8                     *rho,
              RhoThresholdIn            *input
              );



#ifdef  __cplusplus
}
#endif  /* C++ protection. */


#endif







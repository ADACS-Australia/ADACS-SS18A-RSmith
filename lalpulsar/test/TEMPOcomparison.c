/*
*  Copyright (C) 2015 Reinhard Prix [XLALified]
*  Copyright (C) 2007 Chris Messenger
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

/**
 * \author Chris Messenger
 * \file
 * \ingroup lalpulsar_coh
 * \brief Tests for CW barycentric timing functions by comparing to tempo2
 *
 */

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <lal/LALStdlib.h>
#include <lal/LALConstants.h>
#include <lal/AVFactories.h>
#include <lal/SeqFactories.h>
#include <lal/PulsarDataTypes.h>
#include <lal/LALInitBarycenter.h>
#include <lal/GeneratePulsarSignal.h>
#include <lal/Random.h>
#include <lal/LALString.h>
#include <lal/UserInput.h>
#include <lal/TranslateAngles.h>

/*---------- local defines ---------- */
#define TRUE (1==1)
#define FALSE (1==0)

#define GBT_LOCATION_X 882589.65
#define GBT_LOCATION_Y -4924872.32
#define GBT_LOCATION_Z 3943729.348

#define NARRABRI_LOCATION_X -4752329.7000
#define NARRABRI_LOCATION_Y 2790505.9340
#define NARRABRI_LOCATION_Z -3200483.7470

#define ARECIBO_LOCATION_X 2390490.0
#define ARECIBO_LOCATION_Y  -5564764.0
#define ARECIBO_LOCATION_Z 1994727.0

#define NANSHAN_LOCATION_X -228310.702
#define NANSHAN_LOCATION_Y  4631922.905
#define NANSHAN_LOCATION_Z 4367064.059

#define DSS_43_LOCATION_X -4460892.6
#define DSS_43_LOCATION_Y 2682358.9
#define DSS_43_LOCATION_Z -3674756.0

#define PARKES_LOCATION_X  -4554231.5
#define PARKES_LOCATION_Y 2816759.1
#define PARKES_LOCATION_Z -3454036.3

#define JODRELL_LOCATION_X 3822252.643
#define JODRELL_LOCATION_Y -153995.683
#define JODRELL_LOCATION_Z 5086051.443

#define VLA_LOCATION_X  -1601192.
#define VLA_LOCATION_Y -5041981.4
#define VLA_LOCATION_Z 3554871.4

#define NANCAY_LOCATION_X  4324165.81
#define NANCAY_LOCATION_Y 165927.11
#define NANCAY_LOCATION_Z 4670132.83

#define EFFELSBERG_LOCATION_X 4033949.5
#define EFFELSBERG_LOCATION_Y 486989.4
#define EFFELSBERG_LOCATION_Z 4900430.8

#define JODRELLM4_LOCATION_X 3822252.643
#define JODRELLM4_LOCATION_Y -153995.683
#define JODRELLM4_LOCATION_Z 5086051.443

#define GB300_LOCATION_X 881856.58
#define GB300_LOCATION_Y -4925311.86
#define GB300_LOCATION_Z 3943459.70

#define GB140_LOCATION_X 882872.57
#define GB140_LOCATION_Y -4924552.73
#define GB140_LOCATION_Z 3944154.92

#define GB853_LOCATION_X 882315.33
#define GB853_LOCATION_Y -4925191.41
#define GB853_LOCATION_Z 3943414.05

#define LA_PALMA_LOCATION_X 5327021.651
#define LA_PALMA_LOCATION_Y -1719555.576
#define LA_PALMA_LOCATION_Z 3051967.932

#define Hobart_LOCATION_X -3950077.96
#define Hobart_LOCATION_Y 2522377.31
#define Hobart_LOCATION_Z -4311667.52

#define Hartebeesthoek_LOCATION_X 5085442.780
#define Hartebeesthoek_LOCATION_Y 2668263.483
#define Hartebeesthoek_LOCATION_Z -2768697.034

#define WSRT_LOCATION_X 3828445.659
#define WSRT_LOCATION_Y 445223.600000
#define WSRT_LOCATION_Z 5064921.5677

#define COE_LOCATION_X  0.0
#define COE_LOCATION_Y 1.0
#define COE_LOCATION_Z 0.0

/*    882589.65    -4924872.32     3943729.348      GBT                 gbt   */
/* -4752329.7000  2790505.9340     -3200483.7470    NARRABRI            atca */
/*   2390490.0     -5564764.0      1994727.0        ARECIBO             ao */
/*  -228310.702   4631922.905      4367064.059      NANSHAN             nanshan */
/*  -4460892.6     2682358.9       -3674756.0       DSS_43              tid43 */
/*  -4554231.5     2816759.1       -3454036.3       PARKES              pks */
/*  3822252.643   -153995.683      5086051.443      JODRELL             jb */
/*  -1601192.     -5041981.4       3554871.4        VLA                 vla */
/*  4324165.81     165927.11       4670132.83       NANCAY              ncy */
/*  4033949.5      486989.4        4900430.8        EFFELSBERG          eff */
/*  3822252.643   -153995.683      5086051.443      JODRELLM4           jbm4 */
/*  881856.58     -4925311.86      3943459.70       GB300               gb300 */
/*  882872.57     -4924552.73      3944154.92       GB140               gb140 */
/*  882315.33     -4925191.41      3943414.05       GB853               gb853 */
/*  5327021.651    -1719555.576    3051967.932      LA_PALMA            lap */
/*  -3950077.96    2522377.31     -4311667.52       Hobart              hob */
/*  5085442.780   2668263.483     -2768697.034      Hartebeesthoek      hart */
/*  3828445.659  445223.600000  5064921.5677        WSRT                wsrt */
/* # */
/* ####### From Jodrell obsys.dat file */
/* # */
/*  383395.727    -173759.585      5077751.313      MKIII               j   */
/*  3817176.557   -162921.170      5089462.046      TABLEY              k   */
/*  3828714.504   -169458.987      5080647.749      DARNHALL            l   */
/*  3859711.492   -201995.082      5056134.285      KNOCKIN             m   */
/*  3923069.135   -146804.404      5009320.570      DEFFORD             n   */
/*        0.0           1.0              0.0        COE                 coe   */
/* # */
/* ###### Telescope ID changed from the Jodrell obsys.dat file */
/* # */
/*  3822473.365   -153692.318      5085851.303      JB_MKII             jbmk2 */
/*  3822294.825   -153862.275      5085987.071      JB_42ft             jb42 */
/* # */
/* # New telescopes */
/* # 284543.5      -175324.040      2400.            LA_PALMA            p */
/*  1719555.576   5327021.651      3051967.932      LA_PALMA            p */

/* ---------- local types ---------- */

/** user input variables */
typedef struct
{
  CHAR *RAJ;
  CHAR *DECJ;
  REAL8 TstartUTCMJD;
  REAL8 TrefTDBMJD;
  REAL8 DeltaTMJD;
  REAL8 DurationMJD;
  CHAR *det;
  CHAR *ephemEarth;		/* Earth ephemeris file to use */
  CHAR *ephemSun;		/* Sun ephemeris file to use */
  REAL8 f0;
  REAL8 fdot;
  CHAR *PSRJ;
  CHAR *Observatory;
  INT4 randSeed;

} UserVariables_t;

/* a time structure to accurately store MJD times so as not to lose any precision */
typedef struct
{
  INT4 days;       /* integer MJD days */
  REAL8 fracdays;  /* fractional day */
} MJDTime;

/* ---------- global variables ----------*/
extern int vrbflg;

/* ---------- local prototypes ---------- */
int initUserVars ( int argc, char *argv[], UserVariables_t *uvar );

void TDBMJDtoGPS ( LIGOTimeGPS *GPS, MJDTime MJD );
void AddIntervaltoMJD (double interval, MJDTime *MJDout, MJDTime MJDin);
void REAL8toMJD ( MJDTime *MJD, REAL8 x );
int UTCMJDtoGPS ( LIGOTimeGPS *GPS, MJDTime MJD, INT4 leap );
void UTCGPStoMJD ( MJDTime *MJD, LIGOTimeGPS *GPS, INT4 leap );
void TDBGPStoMJD ( MJDTime *MJD, LIGOTimeGPS GPS, INT4 leap );
REAL8 MJDtoREAL8 ( MJDTime MJD );
void deltaMJD ( MJDTime *deltaMJD, MJDTime *x, MJDTime *y );
void GPStoTDBMJD ( MJDTime *TDBMJD, LIGOTimeGPS GPS );

/*============================================================
 * FUNCTION definitions
 *============================================================*/

int
main(int argc, char *argv[])
{
  FILE *fp = NULL;
  BarycenterInput XLAL_INIT_DECL(baryinput);
  INT4 leap0,leap;
  LIGOTimeGPS epoch;
  LIGOTimeGPS TstartSSB, TendSSB, TendGPS;
  INT4 n;
  LIGOTimeGPS *TSSB = NULL;
  MJDTime TstartUTCMJD;
  LIGOTimeGPS TDET;
  REAL8 temp;
  INT4 i;
  MJDTime tempTOA;
  REAL8 dt;
  LIGOTimeGPS TstartGPS;
  MJDTime *TOA = NULL;
  CHAR tempstr[18];
  CHAR *tempstr2;
  CHAR TstartMJDstr[20],TfinishMJDstr[20],TOAstr[22];
  PulsarSignalParams pulsarparams;
  CHAR parfile[256];
  CHAR timfile[256];
  CHAR detcode[16];
  REAL8 TstartUTCMJDtest;
  REAL8 diff;
  MJDTime MJDdiff, MJDtest;

  MJDTime TrefTDBMJD;
  LIGOTimeGPS TrefSSB_TDB_GPS;

  // ----------------------------------------------------------------------
  UserVariables_t XLAL_INIT_DECL(uvar);
  XLAL_CHECK ( initUserVars (argc, argv, &uvar) == XLAL_SUCCESS, XLAL_EFUNC );

  unsigned int seed = uvar.randSeed;
  if ( uvar.randSeed == 0 ) {
    seed = clock();
  }
  srand ( seed );

  // ----- sky position: random or user-specified ----------
  REAL8 alpha, delta;
  CHAR *RAJ = NULL, *DECJ = NULL;

  BOOLEAN have_RAJ  = XLALUserVarWasSet ( &uvar.RAJ );
  BOOLEAN have_DECJ = XLALUserVarWasSet ( &uvar.DECJ );
  if ( have_RAJ )
    {
      XLAL_CHECK ( XLALTranslateHMStoRAD ( &alpha, uvar.RAJ ) == XLAL_SUCCESS, XLAL_EFUNC );
      RAJ = XLALStringDuplicate ( uvar.RAJ );
    }
  else
    { // pick randomly
      alpha = LAL_TWOPI * (1.0 * rand() / ( RAND_MAX + 1.0 ) );  // alpha uniform in [0, 2pi)
      XLAL_CHECK ( (RAJ = XLALTranslateRADtoHMS ( alpha )) != NULL, XLAL_EFUNC );
    }
  if ( have_DECJ )
    {
      XLAL_CHECK ( XLALTranslateDMStoRAD ( &delta, uvar.DECJ ) == XLAL_SUCCESS, XLAL_EFUNC );
      DECJ = XLALStringDuplicate ( uvar.DECJ );
    }
  else
    { // pick randomly
      delta = LAL_PI_2 - acos ( 1 - 2.0 * rand()/RAND_MAX );	// sin(delta) uniform in [-1,1]
      XLAL_CHECK ( (DECJ = XLALTranslateRADtoDMS ( delta )) != NULL, XLAL_EFUNC );
    }

  /* define start time in an MJD structure */
  REAL8toMJD ( &TstartUTCMJD, uvar.TstartUTCMJD );
  XLALPrintInfo ( "TstartUTCMJD=%f converted to MJD days = %d fracdays = %6.12f\n", uvar.TstartUTCMJD, TstartUTCMJD.days, TstartUTCMJD.fracdays );

  /* convert back to test conversions */
  TstartUTCMJDtest = MJDtoREAL8 (TstartUTCMJD);
  diff = uvar.TstartUTCMJD - TstartUTCMJDtest;
  if ( fabs(diff) > 1e-9) {
    fprintf(stderr,"ERROR : Time conversion gives discrepancy of %e sec. Exiting.\n",diff);
    return(-1);
  }
  XLALPrintInfo ( "MJD conversion gives discrepancies of %e sec\n", diff);

  /* use start time to define an epoch for the leap seconds */
  /* Note that epochs are defined in TDB !!! but here we only need to be rough to get a leap second value */
  TDBMJDtoGPS(&epoch,TstartUTCMJD);
  XLALPrintInfo ( "leap second epoch = %d %d\n",epoch.gpsSeconds,epoch.gpsNanoSeconds);

  /* deal with ephemeris files and compute leap seconds */
  EphemerisData *edat;
  XLAL_CHECK ( (edat = XLALInitBarycenter( uvar.ephemEarth, uvar.ephemSun )) != NULL, XLAL_EFUNC );

  leap0 = XLALGPSLeapSeconds (epoch.gpsSeconds);
  XLALPrintInfo ( "leap seconds = %d\n",leap0);

  /* select detector location */
  if (strcmp(uvar.Observatory,"GBT")==0) {
    baryinput.site.location[0] = GBT_LOCATION_X;
    baryinput.site.location[1] = GBT_LOCATION_Y;
    baryinput.site.location[2] = GBT_LOCATION_Z;
    sprintf(detcode,"gbt");
  }
  else if (strcmp(uvar.Observatory,"NARRABRI")==0) {
    baryinput.site.location[0] = NARRABRI_LOCATION_X;
    baryinput.site.location[1] = NARRABRI_LOCATION_Y;
    baryinput.site.location[2] = NARRABRI_LOCATION_Z;
    sprintf(detcode,"atca");
  }
   else if (strcmp(uvar.Observatory,"ARECIBO")==0) {
    baryinput.site.location[0] = ARECIBO_LOCATION_X;
    baryinput.site.location[1] = ARECIBO_LOCATION_Y;
    baryinput.site.location[2] = ARECIBO_LOCATION_Z;
    sprintf(detcode,"ao");
  }
   else if (strcmp(uvar.Observatory,"NANSHAN")==0) {
    baryinput.site.location[0] = NANSHAN_LOCATION_X;
    baryinput.site.location[1] = NANSHAN_LOCATION_Y;
    baryinput.site.location[2] = NANSHAN_LOCATION_Z;
    sprintf(detcode,"nanshan");
  }
   else if (strcmp(uvar.Observatory,"DSS_43")==0) {
    baryinput.site.location[0] = DSS_43_LOCATION_X;
    baryinput.site.location[1] = DSS_43_LOCATION_Y;
    baryinput.site.location[2] = DSS_43_LOCATION_Z;
    sprintf(detcode,"tid43");
  }
  else if (strcmp(uvar.Observatory,"PARKES")==0) {
    baryinput.site.location[0] = PARKES_LOCATION_X;
    baryinput.site.location[1] = PARKES_LOCATION_Y;
    baryinput.site.location[2] = PARKES_LOCATION_Z;
    sprintf(detcode,"pks");
  }
   else if (strcmp(uvar.Observatory,"JODRELL")==0) {
    baryinput.site.location[0] = JODRELL_LOCATION_X;
    baryinput.site.location[1] = JODRELL_LOCATION_Y;
    baryinput.site.location[2] = JODRELL_LOCATION_Z;
    sprintf(detcode,"jb");
  }
   else if (strcmp(uvar.Observatory,"VLA")==0) {
    baryinput.site.location[0] = VLA_LOCATION_X;
    baryinput.site.location[1] = VLA_LOCATION_Y;
    baryinput.site.location[2] = VLA_LOCATION_Z;
    sprintf(detcode,"vla");
  }
   else if (strcmp(uvar.Observatory,"NANCAY")==0) {
    baryinput.site.location[0] = NANCAY_LOCATION_X;
    baryinput.site.location[1] = NANCAY_LOCATION_Y;
    baryinput.site.location[2] = NANCAY_LOCATION_Z;
    sprintf(detcode,"ncy");
  }
  else if (strcmp(uvar.Observatory,"EFFELSBERG")==0) {
    baryinput.site.location[0] = EFFELSBERG_LOCATION_X;
    baryinput.site.location[1] = EFFELSBERG_LOCATION_Y;
    baryinput.site.location[2] = EFFELSBERG_LOCATION_Z;
    sprintf(detcode,"eff");
  }
  else if (strcmp(uvar.Observatory,"JODRELLM4")==0) {
    baryinput.site.location[0] = JODRELLM4_LOCATION_X;
    baryinput.site.location[1] = JODRELLM4_LOCATION_Y;
    baryinput.site.location[2] = JODRELLM4_LOCATION_Z;
    sprintf(detcode,"jbm4");
  }
  else if (strcmp(uvar.Observatory,"GB300")==0) {
    baryinput.site.location[0] = GB300_LOCATION_X;
    baryinput.site.location[1] = GB300_LOCATION_Y;
    baryinput.site.location[2] = GB300_LOCATION_Z;
    sprintf(detcode,"gb300");
  }
  else if (strcmp(uvar.Observatory,"GB140")==0) {
    baryinput.site.location[0] = GB140_LOCATION_X;
    baryinput.site.location[1] = GB140_LOCATION_Y;
    baryinput.site.location[2] = GB140_LOCATION_Z;
    sprintf(detcode,"gb140");
  }
  else if (strcmp(uvar.Observatory,"GB853")==0) {
    baryinput.site.location[0] = GB853_LOCATION_X;
    baryinput.site.location[1] = GB853_LOCATION_Y;
    baryinput.site.location[2] = GB853_LOCATION_Z;
    sprintf(detcode,"gb853");
  }
  else if (strcmp(uvar.Observatory,"LA_PALMA")==0) {
    baryinput.site.location[0] = LA_PALMA_LOCATION_X;
    baryinput.site.location[1] = LA_PALMA_LOCATION_Y;
    baryinput.site.location[2] = LA_PALMA_LOCATION_Z;
    sprintf(detcode,"lap");
  }
  else if (strcmp(uvar.Observatory,"Hobart")==0) {
    baryinput.site.location[0] = Hobart_LOCATION_X;
    baryinput.site.location[1] = Hobart_LOCATION_Y;
    baryinput.site.location[2] = Hobart_LOCATION_Z;
    sprintf(detcode,"hob");
  }
  else if (strcmp(uvar.Observatory,"Hartebeesthoek")==0) {
    baryinput.site.location[0] = Hartebeesthoek_LOCATION_X;
    baryinput.site.location[1] = Hartebeesthoek_LOCATION_Y;
    baryinput.site.location[2] = Hartebeesthoek_LOCATION_Z;
    sprintf(detcode,"hart");
  }
  else if (strcmp(uvar.Observatory,"WSRT")==0) {
    baryinput.site.location[0] = WSRT_LOCATION_X;
    baryinput.site.location[1] = WSRT_LOCATION_Y;
    baryinput.site.location[2] = WSRT_LOCATION_Z;
    sprintf(detcode,"wsrt");
  }
  else if (strcmp(uvar.Observatory,"COE")==0) {
    baryinput.site.location[0] = COE_LOCATION_X;
    baryinput.site.location[1] = COE_LOCATION_Y;
    baryinput.site.location[2] = COE_LOCATION_Z;
    sprintf(detcode,"coe");
  }
  else if (strcmp(uvar.Observatory,"SSB")!=0) {
    fprintf(stderr,"ERROR. Unknown Observatory %s. Exiting.\n",uvar.Observatory);
    return(-1);
  }
  XLALPrintInfo ( "selected observatory %s - observatoryt code = %s\n",uvar.Observatory,detcode);
  XLALPrintInfo ( "baryinput location = %6.12f %6.12f %6.12f\n",baryinput.site.location[0],baryinput.site.location[1],baryinput.site.location[2]);

  /* convert start time to UTC GPS */
  UTCMJDtoGPS(&TstartGPS, TstartUTCMJD, leap0);
  XLALPrintInfo ( "TstartGPS = %d %d\n",TstartGPS.gpsSeconds,TstartGPS.gpsNanoSeconds);

  /* convert back to test conversion */
  UTCGPStoMJD(&MJDtest,&TstartGPS,leap0);
  deltaMJD ( &MJDdiff, &MJDtest, &TstartUTCMJD );
  diff = (MJDdiff.days+MJDdiff.fracdays)*86400;
  if ( fabs(diff)  > 1e-9) {
    fprintf(stderr,"ERROR : Time conversion gives discrepancy of %e sec. Exiting.\n",diff);
    return(-1);
  }
  XLALPrintInfo ( "MJD conversion gives discrepancies of %e sec\n",diff);

  /* define reference time in an MJD structure */
  REAL8toMJD ( &TrefTDBMJD, uvar.TrefTDBMJD );
  XLALPrintInfo ( "TrefTDBMJD converted to MJD days = %d fracdays = %6.12f\n",TrefTDBMJD.days,TrefTDBMJD.fracdays);

  /* convert reference time to TDB GPS */
  TDBMJDtoGPS(&TrefSSB_TDB_GPS,TrefTDBMJD);
  XLALPrintInfo ( "TrefSSB_TDB_GPS = %d %d\n",TrefSSB_TDB_GPS.gpsSeconds,TrefSSB_TDB_GPS.gpsNanoSeconds);

  /* convert back to test conversion */
  TDBGPStoMJD ( &MJDtest, TrefSSB_TDB_GPS, leap0 );
  deltaMJD ( &MJDdiff, &MJDtest, &TrefTDBMJD );
  diff = (MJDdiff.days+MJDdiff.fracdays)*86400;
  if ( fabs(diff)  > 1e-9) {
    fprintf(stderr,"ERROR : Time conversion gives discrepancy of %e sec. Exiting.\n",diff);
    return(-1);
  }
  XLALPrintInfo ( "MJD conversion gives discrepancies of %e sec\n",diff);

  /* fill in required pulsar params structure for Barycentering */
  LALDetector *site = NULL;
  site = (LALDetector *)LALMalloc(sizeof(LALDetector));
  site->location[0] = baryinput.site.location[0];
  site->location[1] = baryinput.site.location[1];
  site->location[2] = baryinput.site.location[2];
  pulsarparams.site = site;

  pulsarparams.pulsar.position.longitude = alpha;
  pulsarparams.pulsar.position.latitude = delta;
  pulsarparams.pulsar.position.system = COORDINATESYSTEM_EQUATORIAL;
  pulsarparams.ephemerides = edat;

  /* generate SSB initial TOA in GPS */
  XLALConvertGPS2SSB ( &TstartSSB, TstartGPS, &pulsarparams);
  XLALPrintInfo ( "TstartSSB = %d %d\n",TstartSSB.gpsSeconds,TstartSSB.gpsNanoSeconds);

  /* define TOA end time in GPS */
  temp = uvar.DurationMJD*86400.0;
  TendGPS = TstartGPS;
  XLALGPSAdd(&TendGPS, temp);
  XLALPrintInfo ( "GPS end time of TOAs = %d %d\n",TendGPS.gpsSeconds,TendGPS.gpsNanoSeconds);

  /* generate SSB end time in GPS (force integer seconds) */
  XLALConvertGPS2SSB (&TendSSB,TendGPS,&pulsarparams);
  XLALPrintInfo  ( "TendSSB = %d %d\n",TendSSB.gpsSeconds,TendSSB.gpsNanoSeconds);

  /* define TOA seperation in the SSB */
  dt = uvar.DeltaTMJD*86400.0;
  n = (INT4)ceil(uvar.DurationMJD/uvar.DeltaTMJD);
  XLALPrintInfo ( "TOA seperation at SSB = %g sec\n",dt);
  XLALPrintInfo ( "number of TOAs to generate = %d\n",n);

  /* allocate memory for artificial SSB TOAs */
  TSSB = (LIGOTimeGPS *)LALMalloc(n*sizeof(LIGOTimeGPS));
  TOA = (MJDTime *)LALMalloc(n*sizeof(MJDTime));

  /* generate artificial SSB TOAs given the phase model phi = 2*pi*(f0*(t-tref) + 0.5*fdot*(t-tref)^2) */
  for  (i=0;i<n;i++)
    {
      REAL8 dtref,fnow,cyclefrac,dtcor;
      LIGOTimeGPS tnow;

      /* define current interval */
      XLALPrintInfo ( "current (t-tstart) = %g sec\n", i * dt);

      /* define current t */
      tnow = TstartSSB;
      XLALGPSAdd(&tnow, i * dt);
      XLALPrintInfo ( "current t = %d %d\n",tnow.gpsSeconds,tnow.gpsNanoSeconds);

      /* define current t-tref */
      dtref = XLALGPSDiff(&tnow,&TrefSSB_TDB_GPS);
      XLALPrintInfo ( "current (t - tref) = %9.12f\n",dtref);

      dtcor = 1;
      while (dtcor>1e-9)
        {

          /* define actual cycle fraction at requested time */
          cyclefrac = fmod(uvar.f0*dtref + 0.5*uvar.fdot*dtref*dtref,1.0);
          XLALPrintInfo ( "cyclefrac = %9.12f\n",cyclefrac);

          /* define instantaneous frequency */
          fnow = uvar.f0 + uvar.fdot*dtref;
          XLALPrintInfo ( "instananeous frequency = %9.12f\n",fnow);

          /* add correction to time */
          dtcor = cyclefrac/fnow;
          dtref -= dtcor;
          XLALPrintInfo ( "timing correction = %9.12f\n",dtcor);
          XLALPrintInfo ( "corrected dtref to = %9.12f\n",dtref);
        } // while dtcor>1e-9

      /* define time of zero phase */
      TSSB[i] = TrefSSB_TDB_GPS;
      XLALGPSAdd(&TSSB[i], dtref);
      XLALPrintInfo ( "TSSB[%d] = %d %d\n",i,TSSB[i].gpsSeconds,TSSB[i].gpsNanoSeconds);
    } // for i < n

  /* loop over SSB time of arrivals and compute detector time of arrivals */
  for (i=0;i<n;i++)
    {
      LIGOTimeGPS TSSBtest;
      LIGOTimeGPS GPStest;

      /* convert SSB to Detector time */
      int ret = XLALConvertSSB2GPS ( &TDET, TSSB[i], &pulsarparams);
      if ( ret != XLAL_SUCCESS ) {
        XLALPrintError ("XLALConvertSSB2GPS() failed with xlalErrno = %d\n", xlalErrno );
	return(-1);
      }

      XLALPrintInfo ( "converted SSB TOA %d %d -> Detector TOA %d %d\n",TSSB[i].gpsSeconds,TSSB[i].gpsNanoSeconds,TDET.gpsSeconds,TDET.gpsNanoSeconds);

      /* convert back for testing conversion */
      XLALConvertGPS2SSB (&TSSBtest,TDET,&pulsarparams);
      diff = XLALGPSDiff(&TSSBtest,&TSSB[i]);
      if ( fabs(diff)  > 1e-9) {
	fprintf(stderr,"ERROR : Time conversion gives discrepancy of %e sec. Exiting.\n",diff);
	return(-1);
      }
      XLALPrintInfo ( "SSB -> detector conversion gives discrepancies of %e sec\n",diff);

      /* recompute leap seconds incase they've changed */
      leap = XLALGPSLeapSeconds (TDET.gpsSeconds);

      /* must now convert to an MJD time for TEMPO */
      /* Using UTC conversion as used by Matt in his successful comparison */
      UTCGPStoMJD (&tempTOA,&TDET,leap);
      XLALPrintInfo ( "output MJD time = %d %6.12f\n",tempTOA.days,tempTOA.fracdays);

      /* convert back to test conversion */
      UTCMJDtoGPS ( &GPStest, tempTOA, leap );
      diff = XLALGPSDiff(&TDET,&GPStest);
      if ( fabs(diff)  > 1e-9) {
	fprintf(stderr,"ERROR. Time conversion gives discrepancy of %e sec. Exiting.\n",diff);
	return(-1);
      }
      XLALPrintInfo ( "MJD time conversion gives discrepancies of %e sec\n",diff);

      /* fill in results */
      TOA[i].days = tempTOA.days;
      TOA[i].fracdays = tempTOA.fracdays;

    } // for i < n

  snprintf(tempstr,15,"%1.13f",TOA[0].fracdays);
  tempstr2 = tempstr+2;
  snprintf(TstartMJDstr,19,"%d.%s",TOA[0].days,tempstr2);
  XLALPrintInfo ( "Converted initial TOA MJD %d %6.12f to the string %s\n",TOA[0].days,TOA[0].fracdays,TstartMJDstr);

  snprintf(tempstr,15,"%1.13f",TOA[n-1].fracdays);
  tempstr2 = tempstr+2;
  snprintf(TfinishMJDstr,19,"%d.%s",TOA[n-1].days,tempstr2);
  XLALPrintInfo ( "*** Converted MJD to a string %s\n",TfinishMJDstr);
  XLALPrintInfo ( "Converted final TOA MJD %d %6.12f to the string %s\n",TOA[n-1].days,TOA[n-1].fracdays,TfinishMJDstr);

  /* define output file names */
  sprintf(parfile,"%s.par",uvar.PSRJ);
  sprintf(timfile,"%s.tim",uvar.PSRJ);

  /* output to par file in format required by TEMPO 2 */
  if ((fp = fopen(parfile,"w")) == NULL) {
    fprintf(stderr,"ERROR. Could not open file %s. Exiting.\n",parfile);
    return(-1);
  }
  fprintf(fp,"PSRJ\t%s\n",uvar.PSRJ);
  fprintf(fp,"RAJ\t%s\t1\n",RAJ);
  fprintf(fp,"DECJ\t%s\t1\n",DECJ);
  fprintf(fp,"PEPOCH\t%6.12f\n",uvar.TrefTDBMJD);
  fprintf(fp,"POSEPOCH\t%6.12f\n",uvar.TrefTDBMJD);
  fprintf(fp,"DMEPOCH\t%6.12f\n",uvar.TrefTDBMJD);
  fprintf(fp,"DM\t0.0\n");
  fprintf(fp,"F0\t%6.16f\t1\n",uvar.f0);
  fprintf(fp,"F1\t%6.16f\t0\n",uvar.fdot);
  fprintf(fp,"START\t%s\n",TstartMJDstr);
  fprintf(fp,"FINISH\t%s\n",TfinishMJDstr);
  fprintf(fp,"TZRSITE\t%s\n",detcode);
  fprintf(fp,"CLK\tUTC(NIST)\n");
  fprintf(fp,"EPHEM\tDE405\n");
  fprintf(fp,"UNITS\tTDB\n");
  fprintf(fp,"MODE\t0\n");

  /* close par file */
  fclose(fp);

  /* output to tim file in format required by TEMPO 2 */
  if ((fp = fopen(timfile,"w")) == NULL) {
    fprintf(stderr,"ERROR. Could not open file %s. Exiting.\n",timfile);
    return(-1);
  }

  fprintf(fp,"FORMAT 1\n");
  for (i=0;i<n;i++)
    {
      /* convert each TOA to a string for output */
      snprintf(tempstr,18,"%1.16f",TOA[i].fracdays);
      tempstr2 = tempstr+2;
      snprintf(TOAstr,22,"%d.%s",TOA[i].days,tempstr2);
      fprintf(fp,"blank.dat\t1000.0\t%s\t1.0\t%s\n",TOAstr,detcode);
      XLALPrintInfo ( "Converting MJD time %d %6.16f to string %s\n",TOA[i].days,TOA[i].fracdays,TOAstr);
    } // for i < n

  /* close tim file */
  fclose(fp);

  /* free memory */
  XLALFree ( TSSB );
  XLALFree ( TOA );
  XLALFree ( site );
  XLALDestroyEphemerisData ( edat );
  XLALDestroyUserVars ();
  LALCheckMemoryLeaks();

  return XLAL_SUCCESS;

} /* main() */


/** register all "user-variables" */
int
initUserVars ( int argc, char *argv[], UserVariables_t *uvar )
{
  XLAL_CHECK ( argc > 0 && (argv != NULL) && (uvar != NULL), XLAL_EINVAL );

  /* set a few defaults */
  uvar->RAJ  = NULL;
  uvar->DECJ = NULL;

  uvar->TstartUTCMJD = 53400;
  uvar->TrefTDBMJD = 53400;
  uvar->DeltaTMJD = 1;
  uvar->DurationMJD = 1800;

  uvar->f0 = 1.0;
  uvar->fdot = 0.0;

  uvar->PSRJ = XLALStringDuplicate ( "TEMPOcomparison" );

  uvar->Observatory = XLALStringDuplicate ( "JODRELL" );

  uvar->randSeed = 1;

  uvar->ephemEarth = XLALStringDuplicate("earth00-19-DE405.dat.gz");
  uvar->ephemSun = XLALStringDuplicate("sun00-19-DE405.dat.gz");

  /* register user input variables */
  XLALRegisterUvarMember( RAJ, 	        STRING, 'r', OPTIONAL, 	"Right ascension hh:mm.ss.ssss [Default=random]");
  XLALRegisterUvarMember( DECJ, 	        STRING, 'j', OPTIONAL, 	"Declination deg:mm.ss.ssss [Default=random]");
  XLALRegisterUvarMember( ephemEarth, 	 STRING, 0,  OPTIONAL, 	"Earth ephemeris file to use");
  XLALRegisterUvarMember( ephemSun, 	 	 STRING, 0,  OPTIONAL, 	"Sun ephemeris file to use");
  XLALRegisterUvarMember( f0,     		REAL8, 'f', OPTIONAL, 	"The signal frequency in Hz at SSB at the reference time");
  XLALRegisterUvarMember( fdot,     		REAL8, 'p', OPTIONAL, 	"The signal frequency derivitive in Hz at SSB at the reference time");
  XLALRegisterUvarMember( TrefTDBMJD, 	REAL8, 'R', OPTIONAL, 	"Reference time at the SSB in TDB in MJD");
  XLALRegisterUvarMember( TstartUTCMJD, 	REAL8, 'T', OPTIONAL, 	"Start time of output TOAs in UTC");
  XLALRegisterUvarMember( DeltaTMJD, 		REAL8, 't', OPTIONAL, 	"Time inbetween TOAs (in days)");
  XLALRegisterUvarMember( DurationMJD, 	REAL8, 'D', OPTIONAL, 	"Full duration of TOAs (in days)");
  XLALRegisterUvarMember( PSRJ,           	STRING, 'n', OPTIONAL, 	"Name of pulsar");
  XLALRegisterUvarMember( Observatory,    	STRING, 'O', OPTIONAL, 	"TEMPO observatory name (GBT,ARECIBO,NARRABRI,NANSHAN,DSS_43,PARKES,JODRELL,VLA,NANCAY,COE,SSB)");
  XLALRegisterUvarMember( randSeed,  		 INT4, 0,  OPTIONAL, 	"The random seed [0 = clock]");

  /* read all command line variables */
  BOOLEAN should_exit = 0;
  XLAL_CHECK( XLALUserVarReadAllInput( &should_exit, argc, argv ) == XLAL_SUCCESS, XLAL_EFUNC );
  if ( should_exit ) {
    exit(1);
  }

  return XLAL_SUCCESS;
} /* initUserVars() */



/* ------------------------------------------------------------------------------- */
/* the following functions have been written to maintain a high level of precision */
/* in storing MJD times */
/* ------------------------------------------------------------------------------- */

/* this function takes a REAL8 input MJD time and returns the same time */
/* but stored in an MJFTime structure to avoid loss of precision.       */
/* A REAL8 can only store a present day MJD time to 12 decimal figure   */
/* corresponding to 1e-7 sec */
void
REAL8toMJD( MJDTime *MJD,REAL8 x )
{
  /* take integer part of input time */
  MJD->days = (INT4)floor(x);
  MJD->fracdays = fmod(x,1.0);
} // REAL8toMJD()

REAL8
MJDtoREAL8 ( MJDTime MJD )
{
  return (REAL8)MJD.days + (REAL8)MJD.fracdays*86400.0;
} // MJDtoREAL8

void
deltaMJD ( MJDTime *dMJD, MJDTime *x, MJDTime *y )
{
  MJDTime z;

  /* remove the days part from the other time */
  z.days = x->days - y->days;
  z.fracdays = x->fracdays;

  /* remove frac days part */
  z.fracdays = z.fracdays - y->fracdays;
  if (z.fracdays<0) {
    z.fracdays = 1.0 + z.fracdays;
    z.days--;
  }

  dMJD->days = z.days;
  dMJD->fracdays = z.fracdays;

} // deltaMJD()

/* this function adds a LALTimeInterval in sec.nanosec to an MJDTime structure */
void
AddIntervaltoMJD ( double interval, MJDTime *MJDout, MJDTime MJDin )
{
  REAL8 fracdays = 0.0;
  INT4 days = 0;
  REAL8 temp,sfd;

  /* convert seconds part to fractional days */
  sfd = interval/86400.0;

  temp = MJDin.fracdays + sfd;
  fracdays = fmod(temp,1.0);
  days = MJDin.days + (INT4)floor(temp);

  MJDout->days = days;
  MJDout->fracdays = fracdays;
} // AddIntervaltoMJD()

void
GPStoTDBMJD ( MJDTime *TDBMJD, LIGOTimeGPS GPS )
{
  REAL8 Tdiff, dtrel;
  REAL8 MJDtemp;
  MJDTime MJDtest;
  LIGOTimeGPS GPStemp;

  /* straight forward conversion from GPS to MJD */
  /* we do not need the more accurate MJDtime structure */
  MJDtemp = ((REAL8)GPS.gpsSeconds + 1e-9*(REAL8)GPS.gpsNanoSeconds)/86400.0 + 44244.0;

  /* Check not before the start of GPS time (MJD 44222) */
  if(MJDtemp < 44244.){
    fprintf(stderr, "Input time is not in range.\n");
    exit(0);
  }

  /* compute the relativistic effect in TDB */
  Tdiff = MJDtemp + (2400000.5-2451545.0);
  /* meanAnomaly = 357.53 + 0.98560028*Tdiff; */ /* mean anomaly in degrees */
  /* meanAnomaly *= LAL_PI_180; */ /* mean anomaly in rads */
  /* dtrel = 0.001658*sin(meanAnomaly) + 0.000014*sin(2.*meanAnomaly); */ /* time diff in seconds */
  dtrel = 0.0016568*sin((357.5 + 0.98560028*Tdiff) * LAL_PI_180) +
    0.0000224*sin((246.0 + 0.90251882*Tdiff) * LAL_PI_180) +
    0.0000138*sin((355.0 + 1.97121697*Tdiff) * LAL_PI_180) +
    0.0000048*sin((25.0 + 0.08309103*Tdiff) * LAL_PI_180) +
    0.0000047*sin((230.0 + 0.95215058*Tdiff) *LAL_PI_180);

  /* define interval that is the magical number factor of 32.184 + 19 leap seconds to the */
  /* start of GPS time plus the TDB-TT correction and add it to the GPS input MJD */
  /* max absolute value of TDB-TT correction is < 0.184 sec */
  /* add this to the input GPS time */
  /* this time is now in TDB but represented as a GPS structure */
  GPStemp = GPS;
  XLALGPSAdd(&GPStemp, 51.184 + dtrel);

  /* convert to an MJD structure */
  MJDtest.days = (INT4)floor((REAL8)GPStemp.gpsSeconds/86400.0) + 44244;
  MJDtest.fracdays = fmod((REAL8)GPStemp.gpsSeconds/86400.0,1.0);
  AddIntervaltoMJD(GPStemp.gpsNanoSeconds / 1e9,TDBMJD,MJDtest);

} // GPStoTDBMJD()

void
TDBMJDtoGPS ( LIGOTimeGPS *GPS, MJDTime MJD )
{
  REAL8 Tdiff, TDBtoTT;
  REAL8 MJDtemp;
  LIGOTimeGPS GPStemp;
  INT4 tempsec, tempnano;

  /* convert MJD to REAL8 for calculation of the TDB-TT factor which */
  /* is small enough to allow not using the more accurate MJDtime structure */
  MJDtemp = (REAL8)MJD.days + MJD.fracdays;
 /*  printf("\tMJDtemp = %6.12f\n",MJDtemp); */

  /* Check not before the start of GPS time (MJD 44222) */
  if(MJDtemp < 44244.){
    fprintf(stderr, "Input time is not in range.\n");
    exit(0);
  }

  Tdiff = MJDtemp + (2400000.5-2451545.0);
  /* meanAnomaly = 357.53 + 0.98560028*Tdiff; */ /* mean anomaly in degrees */
 /*  printf("\tmeanAnomaly (deg) = %6.12f\n",meanAnomaly); */
  /* meanAnomaly *= LAL_PI_180; */ /* mean anomaly in rads */
  /* TDBtoTT = 0.001658*sin(meanAnomaly) + 0.000014*sin(2.*meanAnomaly); */ /* time diff in seconds */
  TDBtoTT = 0.0016568*sin((357.5 + 0.98560028*Tdiff) * LAL_PI_180) +
    0.0000224*sin((246.0 + 0.90251882*Tdiff) * LAL_PI_180) +
    0.0000138*sin((355.0 + 1.97121697*Tdiff) * LAL_PI_180) +
    0.0000048*sin((25.0 + 0.08309103*Tdiff) * LAL_PI_180) +
    0.0000047*sin((230.0 + 0.95215058*Tdiff) *LAL_PI_180);

/*  printf("\tTdiff = %6.12f meanAnomoly (rads) = %6.12f TDBtoTT = %6.12f\n",Tdiff,meanAnomaly,TDBtoTT); */

  /* convert MJDtime to GPS with no corrections (yet) */
  tempsec = (MJD.days-44244)*86400;
 /*  printf("\ttempsec = %d\n",tempsec); */
  tempsec += (INT4)floor(MJD.fracdays*86400.0);
 /*  printf("\ttempsec = %d\n",tempsec); */
  tempnano = (INT4)(1e9*(MJD.fracdays*86400.0 - (INT4)floor(MJD.fracdays*86400.0)));
 /*  printf("\ttempnano = %d\n",tempnano); */
  GPStemp.gpsSeconds = tempsec;
  GPStemp.gpsNanoSeconds = tempnano;

  /* define interval that is the magical number factor of 32.184 + 19 leap seconds to the */
  /* start of GPS time plus the TDB-TT correction and minus it from the input MJD */
  /* max absolute value of TDB-TT correction is < 0.184 sec */
  *GPS = GPStemp;
  XLALGPSAdd(GPS, -(51.184 + TDBtoTT));

} // TDBMJDtoGPS()

int
UTCMJDtoGPS ( LIGOTimeGPS *GPS, MJDTime MJD, INT4 leap )
{
  /* convert MJD to REAL8 for error checking */
  REAL8 MJDtemp = (REAL8)MJD.days + MJD.fracdays;

  /* Check not before the start of GPS time (MJD 44222) */
  XLAL_CHECK ( MJDtemp >= 44244., XLAL_EDOM, "Input time (%f) is not in range.\n", MJDtemp );

  /* convert MJDtime to GPS with no corrections (yet) */
  INT4 tempsec = (MJD.days-44244)*86400 + leap;
  tempsec += (INT4)floor(MJD.fracdays*86400.0);
  INT4 tempnano = (INT4)(1e9*(MJD.fracdays*86400.0 - (INT4)floor(MJD.fracdays*86400.0)));
  GPS->gpsSeconds = tempsec;
  GPS->gpsNanoSeconds = tempnano;

  return XLAL_SUCCESS;

} // UTCMJDtoGPS()

void
UTCGPStoMJD ( MJDTime *MJD, LIGOTimeGPS *GPS, INT4 leap)
{
  LIGOTimeGPS tempGPS;

  /* compute integer MJD days */
  MJD->days = 44244 + (INT4)floor(((REAL8)GPS->gpsSeconds + 1e-9*(REAL8)GPS->gpsNanoSeconds - (REAL8)leap)/86400.0);

  /* compute corresponding exact GPS for this integer days MJD */
  tempGPS.gpsSeconds = (MJD->days-44244)*86400 + leap;
  tempGPS.gpsNanoSeconds = 0;

  /* compute the difference in fractional MJD days */
  MJD->fracdays = XLALGPSDiff(GPS,&tempGPS) / 86400.0;
} // UTCGPStoMJD()

void
TDBGPStoMJD ( MJDTime *MJD, LIGOTimeGPS GPS, INT4 leap )
{
  MJDTime MJDrough;
  LIGOTimeGPS GPSguess;
  double diff;

  /* do rough conversion to MJD */
  UTCGPStoMJD(&MJDrough,&GPS,leap);

  do {

    /* convert rough MJD guess correctly back to GPS */
    TDBMJDtoGPS(&GPSguess,MJDrough);

    /* add difference to MJD */
    diff = XLALGPSDiff(&GPS,&GPSguess);
    AddIntervaltoMJD(diff,MJD,MJDrough);

    /* update current guess */
    MJDrough = *MJD;

  } while (diff >= 2e-9);

} // TDBGPStoMJD()

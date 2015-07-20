/* 2015 G.S. Davies

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


 * \author G.S. Davies
 * \brief
 * lalapps code to provide B_k given SFT data for given a set of pulsar
 * parameters. The Spectral Interpolation code is intended as a black box replacement for the existing
 * heterodyne process.
 */

#include "SpectralInterpolation.h"

/* Define some commonly used macros*/
#define INFTY 1./0.;
#define ROUND(x) (floor(x+0.5))
#define SINC(x)  (sin(LAL_PI*x)/(LAL_PI*x))

static BinaryPulsarParams empty_BinaryPulsarParams;
static LIGOTimeGPS empty_LIGOTimeGPS;
static SFTConstraints empty_SFTConstraints;


int main( int argc, char **argv ){

  InputParams inputParams;
  SplInterParams splParams;


  /* Take variables as input from command line */
  get_input_args(&inputParams, argc, argv);

  fprintf(stderr,"Starting Spectral Interpolation");

  /* if timing the code, then set up structures and start the clock */
  struct timeval timePreTot, timeEndTot;
  if(inputParams.Timing){
    fprintf(stderr,", outputting timing values.\n");
    gettimeofday(&timePreTot, NULL);
  }
  else fprintf(stderr,".\n");

  struct timeval timePreStart, timePreEnd;
  struct timeval timePulStart, timePulEnd;

  if(inputParams.Timing){
    gettimeofday(&timePreStart, NULL);
  }

  /* check input combinations and required parameters are set */
  if (inputParams.startF==0. || inputParams.endF==0.) {
    fprintf(stderr, "SFT start and end frequencies are required.\n");
    exit(1);
  }

  if (inputParams.pardirflag==0 && inputParams.parfileflag==0){
    XLALPrintError("No Pulsar directory or parfile specified\n");
    XLAL_ERROR(XLAL_EIO);
  }
  else if (inputParams.pardirflag==1 && inputParams.parfileflag==1){
    fprintf(stderr,"Pulsar directory and parfile specified. Using specified parfile.\n");
    inputParams.pardirflag=0;
  }

  if (inputParams.nameset==1 && inputParams.pardirflag==1){
    XLALPrintError("WARNING: Pulsar name and directory both set, specified pulsar name is useless.\n");
  }

  /* get data for position and orientation of interferometer */
  splParams.detector = *XLALGetSiteInfo( inputParams.ifo );

  if( inputParams.geocentre ){ /* set site to the geocentre if geoFlag is set */
    splParams.detector.location[0] = 0.0;
    splParams.detector.location[1] = 0.0;
    splParams.detector.location[2] = 0.0;
  }

  /* Initialise variables for use throughout */
  LIGOTimeGPS minStartTimeGPS = empty_LIGOTimeGPS;
  LIGOTimeGPS maxEndTimeGPS = empty_LIGOTimeGPS;
  CHAR *psrname=NULL;
  INT4 segcount=0, numSegs=0;
  INT4Vector *starts=NULL, *stops=NULL;
  char outputFilename[256];
  EmissionTime emit, emitS, emitE;
  CHAR timefileTDB[256], timefileTE405[256], sunfile200[256],
       earthfile200[256],sunfile405[256], earthfile405[256],
       sunfile414[256], earthfile414[256];
  EphemerisData *edat200=NULL,  *edat405=NULL, *edat414=NULL;
  TimeCorrectionData *tdatTDB=NULL,  *tdatTE405=NULL;

  /* Read in pulsar data either from directory or individual file. */
  /* (Possibly change this to use XLALFindFiles in the future) */
  struct dirent **pulsars;
  INT4 n = 0;
  if (inputParams.pardirflag==1 && inputParams.parfileflag==0){
    /* Scan pulsar directory for parfiles and read in the data */
    n=scandir(inputParams.pulsarDir, &pulsars, 0, alphasort);

    if ( n < 0){
      XLALPrintError("scandir failed\n");
      XLAL_ERROR(XLAL_EIO);
    }
  }


  UINT4 numnonpar = (UINT4)0; /* counts number of non-parfiles in directory */
  UINT4 numpulsars = (UINT4)n;
  UINT4 h=0;

  if (inputParams.pardirflag==0 && inputParams.parfileflag==1){
    numpulsars = 3;
  }


  CHAR parname[256];

  BinaryPulsarParams pulparams[numpulsars-2];

  /* set pulparams to be zero to begin with */
  /* using h=2 means that the "." and ".." directories are ignored */
  for(h=2; h<numpulsars; h++){
    pulparams[h-2] = empty_BinaryPulsarParams;
  }

  if (inputParams.pardirflag==1 && inputParams.parfileflag==0){
    XLALFree(pulsars[0]);
    XLALFree(pulsars[1]);
  }

  REAL8 barydInv[numpulsars];
  REAL8 startF=sqrt(-1), endF=sqrt(-1); /* Set to be NaN so that the first comparison always gives the source frequency */
  if(inputParams.Timing){ gettimeofday(&timePulStart, NULL);}



  FILE *fpout[numpulsars];

  for(h=2; h<numpulsars; h++){

    if (inputParams.pardirflag==1 && inputParams.parfileflag==0){
      /* ignores any files that don't end in ".par"*/
      if(strstr(pulsars[h]->d_name,".par") == NULL){
        XLALFree(pulsars[h]);
        numnonpar++;
        continue;
      }
    }

    /* read parfile */
    if (inputParams.pardirflag==1 && inputParams.parfileflag==0){
      sprintf(parname,"%s/%s",inputParams.pulsarDir, pulsars[h]->d_name);
    }
    else if (inputParams.pardirflag==0 && inputParams.parfileflag==1){
      sprintf(parname,"%s",inputParams.paramfile);
    }

    UINT4 ParNumber = h - numnonpar - 2;

    XLALReadTEMPOParFile( &pulparams[ParNumber], parname);

    if (pulparams[ParNumber].f0==empty_BinaryPulsarParams.f0 ){

      fprintf(stderr,"Parnumber %d out of %d\t",ParNumber,numpulsars-numnonpar-2);
      XLALPrintError("Error... Could not open source parameter file %s!", parname);
      return 0;
    }

    if (inputParams.pardirflag==1 && inputParams.parfileflag==0){
      /* ignore anything with frequency outside the SFT range */
      if ((pulparams[ParNumber].f0*inputParams.freqfactor)<inputParams.startF
           || (pulparams[ParNumber].f0*inputParams.freqfactor)>inputParams.endF){
        fprintf(stderr,"For pulsar %s, the GW frequency (%f Hz) lies outside of SFT boundaries (%f to %f Hz).\n",
                     pulparams[ParNumber].jname,pulparams[ParNumber].f0*inputParams.freqfactor, inputParams.startF,inputParams.endF);
        numnonpar++;
        continue;
      }
    }


    /* create pulsar name */
    if ( inputParams.nameset )
      psrname = XLALStringDuplicate(inputParams.PSRname);
    else if (pulparams[ParNumber].jname)
      psrname = XLALStringDuplicate(pulparams[ParNumber].jname);
    else if ( pulparams[ParNumber].bname )
      psrname= XLALStringDuplicate( pulparams[ParNumber].bname );
    else if ( pulparams[ParNumber].name )
      psrname= XLALStringDuplicate( pulparams[ParNumber].name );


    sprintf(outputFilename, "%s/SplInter_%s_%s",inputParams.outputdir,psrname,
      splParams.detector.frDetector.prefix);


    if( (fpout[ParNumber] = fopen(outputFilename,"w"))==NULL){
        fprintf(stderr, "Error... can't open output file %s!\n", outputFilename);
        return 0;
      }

    /* buffer the output, so that file system is not overloaded when outputing */
    if( setvbuf(fpout[ParNumber], NULL, _IOFBF, 0x10000) )
        fprintf(stderr, "Warning: Unable to set output file buffer!");



    startF=fmaxf(fminf(startF, pulparams[ParNumber].f0*inputParams.freqfactor),inputParams.startF+1.5);
    endF=fminf(fmaxf(endF, pulparams[ParNumber].f0*inputParams.freqfactor),inputParams.endF-1.5);


    /* If position epoch is not set but epoch is set position epoch to be epoch */
    if(pulparams[ParNumber].posepoch == 0. && pulparams[ParNumber].pepoch != 0.)
        pulparams[ParNumber].posepoch = pulparams[ParNumber].pepoch;

    /* Give distance to pulsar from parameter files to barycenter routine */
    /* Use parallax or distance value given - use parallax first. In 1/s. */
    if( pulparams[ParNumber].px != 0. ){
      barydInv[ParNumber] = ( 3600. / LAL_PI_180 )*pulparams[ParNumber].px /
        (LAL_C_SI*LAL_PC_SI/LAL_LYR_SI);
    }
    else if( pulparams[ParNumber].dist != 0. ){
      barydInv[ParNumber] = 1./(pulparams[ParNumber].dist*1e3*LAL_C_SI*LAL_PC_SI/LAL_LYR_SI);
    }
    else
      barydInv[ParNumber] = 0.; /* no parallax */

  }

  numpulsars = numpulsars-numnonpar-2; /* reset numpulsars to be the actual number of pulsars instead of number of files */


  if(inputParams.Timing) gettimeofday(&timePulEnd, NULL);
  /* if timing, then stop the clock on pulsar parameter load time. */

  if (inputParams.pardirflag==1 && inputParams.parfileflag==0){
    fprintf(stderr, "Number of Pulsars with frequencies within the SFT boundaries: %d\n", numpulsars);
  }


  /* load all types of ephemeris files, as uses little RAM/computational effort, and different parfiles may use a combination */
  sprintf(earthfile200, "%s/earth00-19-DE200.dat.gz",inputParams.ephemdir);
  sprintf(sunfile200, "%s/sun00-19-DE200.dat.gz",inputParams.ephemdir);
  sprintf(earthfile405, "%s/earth00-19-DE405.dat.gz",inputParams.ephemdir);
  sprintf(sunfile405, "%s/sun00-19-DE405.dat.gz",inputParams.ephemdir);
  sprintf(earthfile414, "%s/earth00-19-DE414.dat.gz",inputParams.ephemdir);
  sprintf(sunfile414, "%s/sun00-19-DE414.dat.gz",inputParams.ephemdir);

  edat200 = XLALMalloc(sizeof(*edat200));
  edat405 = XLALMalloc(sizeof(*edat405));
  edat414 = XLALMalloc(sizeof(*edat414));

  /* Load time correction files. */
  sprintf(timefileTDB,"%s/tdb_2000-2019.dat.gz",inputParams.ephemdir);
  sprintf(timefileTE405, "%s/te405_2000-2019.dat.gz",inputParams.ephemdir);

  /*  read in ephemeris files */
  edat200 = XLALInitBarycenter(earthfile200, sunfile200);
  edat405 = XLALInitBarycenter(earthfile405, sunfile405);
  edat414 = XLALInitBarycenter(earthfile414, sunfile414);

  /* check it worked */
  if (edat200==NULL || edat405==NULL || edat414==NULL){
    XLALPrintError("Error, could not read sun and earth ephemeris files - check that the ephemeris directory is correct");
    exit(1);
  }

  /* read in time correction files */
  tdatTDB = XLALInitTimeCorrections(timefileTDB);
  tdatTE405 = XLALInitTimeCorrections(timefileTE405);

  /* check it worked */
  if (tdatTDB==NULL || tdatTE405==NULL ){
    XLALPrintError("Error, could not read time correction ephemeris file - check that the ephemeris directory is correct");
    exit(1);
  }


  /* ------ Set up Science segment list ------ */
  /* Initialise memory for starts and stops vectors - beginning and end of segments */
  if( (starts = XLALCreateINT4Vector(1)) == NULL ||
      (stops = XLALCreateINT4Vector(1)) == NULL ){
    XLALPrintError("Error, allocating segment list memory.\n");
    exit(1);
  }

  if(pulparams[0].finishTime==0) pulparams[0].finishTime=INFTY; 
  /* Set finish time to infinity if default value is used */

  REAL8 segmentsStart = 0., segmentsEnd = 0;
  if (inputParams.pardirflag){
    /* If using directory, segment start and end are either input from command line of defaulted to be or all time */
    segmentsStart=inputParams.startTime;
    segmentsEnd = inputParams.endTime;
  }
  else {
    if ((inputParams.startTime > pulparams[0].finishTime && pulparams[0].finishTime !=0) || inputParams.endTime < pulparams[0].startTime){
      XLALPrintError("Specified start and end times are outside of the parameter file time range.");
      exit(1);
    }
    else {
      /* If times not specified, use times from parfile or if they don't exist, use all time */
      segmentsStart=fmaxf(pulparams[0].startTime, inputParams.startTime);
      segmentsEnd=fminf(inputParams.endTime, pulparams[0].finishTime);
    }
  }

  /* if the lenght of data is not enough for minimum segment length specified then exit */
  if(segmentsEnd-segmentsStart < inputParams.minSegLength){
    XLALPrintError("Applicable length of segment list, %.0f to %.0f is less than specified minimum duration of data, %.0f\n",segmentsStart, segmentsEnd, inputParams.minSegLength);
    exit(1);
  }

  numSegs = get_segment_list(starts, stops, inputParams.minSegLength,segmentsStart,segmentsEnd, inputParams.segfile);

  /* if no segments are longer than the minumum segment time set then exit */
  if( numSegs == 0 ){
    fprintf(stderr, "Applicable length of segment list is less than specified minimum duration of data.\n");
    exit(1);
  }


  /* Resize memory according to how many segments there are*/
  if( (starts = XLALResizeINT4Vector(starts, numSegs)) == NULL ||
      (stops = XLALResizeINT4Vector(stops, numSegs)) == NULL ){
    XLALPrintError("Error, re-allocating segment list memory.\n");
    exit(1);
  }

  /* stop clock for pre-loop things*/
  if(inputParams.Timing) gettimeofday(&timePreEnd, NULL);



  UINT4 TotalNSFTs = 0;
  REAL8 TotalInterTime = 0., TotalCatalogTime = 0., TotalLoadTime = 0.;
  /* --------------------------------------- */
  /* --- BIT THAT DOES THE INTERPOLATION --- */
  /* --------------------------------------- */
  /* Loop Through each segment */
  while(segcount<numSegs){


    UINT4 startt = 0, endt = 0;
    REAL8 baryAlpha[numpulsars], baryDelta[numpulsars];
    SFTConstraints constraints = empty_SFTConstraints;

    startt=starts->data[segcount];
    endt=stops->data[segcount];

    fprintf(stderr, "Segment %.0d-%.0d",startt,endt);

    /* If using directory, check whether the segment is included in any of the pulsar parameter file times*/
    /* relTime is simply whether ANY of the parfiles are relevant, if one is then the segment will be read*/
    if (inputParams.pardirflag){
      UINT4 relTime=0;
      for (h=0; h<numpulsars;h++){
        if (relTime==1) continue; /* if the segment has been flagged as relevant for any pulsar already, then continue */
        if ( pulparams[h].startTime!=0 && pulparams[h].finishTime !=0 ){ /* if parfile has start & end times specified */
          if ( (pulparams[h].startTime > endt) || (pulparams[h].finishTime < startt)) {
      /* Segment entirely below parfile bounds || segment entirely above parfile bounds */
            relTime=0.;
          }
          else relTime=1.; /* segment is relevant as (at least some) of it is within the parfile boundaries */
        }
        else relTime=1.; /* segment is relevant as the parfile is applicable forever and ever and ever */
      }
      for (h=0; h<numpulsars;h++){
        /* also for each pulsar, calculate alpha and delta as approx the same over the segment */
        REAL8 dtpos = 0.;

        dtpos = (startt+endt)/2. - pulparams[h].posepoch;
        /* calculate new sky position including proper motion */
        baryDelta[h] = pulparams[h].dec + dtpos*pulparams[h].pmdec;
        baryAlpha[h] = pulparams[h].ra + dtpos*pulparams[h].pmra/cos(baryDelta[h]);
      }
      if ( relTime == 0 ){
        segcount++;
        fprintf(stderr," is not relevant for any parfiles.\n");
        continue;
      }
    }

    if (inputParams.parfileflag){
      if ( pulparams[0].startTime!=0 && pulparams[0].finishTime !=0 ){ /* if parfile has start & end times specified */
        if ( (pulparams[0].startTime > endt) || (pulparams[0].finishTime < startt)) {
    /* Segment entirely below parfile bounds || segment entirely above parfile bounds */
          fprintf(stderr," is not relevant for specified parfile.\n");
          segcount++;
          continue;
        }
      }
      REAL8 dtpos = 0.;

      dtpos = (startt+endt)/2 - pulparams[0].posepoch;
      baryDelta[0] = pulparams[0].dec + dtpos*pulparams[0].pmdec;
      baryAlpha[0] = pulparams[0].ra + dtpos*pulparams[0].pmra/cos(baryDelta[0]);

    }

    struct timeval timeCatalogStart, timeCatalogEnd,timeLoadStart, timeLoadEnd,timeInterpolateStart, timeInterpolateEnd;
    REAL8 tInterpolate=0., tLoad=0., tCatalog=0.;

    /* Use segment times to set up constraints of times for SFTs  */
    XLALGPSSetREAL8( &minStartTimeGPS, (REAL8)startt );
    XLALGPSSetREAL8( &maxEndTimeGPS, (REAL8)endt );

    constraints.minStartTime = &minStartTimeGPS;
    constraints.maxStartTime = &maxEndTimeGPS;
    constraints.detector = inputParams.ifo;

    /* get full SFT-catalog of all matching SFTs within the segment */
    SFTCatalog *catalog = NULL;

    /* start catalogue load clock */
    if(inputParams.Timing) gettimeofday(&timeCatalogStart, NULL);

    /* Set up SFT cache filename and cache opening string */
    CHAR cacheFile[256], cacheFileCheck[256];
    if(inputParams.cacheDir){
      sprintf(cacheFile,"list:%s/Segment_%d-%d.sftcache",inputParams.filePattern,startt,endt);
      sprintf(cacheFileCheck,"%s/Segment_%d-%d.sftcache",inputParams.filePattern,startt,endt);
    }
    else{
      sprintf(cacheFile,"%s",inputParams.filePattern);
      CHAR *rmlist=inputParams.filePattern;
      rmlist+=5;
      sprintf(cacheFileCheck,"%s",rmlist);
    }

    /* Check SFT cache file exists and is not empty*/
    struct stat fileStat;
    if(stat(cacheFileCheck,&fileStat) < 0){
      segcount++;
      fprintf(stderr," SFT cache file %s does not exist.\n", cacheFileCheck);
      continue;
    }

    if(fileStat.st_size==0){
      segcount++;
      fprintf(stderr," SFT cache file %s is empty.\n", cacheFileCheck);
      continue;
    }

    catalog = XLALSFTdataFind(cacheFile, &constraints);

    if(inputParams.Timing){
      gettimeofday(&timeCatalogEnd, NULL);
      tCatalog  = (REAL8)timeCatalogEnd.tv_usec*1.e-6 + (REAL8)timeCatalogEnd.tv_sec - (REAL8)timeCatalogStart.tv_usec*1.e-6 - (REAL8)timeCatalogStart.tv_sec;
      TotalCatalogTime += tCatalog;
    }


    if ( !catalog ){
      XLALPrintError ("\nCATALOG READ ROUTINE FAILED\n");
      segcount++;
      continue;
    }

    /* check if catalog is empty - i.e. there are no SFTs from this segment in the location specified */
    if ( catalog->length == 0 ){
      XLALPrintError (" contains no matching SFTs\n");
      segcount++;
      XLALFree(catalog);
      continue;
    }

    /* add number of SFTs in current catalogue to total*/
    TotalNSFTs +=(catalog->length);


    SFTVector *SFTdat = NULL;
    UINT4 sftnum=0;
    REAL8 deltaf=0.;

    /* Check if the input end frequency is above the maximum frequency of the SFTs minus 1.5Hz,
       and if the input Start frequency is below the start of the SFT plus 1.5Hz*/

    endF=fminf(endF+1.5,(REAL8)(catalog->data->header.f0+catalog->data->header.deltaF*catalog->data->numBins)-1.5);
    startF=fmaxf(startF-1.5,(REAL8)(catalog->data->header.f0)+1.5);

    if(inputParams.Timing) gettimeofday(&timeLoadStart, NULL);
    /* Load the SFTs from the catalog */
    if (inputParams.pardirflag){
      /* If taking multiple pulsars input, load the entire SFT within the input boundaries */
      SFTdat = XLALLoadSFTs( catalog, startF, endF);

    }
    else if (inputParams.parfileflag){
     /* If taking single pulsar input, approximate the frequency and load a 3Hz band around this*/

      REAL8 tAv=0., tAv2=0., fAp=0.;

      /* Approximate the frequency for the time at the middle of the segment, */
      /* this doesn't take relative motion effects into account */

      tAv=((REAL8)endt+(REAL8)startt)/2. - pulparams[0].pepoch;

      tAv2=tAv*tAv;
      fAp=inputParams.freqfactor*(
               pulparams[0].f0
               + pulparams[0].f1*tAv
               + (1./2.)*pulparams[0].f2*tAv2
               + (1./6.)*pulparams[0].f3*tAv2*tAv
               + (1./24.)*pulparams[0].f4*tAv2*tAv2
               + (1./120.)*pulparams[0].f5*tAv2*tAv2*tAv
               + (1./720.)*pulparams[0].f6*tAv2*tAv2*tAv2
               + (1./5040.)*pulparams[0].f7*tAv2*tAv2*tAv2*tAv
               + (1./40320.)*pulparams[0].f8*tAv2*tAv2*tAv2*tAv2
               + (1./362880.)*pulparams[0].f9*tAv2*tAv2*tAv2*tAv2*tAv);


      /* Load the SFTs */
      SFTdat = XLALLoadSFTs( catalog, fAp-1.5, fAp+1.5);

    }

    /* stop sft load clock */
    if(inputParams.Timing){
      gettimeofday(&timeLoadEnd, NULL);
      tLoad  = (REAL8)timeLoadEnd.tv_usec*1.e-6 + (REAL8)timeLoadEnd.tv_sec - (REAL8)timeLoadStart.tv_usec*1.e-6 - (REAL8)timeLoadStart.tv_sec;
    }

    /* check that the SFT has loaded properly */
    if (SFTdat == NULL){
      XLALPrintError (" SFT data has not loaded properly");
      exit(1);
    }

    /* start interpolation clock */
    if(inputParams.Timing) gettimeofday(&timeInterpolateStart, NULL);

    /* Loop through each SFT */
    for (sftnum=0; sftnum<(SFTdat->length); sftnum++){

      /* deltaf is the frequency bin separation, this is used in a lot of the calculations as */
      /* deltaf = 1/(SFT length)*/
      deltaf=SFTdat->data->deltaF;
      REAL8 timestamp=0;
      timestamp=XLALGPSGetREAL8(&SFTdat->data[sftnum].epoch);


      for (h=0; h<numpulsars;h++){
        /* Check if the pulsar has been ruled out of analysis entirely previously */
        if (pulparams[h].f0==empty_BinaryPulsarParams.f0){
          continue;
        }

        /* Check if the pulsar parameter file given is not relevant for this SFT. */
        /* Due to previous checks, this will be the case when multiple parfiles  */
        /* are used but this SFT is only relevant for one or more of the others. */
        if ( pulparams[h].startTime!=0 && pulparams[h].finishTime !=0 ){
          if ( pulparams[h].startTime > timestamp || pulparams[h].finishTime < timestamp ){
            continue;
          }
        }

        /* Initialise variables whihc change for each SFT */
        REAL8 InterpolatedImagValue=0., InterpolatedRealValue=0.,
              UnnormalisedInterpolatedImagValue=0., UnnormalisedInterpolatedRealValue=0.,
              AbsSquaredWeightSum=0;
        INT4 datapoint=0;
        EarthState earth, earthS, earthE;
        REAL8 phaseShift = 0., fnew = 0., f1new = 0., sqrtf1new = 0.;
        REAL8 tdt = 0., tdt2 = 0.;
        REAL8 tdtS = 0., tdtS2 = 0.;
        REAL8 tdtE = 0., tdtE2 = 0.;
        EphemerisData *edat=NULL;
        TimeCorrectionData *tdat=NULL;
        TimeCorrectionType ttype;
        BarycenterInput baryInput, baryInputS, baryInputE;

        /* ------ Barycenter routine ------ */

        /* Timestamp needed in GPS format*/
        XLALGPSSetREAL8( &baryInput.tgps, timestamp +  1./(2.*deltaf) );

        /* set up location of detector */
        baryInput.site.location[0] = splParams.detector.location[0]/LAL_C_SI;
        baryInput.site.location[1] = splParams.detector.location[1]/LAL_C_SI;
        baryInput.site.location[2] = splParams.detector.location[2]/LAL_C_SI;

        /* get sky position and inverse distance from previously calculated values */
        baryInput.delta = baryDelta[h];
        baryInput.alpha = baryAlpha[h];
        baryInput.dInv = barydInv[h];

        baryInputE.delta = baryDelta[h];
        baryInputE.alpha = baryAlpha[h];
        baryInputE.dInv = barydInv[h];

        baryInputS.delta = baryDelta[h];
        baryInputS.alpha = baryAlpha[h];
        baryInputS.dInv = barydInv[h];

        /* check the time correction and ephemeris types*/
        if (pulparams[h].units!=NULL){
          if (!strcmp(pulparams[h].units, "TDB")) {
            tdat=tdatTDB;
            ttype=TIMECORRECTION_TDB;
          }
          else {
            tdat=tdatTE405;
            ttype=TIMECORRECTION_TCB;
          }
        }
        else{
          ttype=TIMECORRECTION_ORIGINAL;
          tdat=NULL;
        }

        /* create pulsar name */
        if ( inputParams.nameset )
          psrname = XLALStringDuplicate(inputParams.PSRname);
        else if (pulparams[h].jname)
          psrname = XLALStringDuplicate(pulparams[h].jname);
        else if ( pulparams[h].bname )
          psrname= XLALStringDuplicate( pulparams[h].bname );
        else if ( pulparams[h].name )
          psrname= XLALStringDuplicate( pulparams[h].name );

        /* set up which ephemeris type is being used for this source */
        if(pulparams[h].ephem!=empty_BinaryPulsarParams.ephem) {
          if (!strcmp(pulparams[h].ephem, "DE405")) { edat=edat405; }
          else if (!strcmp(pulparams[h].ephem, "DE414")) { edat=edat414; }
          else if (!strcmp(pulparams[h].ephem, "DE200")) { edat=edat200; }
        }
        else{
          edat=edat405; /* default is that DE405 is used*/
        }

        /* Perform Barycentering Routine at middle of SFT, for calculation of phase shift and frequency */
        XLALBarycenterEarthNew(&earth, &baryInput.tgps, edat, tdat, ttype);
        XLALBarycenter(&emit, &baryInput, &earth);

        /* Also erform Barycentering Routine at start and end of SFT, for calculation of frequency derivatives */

        XLALGPSSetREAL8( &baryInputE.tgps, timestamp +  1./(deltaf) );
        XLALGPSSetREAL8( &baryInputS.tgps, timestamp );

        XLALBarycenterEarthNew(&earthE, &baryInputE.tgps, edat, tdat, ttype);
        XLALBarycenterEarthNew(&earthS, &baryInputS.tgps, edat, tdat, ttype);

        XLALBarycenter(&emitE, &baryInputE, &earthE);
        XLALBarycenter(&emitS, &baryInputS, &earthS);

        /* If the 'detector at barycenter' flag is set, then reset emit to appropriate values */
        if (inputParams.baryFlag){
          emit.tDot=1.;
          emit.deltaT=0.;
          emitS.tDot=1.;
          emitS.deltaT=0.;
          emitE.tDot=1.;
          emitE.deltaT=0.;
        }

        REAL8 totaltDot = 0., totaltDotstart = 0.,totaltDotend = 0.;
        REAL8 totaldeltaT = 0., totaldeltaTstart = 0.,totaldeltaTend = 0.;

        /* Get binary pulsar corrections if source is a binary pulsar */
        /* because XLALBinaryPulsarDeltaT only returns delta t, not tdot, we need to */
        /* calculate above and below the start and end frequencies to find gradient. */

        /* These are denoted for start(S)/end(E) and plus(P)/minus(M) 1 second.  */

        if( pulparams[h].model!=empty_BinaryPulsarParams.model ){
          BinaryPulsarInput binInputS, binInputM, binInputE, binInputSP, binInputSM, binInputEP, binInputEM;
          BinaryPulsarOutput binOutputS, binOutputM, binOutputE, binOutputSP, binOutputSM, binOutputEP, binOutputEM;

          /* set up times used for binary input */
          binInputS.tb = timestamp + emitS.deltaT;
          binInputE.tb = timestamp + 1./deltaf + emitE.deltaT;
          binInputM.tb = timestamp + 1./(2.*deltaf) + emit.deltaT;

          /* Make assumption that emitS, emitE earthS and earthE are constant over two seconds */
          binInputSM.tb = timestamp + emitS.deltaT-1;
          binInputSP.tb = timestamp + emitS.deltaT+1;
          binInputEM.tb = timestamp + 1./deltaf + emitE.deltaT-1;
          binInputEP.tb = timestamp + 1./deltaf + emitE.deltaT+1;

          binInputS.earth = earthS;
          binInputE.earth = earthE;
          binInputM.earth = earth;

          binInputSP.earth = earthS;
          binInputEP.earth = earthE;
          binInputSM.earth = earthS;
          binInputEM.earth = earthE;

          XLALBinaryPulsarDeltaT( &binOutputS, &binInputS, &pulparams[h] );
          XLALBinaryPulsarDeltaT( &binOutputE, &binInputE, &pulparams[h] );
          XLALBinaryPulsarDeltaT( &binOutputM, &binInputM, &pulparams[h] );

          XLALBinaryPulsarDeltaT( &binOutputSM, &binInputSM, &pulparams[h] );
          XLALBinaryPulsarDeltaT( &binOutputEM, &binInputEM, &pulparams[h] );
          XLALBinaryPulsarDeltaT( &binOutputSP, &binInputSP, &pulparams[h] );
          XLALBinaryPulsarDeltaT( &binOutputEP, &binInputEP, &pulparams[h] );


          /* Add the barycentering and binary terms together to get total deltat */
          totaldeltaT = emit.deltaT + binOutputM.deltaT;
          totaldeltaTstart = emitS.deltaT + binOutputS.deltaT;
          totaldeltaTend = emitE.deltaT + binOutputE.deltaT;

          /* Add the barycentering and binary terms together to get total tdot */
          totaltDot = emit.tDot + (binOutputE.deltaT-binOutputS.deltaT)*deltaf;
          totaltDotend = emitE.tDot + (binOutputEP.deltaT-binOutputEM.deltaT)/2;
          totaltDotstart = emitS.tDot + (binOutputSP.deltaT-binOutputSM.deltaT)/2;

        }
        else{
          /* If not a binary, then relative motion effects are only due to detector motion */
          totaldeltaT = emit.deltaT;
          totaldeltaTend = emitE.deltaT;
          totaldeltaTstart = emitS.deltaT;

          totaltDot = emit.tDot;
          totaltDotend = emitE.tDot;
          totaltDotstart = emitS.tDot;
        }


        /* Calculate relevant time difference to epoch for use in calculations */
        tdt=timestamp - pulparams[h].pepoch + 1./(2.*deltaf) + totaldeltaT;
        tdtS=timestamp - pulparams[h].pepoch + totaldeltaTstart;
        tdtE=timestamp - pulparams[h].pepoch + 1./(2.*deltaf) + totaldeltaTend ;

        /* SFT start time - parfile epoch + 1/2 SFT length + barycentre timeshift */

        tdt2=tdt*tdt;  /* tdt^2 for /slightly/ faster computation */
        tdtE2=tdtE*tdtE;  /* tdt^2 for /slightly/ faster computation */
        tdtS2=tdtS*tdtS;  /* tdt^2 for /slightly/ faster computation */

        /* calculate frequency at the centre of the SFT */
        fnew=inputParams.freqfactor*totaltDot*(
               pulparams[h].f0
               + pulparams[h].f1*tdt
               + (1./2.)*pulparams[h].f2*tdt2
               + (1./6.)*pulparams[h].f3*tdt*tdt2
               + (1./24.)*pulparams[h].f4*tdt2*tdt2
               + (1./120.)*pulparams[h].f5*tdt2*tdt2*tdt
               + (1./720.)*pulparams[h].f6*tdt2*tdt2*tdt2
               + (1./5040.)*pulparams[h].f7*tdt2*tdt2*tdt2*tdt
               + (1./40320.)*pulparams[h].f8*tdt2*tdt2*tdt2*tdt2
               + (1./362880.)*pulparams[h].f9*tdt2*tdt2*tdt2*tdt2*tdt2);

        if((fnew < inputParams.startF) || (fnew > inputParams.endF)){
          fprintf(stderr,"Pulsar %s has frequency %.4f outside of the frequency range %f-%f at time %.0f\n",
              psrname, fnew, inputParams.startF, inputParams.endF, timestamp);
          continue;
        }


        /* Calculate difference in phase between beginning of SFT and the epoch.  */
        /* Using fmod means that phaseShift is always between 0 and 2 PI, so that */
        /* inaccuracies in cos and sin functions are minimised. */
        phaseShift = 2.*LAL_PI*fmod(inputParams.freqfactor*(
               pulparams[h].f0*tdt
               + (1./2.)*pulparams[h].f1*tdt2
               + (1./6.)*pulparams[h].f2*tdt2*tdt
               + (1./24.)*pulparams[h].f3*tdt2*tdt2
               + (1./120.)*pulparams[h].f4*tdt2*tdt2*tdt
               + (1./720.)*pulparams[h].f5*tdt2*tdt2*tdt2
               + (1./5040.)*pulparams[h].f6*tdt2*tdt2*tdt2*tdt
               + (1./40320.)*pulparams[h].f7*tdt2*tdt2*tdt2*tdt2
               + (1./362880.)*pulparams[h].f8*tdt2*tdt2*tdt2*tdt2*tdt
               + (1./3628800.)*pulparams[h].f9*tdt2*tdt2*tdt2*tdt2*tdt2),1);

        /* calculate frequency at start of SFT */
        REAL8 fstart = inputParams.freqfactor*totaltDotstart*(
               pulparams[h].f0
               + pulparams[h].f1*tdtS
               + (1./2.)*pulparams[h].f2*tdtS2
               + (1./6.)*pulparams[h].f3*tdtS*tdtS2
               + (1./24.)*pulparams[h].f4*tdtS2*tdtS2
               + (1./120.)*pulparams[h].f5*tdtS2*tdtS2*tdtS
               + (1./720.)*pulparams[h].f6*tdtS2*tdtS2*tdtS2
               + (1./5040.)*pulparams[h].f7*tdtS2*tdtS2*tdtS2*tdtS
               + (1./40320.)*pulparams[h].f8*tdtS2*tdtS2*tdtS2*tdtS2
               + (1./362880.)*pulparams[h].f9*tdtS2*tdtS2*tdtS2*tdtS2*tdtS2);

        /* calculate frequency at end of SFT */
        REAL8 fend = inputParams.freqfactor*totaltDotend*(
               pulparams[h].f0
               + pulparams[h].f1*tdtE
               + (1./2.)*pulparams[h].f2*tdtE2
               + (1./6.)*pulparams[h].f3*tdtE*tdtE2
               + (1./24.)*pulparams[h].f4*tdtE2*tdtE2
               + (1./120.)*pulparams[h].f5*tdtE2*tdtE2*tdtE
               + (1./720.)*pulparams[h].f6*tdtE2*tdtE2*tdtE2
               + (1./5040.)*pulparams[h].f7*tdtE2*tdtE2*tdtE2*tdtE
               + (1./40320.)*pulparams[h].f8*tdtE2*tdtE2*tdtE2*tdtE2
               + (1./362880.)*pulparams[h].f9*tdtE2*tdtE2*tdtE2*tdtE2*tdtE2);


        /* calculate effective fdot including the frequency derivative obtained from the barycentering */
        f1new= (fend - fstart)*deltaf;

        sqrtf1new = sqrt(fabs(f1new)); /* square root of the absolute value of f1 for use in calculations */
        REAL8 signf1new = f1new/fabs(f1new); /* sign of f1 for use in calculations */

        UINT4 dataLength = 0, dpNum = 0;

        /* Load data into RAM as will be reusing multiple times */
        dataLength = (UINT4)(ROUND((fnew+inputParams.bandwidth/2.-SFTdat->data->f0)/deltaf
                          -(fnew-inputParams.bandwidth/2.-SFTdat->data->f0)/deltaf));

        /* check data load */
        if(dataLength<1){
          XLALPrintError("Error setting length of data");
          continue;
        }

        /* initialise and create various vectors for use in calculations */
        REAL8Vector *dataFreqs = NULL, *ReDp = NULL, *ImDp = NULL, *ReMu = NULL, *ImMu = NULL, *MuMu = NULL;
        UINT4Vector *dpUsed = NULL;

        if( ((dataFreqs = XLALCreateREAL8Vector(dataLength)) == NULL ) || /* datapoint frequency value*/
          ((ReDp = XLALCreateREAL8Vector(dataLength)) == NULL ) || /* Data Real Value*/
          ((ImDp = XLALCreateREAL8Vector(dataLength)) == NULL ) || /* Data Imag Value*/
          ((ReMu = XLALCreateREAL8Vector(dataLength)) == NULL ) || /* Real Value of model used to calculate least squares*/
          ((ImMu = XLALCreateREAL8Vector(dataLength)) == NULL ) || /* Imag Value of model used to calculate least squares*/
          ((MuMu = XLALCreateREAL8Vector(dataLength)) == NULL ) || /* Normalisation factor  - sum of squares of model , with some constants that can be cancelled*/
          ((dpUsed = XLALCreateUINT4Vector(dataLength)) == NULL) ){ /* Whether the datapoint is used or not - residual removal etc */
          XLALPrintError("Error, allocating data memory.\n");
          exit(1);
        }


        REAL8 ReStdDev = 0., ImStdDev = 0., StdDev = 0., ReStdDevSum = 0., ImStdDevSum = 0.;

        /* Load the data into the ReDp, ImDp vectors and set dataFreqs */
        /* While doing this, add the square of each in order to obtain the std deviation of the data */
        for(datapoint=ROUND((fnew-inputParams.bandwidth/2.-SFTdat->data->f0)/deltaf);
                   datapoint<= ROUND((fnew+inputParams.bandwidth/2.-SFTdat->data->f0)/deltaf); datapoint++ ){
          dpNum  = (UINT4)(datapoint-ROUND((fnew-inputParams.bandwidth/2.-SFTdat->data->f0)/deltaf));

          ReDp->data[dpNum] = creal(SFTdat->data[sftnum].data->data[datapoint]);

          ImDp->data[dpNum] = cimag(SFTdat->data[sftnum].data->data[datapoint]);

          dataFreqs->data[dpNum] = SFTdat->data->f0+datapoint*deltaf;

          ReStdDevSum += (ReDp->data[dpNum])*(ReDp->data[dpNum]);

          ImStdDevSum += (ImDp->data[dpNum])*(ImDp->data[dpNum]);

        } /* close datapoint loop*/

        /* Obtain the std deviation combined from Real and Imaginary parts of SFT */

        ReStdDev = sqrt(ReStdDevSum/(dataLength-1));
        ImStdDev = sqrt(ImStdDevSum/(dataLength-1));

        StdDev = sqrt(ReStdDev*ReStdDev + ImStdDev*ImStdDev)/2;


        /* if removing outliers, remove any data point whihc is outside of closest 10 datapoints with real or */
        /* imaginary parts above threshold number of standard deviations. */
        for(dpNum = 0; dpNum< dataFreqs->length; dpNum++){
          if( ((fabs(ReDp->data[dpNum]) < 2*inputParams.stddevthresh*StdDev &&
              fabs(ImDp->data[dpNum]) < 2*inputParams.stddevthresh*StdDev) || /* Keep if datapoint is within 4 std devs */
              inputParams.stddevthresh==0) || /* keep if threshold not set */
              ( fabs(dataFreqs->data[dpNum]-fnew) < 5*deltaf)  ){ /* ensure keeping closest 10 points to fnew to keep signals */
            dpUsed->data[dpNum] = 1;
          }
          else{
            /* If not within these limits, set datapoint as not used. */
            dpUsed->data[dpNum] = 0;
          }
        } /* close datapoint loop*/

        /* Data has now been loaded into ReDp/ImDp and dpUsed set to remove initial outliers */

        /* Set values of Model used for interpolation */
        /* Check if there is a significant signal spread over frequencies over the course of the SFT */
        if(fabs(f1new)/(deltaf*deltaf)>0.1){ /*i.e. if signal is spread more than 0.1 bins over the course of the SFT */
          for(dpNum=0;dpNum<dataLength;dpNum++ ){

            if(dpUsed->data[dpNum] == 1){
              REAL8  FresPlusC = 0., FresMinC = 0.,FresPlusS = 0., FresMinS = 0. , delPhase = 0.;

              delPhase = phaseShift-LAL_PI*(fnew-dataFreqs->data[dpNum])*(fnew-dataFreqs->data[dpNum])/f1new
                                   -LAL_PI*(dataFreqs->data[dpNum])/deltaf;

              /* calculate fresnel integrals for start and end of SFT */
              XLALFresnel(&FresPlusC,&FresPlusS,((fnew-dataFreqs->data[dpNum])*LAL_SQRT2*sqrtf1new/f1new+sqrtf1new/LAL_SQRT2/deltaf));
              XLALFresnel(&FresMinC,&FresMinS,((fnew-dataFreqs->data[dpNum])*LAL_SQRT2*sqrtf1new/f1new-sqrtf1new/LAL_SQRT2/deltaf));

              ReMu->data[dpNum] = 1/(LAL_SQRT2*sqrtf1new)*(cos(delPhase)*(FresPlusC-FresMinC)
                                       -signf1new*sin(delPhase)*(FresPlusS-FresMinS));

              ImMu->data[dpNum] = 1/(LAL_SQRT2*sqrtf1new)*(sin(delPhase)*(FresPlusC-FresMinC)
                                       +signf1new*cos(delPhase)*(FresPlusS-FresMinS));

              MuMu->data[dpNum] = 1/(2*fabs(f1new))*((FresPlusC-FresMinC)*(FresPlusC-FresMinC)
                                                        +(FresPlusS-FresMinS)*(FresPlusS-FresMinS));
            } /* close if datapoint used statement */

          } /* close datapoint loop */

        }
        else{/* signal is not spread */

          /* Check if calculated pulsar frequency lies on an bin or within 0.1% (SFT bin value is 99.9998% of true value) */
          /* of the SFT to a frequency bin.*/
          /* If it does, just perform the phaseshift, including phase shift from change in frequency. */
          /* (This is essentially to avoid any "divide by zero" problems) */

          if(fmod((fnew-SFTdat->data->f0),deltaf)<(0.01*deltaf) ||
                   fmod((fnew-SFTdat->data->f0),deltaf)>(0.99*deltaf)){
            for(dpNum=0;dpNum<dataLength;dpNum++ ){

              if( dpNum == (UINT4)( ROUND( inputParams.bandwidth/(2.*deltaf) ) ) ){
                ReMu->data[dpNum] = cos(phaseShift-LAL_PI*dataFreqs->data[dpNum]/deltaf)/deltaf;
                ImMu->data[dpNum] = sin(phaseShift-LAL_PI*dataFreqs->data[dpNum]/deltaf)/deltaf;
                MuMu->data[dpNum] = 1/(deltaf*deltaf);
              }/* close 'if on signal frequency bin' statement */
              else{
                ReMu->data[dpNum] = 0;
                ImMu->data[dpNum] = 0;
                MuMu->data[dpNum] = 0;
              } /* close 'else a different bin' statement */

            } /* close 'for each datapoint' loop*/

          } /* close 'if on a bin' statement */
          else{ /* signal is not on a bin - interpolate with a sinc*/

            for(dpNum=0;dpNum<dataLength;dpNum++ ){

              if(dpUsed->data[dpNum] == 1){ /* if the datapoint is not an outlier */

                ReMu->data[dpNum] = 1/(deltaf)*SINC((dataFreqs->data[dpNum]-fnew)/deltaf)
                                      *cos(phaseShift-LAL_PI*dataFreqs->data[dpNum]/deltaf);

                ImMu->data[dpNum] = 1/(deltaf)*SINC((dataFreqs->data[dpNum]-fnew)/deltaf)
                                      *sin(phaseShift-LAL_PI*dataFreqs->data[dpNum]/deltaf);

                MuMu->data[dpNum] = 1./(deltaf*deltaf)*SINC((dataFreqs->data[dpNum]-fnew)/deltaf)
                                                          *SINC((dataFreqs->data[dpNum]-fnew)/deltaf);

              } /* close 'if the datapoint is not an outlier' statement */

            } /* close datapoint loop */

          } /* close 'else interpolate with a sinc' statement */

        } /* close 'else is not spread' statement */

        /* At this oint we have now loaded the data, set the model and performed the first outlier removal */
        /* now we calculate Bk and sigmak, and perform residual outlier removal */

        /* initialise and create residual and signal best estimate vectors*/
        REAL8Vector *ReResiduals = NULL, *ImResiduals = NULL, *ReHf = NULL, *ImHf = NULL;

        if( ((ReResiduals = XLALCreateREAL8Vector(dataLength)) == NULL ) || /* Value of residuals in Real part of SFT */
            ((ImResiduals = XLALCreateREAL8Vector(dataLength)) == NULL ) || /* Value of residuals in Imag part of SFT **/
            ((ReHf = XLALCreateREAL8Vector(dataLength)) == NULL ) || /* Expected Value in Real part of SFT given interpolated Bk */
            ((ImHf = XLALCreateREAL8Vector(dataLength)) == NULL ) ){ /* Expected Value in Imag part of SFT given interpolated Bk  */
          XLALPrintError("Error, allocating residuals memory.\n");
          exit(1);
        }


        /* Residual noise value needs to be initialised outside the while loop as it will be outputted */
        REAL8 ResStdDev = 0.;


        /* repeat the following section until all outliers have been removed */
        UINT4 numReduced = 1; /* i.e. flag to say that the number of datapoints has been reduced */


        /* Iterate through Bk and sigmak calculation, and residual outlier removal so that */
        /* Bk and sigmak are recalculated if any of the datapoints have been removed. */
        while(numReduced){
          UINT4 dpCountStart = 0, dpCountEnd = 0;
          /* count the number of datapoints being used to begin with */
          for(dpNum=0; dpNum<dataLength; dpNum++ ){
            dpCountStart += dpUsed->data[dpNum];
          }

          /* Perform least squares fit*/
          for(dpNum=0;dpNum<dataLength;dpNum++ ){

            if(dpUsed->data[dpNum] == 1){ /* use only datapoints which have not been removed */

              REAL8  ReVal = 0., ImVal = 0;

              ReVal=ReDp->data[dpNum]*ReMu->data[dpNum]+ImDp->data[dpNum]*ImMu->data[dpNum];
              ImVal=ImDp->data[dpNum]*ReMu->data[dpNum]-ReDp->data[dpNum]*ImMu->data[dpNum];
              AbsSquaredWeightSum += MuMu->data[dpNum];

              UnnormalisedInterpolatedRealValue+=ReVal;
              UnnormalisedInterpolatedImagValue+=ImVal;

            } /* close if datapoint is used statement */

          }/* close datapoint loop*/

          /* Combine sums to find interpolated values*/
          InterpolatedRealValue=UnnormalisedInterpolatedRealValue/AbsSquaredWeightSum;
          InterpolatedImagValue=UnnormalisedInterpolatedImagValue/AbsSquaredWeightSum;

          /* Reset the following to zero in case this is the (>1)th loop through the while statement */
          REAL8 ResReStdDev = 0., ResImStdDev = 0., ResReStdDevSum = 0., ResImStdDevSum = 0.;

          /* Residual cleaning */
          for(dpNum=0;dpNum<dataLength;dpNum++ ){

            if(dpUsed->data[dpNum] == 1){

              /* calculate estimated signal in SFT from B_k and model */
              ReHf->data[dpNum]=InterpolatedRealValue*ReMu->data[dpNum]-InterpolatedImagValue*ImMu->data[dpNum];
              ImHf->data[dpNum]=InterpolatedImagValue*ReMu->data[dpNum]+InterpolatedRealValue*ImMu->data[dpNum];

              /* take this from the atual SFT to obtain residuals */
              ReResiduals->data[dpNum] = ReDp->data[dpNum] - ReHf->data[dpNum];
              ImResiduals->data[dpNum] = ImDp->data[dpNum] - ImHf->data[dpNum];

              ResReStdDevSum += ReResiduals->data[dpNum]*ReResiduals->data[dpNum];
              ResImStdDevSum += ImResiduals->data[dpNum]*ImResiduals->data[dpNum];

            } /* close if datapoint is used statement */

          }/* close datapoint loop*/

          /* calculate standard deviation of the residuals - average or real and imag parts */
          ResReStdDev = sqrt(ResReStdDevSum/(dpCountStart-1));
          ResImStdDev = sqrt(ResImStdDevSum/(dpCountStart-1));

          /* calculate combined standard deviation */
          ResStdDev = sqrt(ResReStdDev*ResReStdDev+ResImStdDev*ResImStdDev)/2;


          /* Use residual standard deviation to remove datapoints - always keep closest 4 datapoints */
          for(dpNum = 0; dpNum < dataFreqs->length; dpNum++){

            if(dpUsed->data[dpNum] == 1){ /* dont change currently unused points */

              if( (fabs(ReResiduals->data[dpNum]) < inputParams.stddevthresh*(ResStdDev) &&
                  fabs(ImResiduals->data[dpNum]) < inputParams.stddevthresh*(ResStdDev)) || /* Keep if datapoint is within std devs */
                   inputParams.stddevthresh==0 || /* keep if threshold not set */
                  ( fabs(dataFreqs->data[dpNum]-fnew) < 2*deltaf)  ){ /* ensure keeping closest 4 points to fnew to keep signals */
                dpUsed->data[dpNum] = 1;
              }
              else{
                dpUsed->data[dpNum] = 0;
              }
            }/* close if datapoint is used loop*/

          } /* close datapoint loop*/

          /* count the number of datapoints being used after cleaning  */
          for(dpNum=0; dpNum<dataLength; dpNum++ ){
            dpCountEnd += dpUsed->data[dpNum];
          }


          if(dpCountStart == dpCountEnd) numReduced = 0; /* if no data points have been removed, exit the while loop */

        } /* Close outlier removal while loop */

        /* Destroy data and model structures to prevent memory leak */
        XLALDestroyREAL8Vector(ReResiduals);
        XLALDestroyREAL8Vector(ImResiduals);
        XLALDestroyREAL8Vector(dataFreqs);
        XLALDestroyREAL8Vector(ReDp);
        XLALDestroyREAL8Vector(ImDp);
        XLALDestroyREAL8Vector(ReMu);
        XLALDestroyREAL8Vector(ImMu);
        XLALDestroyREAL8Vector(MuMu);
        XLALDestroyREAL8Vector(ReHf);
        XLALDestroyREAL8Vector(ImHf);
        XLALDestroyUINT4Vector(dpUsed);

        /* print time, ReBk, ImBk and noise estimate to output file*/
        fprintf(fpout[h],"%.0f\t%.6e\t%.6e\t%.6e\n",timestamp+1./(2.*deltaf),InterpolatedRealValue
           ,InterpolatedImagValue, ResStdDev*deltaf*LAL_SQRT2);


      } /* close pulsar loop */

    }/* close SFTnum loop */
    if(inputParams.Timing){
      gettimeofday(&timeInterpolateEnd, NULL);
      tInterpolate  = (REAL8)timeInterpolateEnd.tv_usec*1.e-6 + (REAL8)timeInterpolateEnd.tv_sec - (REAL8)timeInterpolateStart.tv_usec*1.e-6 - (REAL8)timeInterpolateStart.tv_sec;
    }
    /* destroy SFT catalogue to prevent memory leak*/
    XLALDestroySFTCatalog(catalog);
    /* print timing information to stderr */
    if(inputParams.Timing){
      fprintf(stderr, " done. Number of SFTs = %d. Obtaining Catalog took %.5fs, SFT Load took %.5fs, Interpolation took %.5fs.\n",
         (INT4)(SFTdat->length), tCatalog, tLoad, tInterpolate );
      TotalInterTime += tInterpolate;
      TotalLoadTime += tLoad;
    }
    else{
      fprintf(stderr," done.\n");
    }

    XLALDestroySFTVector(SFTdat);
    segcount++; /* move to next segment */
  }/* close segment loop */

  /* close all the output files */
  for(h=0;h<numpulsars;h++) fclose(fpout[h]);


  REAL8 tPre =0. , tPul =0.;
  if(inputParams.Timing){
    tPre  = (REAL8)timePreEnd.tv_usec*1.e-6 + (REAL8)timePreEnd.tv_sec - (REAL8)timePreStart.tv_usec*1.e-6 - (REAL8)timePreStart.tv_sec;
    tPul  = (REAL8)timePulEnd.tv_usec*1.e-6 + (REAL8)timePulEnd.tv_sec - (REAL8)timePulStart.tv_usec*1.e-6 - (REAL8)timePulStart.tv_sec;
  }

  fprintf(stderr,"Spectral Interpolation Complete.\n");

  /* print timing information */
  if(inputParams.Timing) fprintf(stderr,":\nPre-Interpolation things took %.5fs, of which source parameter load time was %.5fs\nTotal Number of SFTs = %d\nTotal catalog time %.5fs\nTotal SFT Load time %.5fs\nTotal Interpolation time %.5fs.\n",tPre,tPul, TotalNSFTs, TotalCatalogTime, TotalLoadTime, TotalInterTime);


  if(inputParams.stddevthresh != 0){ /* do the Bk outlier removal routine if the threshold has been set */
    fprintf(stderr,"\nPerforming Outlier Removal routine with noise threshold of %.5f",inputParams.stddevthresh);
    struct timeval timeORStart, timeOREnd;
    if(inputParams.Timing) gettimeofday(&timeORStart, NULL);

    for(h=0;h<numpulsars;h++){

      char BkFilename[256];
      CHAR *parName=NULL;
      /* -------- Set up file for output -------- */
      /* create filename */
      if ( inputParams.nameset ) parName=inputParams.PSRname;
      else if ( pulparams[h].jname ) parName = XLALStringDuplicate( pulparams[h].jname);
      else if ( pulparams[h].bname ) parName = XLALStringDuplicate( pulparams[h].bname );
      else if ( pulparams[h].name ) parName= XLALStringDuplicate( pulparams[h].name );
      else{
        fprintf(stderr, "No pulsar name specified\n");
        exit(1);
      }
      sprintf(BkFilename, "%s/SplInter_%s_%s",inputParams.outputdir ,parName,
            splParams.detector.frDetector.prefix);

      FILE *BkFile=NULL;
      BkFile = fopen(BkFilename,"r");
      if(BkFile == NULL){
        XLALPrintError("Error opening file");
      }

      long offset;
      CHAR jnkstr[256]; /* junk string to contain comment lines */

      INT4Vector *timeStamp=NULL;
      REAL8Vector *ReData=NULL, *ImData=NULL, *NData=NULL;

      /* Make structures for storing data */
      timeStamp = XLALCreateINT4Vector( TotalNSFTs );
      ReData = XLALCreateREAL8Vector( TotalNSFTs );
      ImData = XLALCreateREAL8Vector( TotalNSFTs );
      NData = XLALCreateREAL8Vector( TotalNSFTs );

      /* Read in data */
      UINT4 k=0;
      while(!feof(BkFile) && k < TotalNSFTs){
        offset = ftell(BkFile); /* get current position of file stream */

        if(fscanf(BkFile, "%s", jnkstr) == EOF) /* scan in value and check if == to # */
          break; /* break if there is an extra line at the end of file containing
                    nothing */

        fseek(BkFile, offset, SEEK_SET); /* if line doesn't start with a # then it is
                                      data */
        if( fscanf(BkFile, "%d%lf%lf%lf", &timeStamp->data[k], &ReData->data[k],
                   &ImData->data[k],&NData->data[k]) == EOF ) break;
        k++;
      }

      fclose(BkFile);

      FILE *BkFileOut=NULL;
      BkFileOut = fopen(BkFilename,"w");
      UINT4 p=0;
      UINT4 startlen=ReData->length;
      REAL8 meanNoiseEst = 0, meanAbsValue = 0.;

      /* Set up file buffer so that file system is not overloaded on output */
      /* buffer will be 1Mb */
      if( setvbuf(BkFileOut, NULL, _IOFBF, 0x10000) )
        fprintf(stderr, "Warning: Unable to set output file buffer!");

      for(p=0;p<NData->length;p++){
        meanAbsValue += sqrt(ReData->data[p]*ReData->data[p]+ImData->data[p]*ImData->data[p])/NData->length;
        meanNoiseEst += NData->data[p]/NData->length;
      }

      /* remove all datapoints with Re(Bk),Im(Bk) or SigK more than stddevthresh */
     UINT4 j=0; /* to count the length of the new data, to resize later */
      for(p=0;p<startlen;p++){
        if(fabs(ReData->data[p]) < meanAbsValue*inputParams.stddevthresh &&
           fabs(ImData->data[p]) < meanAbsValue*inputParams.stddevthresh &&
           NData->data[p] < meanNoiseEst*inputParams.stddevthresh){
          ReData->data[j] = ReData->data[p];
          ImData->data[j] = ImData->data[p];
          NData->data[j] = NData->data[p];
          timeStamp->data[j] = timeStamp->data[p];
          j++;
        }
      }
      if(j!=startlen){ /* if any datapoints have been removed, resize the vectors and rerun outlier removal */
        if( (ReData = XLALResizeREAL8Vector(ReData, j)) == NULL ||
            (ImData = XLALResizeREAL8Vector(ImData, j)) == NULL ||
            (NData = XLALResizeREAL8Vector(NData, j)) == NULL ||
            (timeStamp = XLALResizeINT4Vector(timeStamp, j)) == NULL )
          {  XLALPrintError("Error resizing thresholded data.\n");  }
        for(p=0;p<NData->length;p++){
          meanNoiseEst += NData->data[p]/NData->length;
          meanAbsValue += sqrt(ReData->data[p]*ReData->data[p]+ImData->data[p]*ImData->data[p])/NData->length;
        }

        j=0; /* reset counters */
        for(p=0;p<ReData->length;p++){
          if(fabs(ReData->data[p]) < meanAbsValue*inputParams.stddevthresh &&
             fabs(ImData->data[p]) < meanAbsValue*inputParams.stddevthresh &&
             NData->data[p] < meanNoiseEst*inputParams.stddevthresh){
            ReData->data[j] = ReData->data[p];
            ImData->data[j] = ImData->data[p];
            NData->data[j] = NData->data[p];
            timeStamp->data[j] = timeStamp->data[p];
            j++;
          }
        }
        if(j!=ReData->length){
          if( (ReData = XLALResizeREAL8Vector(ReData, j)) == NULL ||
              (ImData = XLALResizeREAL8Vector(ImData, j)) == NULL ||
              (NData = XLALResizeREAL8Vector(NData, j)) == NULL ||
              (timeStamp = XLALResizeINT4Vector(timeStamp, j)) == NULL )
            {  XLALPrintError("Error resizing thresholded data.\n");  }
        }
      }

      /* Output the t, B_k (and sigma_k) to the file */
      /* (no buffer needed this time as looping through pulsars rather than SFTs) */
      for(p=0;p<ReData->length;p++){

        fprintf(BkFileOut,"%d\t%.6e\t%.6e\t%.6e\n",timeStamp->data[p],ReData->data[p]
           ,ImData->data[p], NData->data[p]);
      }

      fclose(BkFileOut);

      /* destroy vectors to prvent memory leaks */
      XLALDestroyINT4Vector(timeStamp);
      XLALDestroyREAL8Vector(ReData);
      XLALDestroyREAL8Vector(ImData);
      XLALDestroyREAL8Vector(NData);

    }
    fprintf(stderr," done.\n");
    if(inputParams.Timing) gettimeofday(&timeOREnd, NULL);
    if(inputParams.Timing){
      REAL8 tOR  = (REAL8)timeOREnd.tv_usec*1.e-6 + (REAL8)timeOREnd.tv_sec - (REAL8)timeORStart.tv_usec*1.e-6 - (REAL8)timeORStart.tv_sec;
      fprintf(stderr,"Outlier Removal took %.5fs",tOR);
    }

  }

  if(inputParams.Timing){
    gettimeofday(&timeEndTot, NULL);
    REAL8 tTot  = (REAL8)timeEndTot.tv_usec*1e-6 + (REAL8)timeEndTot.tv_sec - (REAL8)timePreTot.tv_usec*1e-6 - (REAL8)timePreTot.tv_sec;
    fprintf(stderr,"Total time taken: %.5fs\n",tTot);
  }


  return 0;
}


void get_input_args(InputParams *inputParams, int argc, char *argv[]){
  struct option long_options[] =
  {
    { "help",                     no_argument,        0, 'h' },
    { "ifo",                      required_argument,  0, 'i' },
    { "sft-cache",                required_argument,  0, 'F' },
    { "sft-loc",                  required_argument,  0, 'L' },
    { "param-dir",                required_argument,  0, 'd' },
    { "param-file",               required_argument,  0, 'P' },
    { "psr-name",                 required_argument,  0, 'N' },
    { "output-dir",               required_argument,  0, 'o' },
    { "ephem-dir",                required_argument,  0, 'e' },
    { "seg-file",                 required_argument,  0, 'l' },
    { "start-freq",		  required_argument,  0, 'S' },
    { "end-freq",		  required_argument,  0, 'E' },
    { "freq-factor",              required_argument,  0, 'm' },
    { "bandwidth",                required_argument,  0, 'b' },
    { "min-seg-length",           required_argument,  0, 'M' },
    { "starttime",                required_argument,  0, 's' },
    { "finishtime",               required_argument,  0, 'f' },
    { "stddevthresh",             required_argument,  0, 'T' },
    { "output-timing",            no_argument,     NULL, 't' },
    { "geocentreFlag",            no_argument,     NULL, 'g' },
    { "baryFlag",                 no_argument,     NULL, 'B' },
    { 0, 0, 0, 0 }
  };

  char args[] = "hi:F:L:d:P:N:o:e:l:S:E:m:b:M:s:f:T:ntgB";

  /* set defaults */
  inputParams->freqfactor = 2.0; /* default is to look for gws at twice the
pulsar spin frequency */
  inputParams->bandwidth = 0.3; /* Default is a 0.3Hz bandwidth search */
  inputParams->geocentre = 0; /* Default is not to look at the geocentre */
  inputParams->minSegLength = 1800; /* Default minimum segment length is 1800s */
  inputParams->baryFlag = 0; /* Default is to perform barycentring routine */
  inputParams->endF=0.; /* in order to check if the end frequency is set */
  inputParams->startF=0.; /* in order to check if the start frequency is set*/
  inputParams->parfileflag=0.; /* in order to check if the Parfile is set */
  inputParams->pardirflag=0.; /* in order to check if the Parfile directory is set*/
  inputParams->nameset=0.; /* flag in order to check if pulsar name is set*/
  inputParams->startTime = 0.; /* Start time not set - use zero */
  inputParams->endTime = INFTY; /* end time not set - use infinity */
  inputParams->Timing=0.; /* default not to output timing info to stderr */
  inputParams->cacheDir=0.; /* default is that directory flag is zero */
  inputParams->cacheFile=0.; /* default that file flag is zero */
  inputParams->stddevthresh=0.; /* default not to do outlier removal */



  /* get input arguments */
  while(1){
    int option_index = 0;
    int c;

    c = getopt_long( argc, argv, args, long_options, &option_index );
    if ( c == -1 ) /* end of options */
      break;

    switch(c){
      case 'h': /* Help */
        fprintf(stderr, USAGE);
        exit(1);
      case 'i': /* interferometer */
        snprintf(inputParams->ifo, sizeof(inputParams->ifo), "%s", optarg);
        break;
      case 'g': /* geocentre flag */
        inputParams->geocentre = 1;
        break;
      case 'B': /* barycentre flag */
        inputParams->baryFlag = 1;
        break;
      case 'm': /* frequency Factor */
        inputParams->freqfactor = atof(optarg);
        break;
      case 'S': /* start frequency of the SFTs */
        inputParams->startF = atof(optarg);
        break;
      case 'E': /* end frequency of the SFTs */
        inputParams->endF = atof(optarg);
        break;
      case 'd': /* pulsar par file directory */
        snprintf(inputParams->pulsarDir, sizeof(inputParams->pulsarDir), "%s",
          optarg);
        inputParams->pardirflag=1;
        break;
      case 'e': /* ephemeris directory */
        snprintf(inputParams->ephemdir, sizeof(inputParams->ephemdir), "%s",
          optarg);
        break;
      case 'P': /* Pulsar File*/
        snprintf(inputParams->paramfile, sizeof(inputParams->paramfile), "%s",
          optarg);
        inputParams->parfileflag=1;
        break;
      case 'N': /* Pulsar name set */
        snprintf(inputParams->PSRname, sizeof(inputParams->PSRname), "%s",
          optarg);
        inputParams->nameset=1;
        break;
      case 'F': /* filepath to SFTs */
        snprintf(inputParams->filePattern, sizeof(inputParams->filePattern), "%s",
          optarg);
        inputParams->cacheFile=1;
        break;
      case 'L': /* filepath to SFTcache directory */
        snprintf(inputParams->filePattern, sizeof(inputParams->filePattern), "%s",
          optarg);
        inputParams->cacheDir=1;
        break;
      case 'l': /* segment file */
        snprintf(inputParams->segfile, sizeof(inputParams->segfile), "%s",
          optarg);
        break;
      case 'o': /* output-dir */
        snprintf(inputParams->outputdir, sizeof(inputParams->outputdir), "%s",
          optarg);
        break;
      case 'b': /* bandwidth */
        inputParams->bandwidth = atof(optarg);
        break;
      case 's': /* Start time of the analysis */
        inputParams->startTime = atof(optarg);
        break;
      case 'f': /* finish time of the analysis */
        inputParams->endTime = atof(optarg);
        break;
      case 'T': /* standard deviation threshold for outlier removal */
        inputParams->stddevthresh = atof(optarg);
        break;
      case 'M': /* Minimum Segment Length Allowed */
        inputParams->minSegLength = atof(optarg);
        break;
      case 't': /* write timing values to stderr */
        inputParams->Timing = 1;
        break;
      case '?':
        fprintf(stderr, "unknown error while parsing options\n" );
      default:
        fprintf(stderr, "unknown error while parsing options\n" );
    }
  }
}

/* read in science segment list file - returns the number of segments */
INT4 get_segment_list(INT4Vector *starts, INT4Vector *stops, REAL8 minDur, REAL8 startSegs , REAL8 endSegs, CHAR *seglistfile){
  FILE *fp=NULL;
  INT4 i=0;
  long offset;
  CHAR jnkstr[256]; /* junk string to contain comment lines */
  INT4 num, dur; /* variable to contain the segment number and duration */
  INT4 linecount=0; /* number of lines in the segment file */
  INT4 ch=0;

  if ( (endSegs-startSegs) < minDur){
    fprintf(stderr, "Error... can't get a segment of length %f between times %f and %f.\n", minDur, startSegs, endSegs);
    exit(1);
  }

  if((fp=fopen(seglistfile, "r"))==NULL){
    fprintf(stderr, "Error... can't open science segment list file.\n");
    exit(1);
  }

  /* count number of lines in the file */
  while ( (ch = fgetc(fp)) != EOF ){
    if ( ch == '\n' ) /* check for return at end of line */
      linecount++;
  }
  
  /* allocate memory for vectors */
  if( (starts = XLALResizeINT4Vector( starts, linecount )) == NULL ||
      (stops = XLALResizeINT4Vector( stops, linecount )) == NULL )
    {  XLALPrintError("Error resizing segment lists.\n");  }

  /* rewind file pointer */
  rewind(fp);

  /* segment list files have comment lines starting with a # so want to ignore
     those lines */
  while(!feof(fp)){
    offset = ftell(fp); /* get current position of file stream */

    if(fscanf(fp, "%s", jnkstr) == EOF) /* scan in value and check if == to # */
      break; /* break if there is an extra line at the end of file containing
                nothing */

    if(strstr(jnkstr, "#")){
       /* if == # then skip to the end of the line */
      if ( fscanf(fp, "%*[^\n]") == EOF ) break;
      continue;
    }
    else{
      fseek(fp, offset, SEEK_SET); /* if line doesn't start with a # then it is
                                      data */
      INT4 starttemp=0,stoptemp=0;
      INT4 starttemp2=0,stoptemp2=0;
      if( fscanf(fp, "%d%d%d%d", &num, &starttemp, 
                 &stoptemp, &dur) == EOF ) break;
      /*format is segwizard type: num starts stops dur */

      /* perform checks on whether the segment is within or overlapping
       the segment and then output the correct combination */
      if ( starttemp > endSegs || stoptemp < startSegs) continue; /* if outside of time range, ignore */
      else {
        if ( starttemp < startSegs ){ /* if segment starts before the specified start time */
          starttemp2 = startSegs; /* Start time is specified start */
        }
        else{
          starttemp2 = starttemp; /* Start time is segment start */
        }
        if ( stoptemp > endSegs){ /* if segment ends after the specified end time */
          stoptemp2 = endSegs; /* end time is specified end */
        }
        else{
          stoptemp2 = stoptemp; /* end time is segment end */
        }

        if ( (stoptemp2-starttemp2) < minDur) continue;

        starts->data[i]=starttemp2;
        stops->data[i]=stoptemp2;

        i++;
      }
    }
  }
  starts = XLALResizeINT4Vector( starts, i );
  stops = XLALResizeINT4Vector( stops, i );

  fclose(fp);

  return i;
}


/* Function to compute the Fresnel Integrals, [largely from Numerical Recipes in C (1992)]*/
INT4 XLALFresnel(REAL8 *C, REAL8 *S, REAL8 x){

  INT4 k, n, odd;
  REAL8 a, absx, fact, sign, sum, sumc, sums, term, test;
  COMPLEX16 b, cc, d, h, del, cs;

  REAL8 FC = 0., FS = 0.;

  absx=fabs(x);

  if(absx<sqrt(XLAL_FRESNEL_FPMIN)){ /* If x is less than the minimum floating point value */
    FS = 0.0;                        /* then approximate accordingly*/
    FC = absx;
  }
  else if (absx <= XLAL_FRESNEL_XMIN){ /* If x is less than specified value, then use the power series*/
    sum = sums = 0.0;
    sumc = absx;
    sign = 1.0;
    fact = LAL_PI_2*absx*absx;
    odd = 1;
    term = absx;
    n = 3;
    for (k = 1; k<XLAL_FRESNEL_MAXIT; k++){
      term*=fact/k;
      sum +=sign*term/n;
      test = fabs(sum)*XLAL_FRESNEL_EPS;
      if(odd){
        sign = -sign;
        sums = sum;
        sum = sumc;
      }
      else{
        sumc = sum;
        sum = sums;
      }
      if (term < test) break;
      odd = !odd;
      n+=2;
    }
    if (k > XLAL_FRESNEL_MAXIT) {
      XLALPrintError("XLAL_Fresnel failed in series");
      XLAL_ERROR(XLAL_EFAILED);
    }
    FS = sums;
    FC = sumc;
  }
  else{ /* If x is greater than specified value, then use the continued fraction*/
    b = 1.0 - I*LAL_PI*absx*absx;
    cc = 1.0/XLAL_FRESNEL_FPMIN;
    d = h = 1./b;
    n = -1;
    for (k = 2;  k<XLAL_FRESNEL_MAXIT; k++){
      n+=2;
      a = -n*(n+1);
      b += 4.0;
      d = 1./(a*d+b);
      cc = b+a/cc;
      del = cc*d;
      h = h*del;
      if(fabs(creal(del)-1.0)+fabs(cimag(del))<XLAL_FRESNEL_EPS) break;
    }
    if (k>XLAL_FRESNEL_MAXIT){
      XLALPrintError("XLAL_Fresnel failed in continued fraction");
      XLAL_ERROR(XLAL_EFAILED);
    }
    h = (absx-I*absx)*h;
    cs = (0.5+I*0.5)*(1.-h*(cos(LAL_PI_2*absx*absx)+I*sin(LAL_PI_2*absx*absx)));
    FC = creal(cs);
    FS = cimag(cs);
  }
  if(x<0){ /* Use antisymmetry if x is negative */
    FS = -FS;
    FC = -FC;
  }
  C = memcpy(C,&FC,sizeof(REAL8));
  S = memcpy(S,&FS,sizeof(REAL8));
  return XLAL_SUCCESS;
}

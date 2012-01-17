/*
 *  Copyright (C) 2007 Chad Hanna, Alexander Dietz, Duncan Brown, Gareth Jones, Jolien Creighton, Nickolas Fotopoulos, Patrick Brady, Stephen Fairhurst, Tania Regimbau
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



/*-----------------------------------------------------------------------
 *
 * File Name: inspinj.c
 *
 * Author: Brown, D. A., Creighton, J. D. E. and Dietz A. IPN contributions from Predoi, V.
 *
 * Revision: $Id$
 *
 *-----------------------------------------------------------------------
 */

#include <ctype.h>
#include <getopt.h>
#include <lalapps.h>
#include <lal/Date.h>
#include <lal/LIGOMetadataTables.h>
#include <lal/LIGOMetadataInspiralUtils.h>
#include <lal/LIGOLwXMLInspiralRead.h>
#include <lal/LIGOLwXML.h>
#include <lal/Random.h>
#include <lal/AVFactories.h>
#include <lal/InspiralInjectionParams.h>
#include <lal/LALDetectors.h>
#include <lal/LALSimulation.h>
#include <processtable.h>
#include <lal/RingUtils.h>
#include <LALAppsVCSInfo.h>

#include "inspiral.h"

RCSID( "$Id$" );

#define CVS_REVISION "$Revision$"
#define CVS_ID_STRING "$Id$"
#define CVS_SOURCE "$Source$"
#define CVS_DATE "$Date$"
#define CVS_NAME_STRING "$Name$"
#define PROGRAM_NAME "inspinj"

#define ADD_PROCESS_PARAM( pptype, format, ppvalue ) \
  this_proc_param = this_proc_param->next = (ProcessParamsTable *) \
calloc( 1, sizeof(ProcessParamsTable) ); \
snprintf( this_proc_param->program, LIGOMETA_PROGRAM_MAX, "%s", \
    PROGRAM_NAME ); \
snprintf( this_proc_param->param, LIGOMETA_PARAM_MAX, "--%s", \
    long_options[option_index].name ); \
snprintf( this_proc_param->type, LIGOMETA_TYPE_MAX, "%s", pptype ); \
snprintf( this_proc_param->value, LIGOMETA_VALUE_MAX, format, ppvalue );

#ifdef __GNUC__
#define UNUSED __attribute__ ((unused))
#else
#define UNUSED
#endif

/*
 *  *********************************
 *  Definition of the prototypes
 *  *********************************
 */
extern int vrbflg;
ProcessParamsTable *next_process_param( const char *name, const char *type,
    const char *fmt, ... );
void read_mass_data( char *filename );
void read_nr_data( char* filename );
void read_source_data( char* filename );
void sourceComplete(void);
void drawFromSource( REAL8 *rightAscension,
    REAL8 *declination,
    REAL8 *distance,
    CHAR  name[LIGOMETA_SOURCE_MAX] );
void read_IPN_grid_from_file( char *fname );
void drawFromIPNsim( REAL8 *rightAscension,
    REAL8 *declination  );
void drawLocationFromExttrig( SimInspiralTable* table );
void drawMassFromSource( SimInspiralTable* table );
void drawMassSpinFromNR( SimInspiralTable* table );
void drawMassSpinFromNRNinja2( SimInspiralTable* table );

void adjust_snr(SimInspiralTable *inj, REAL8 target_snr, const char *ifos);
void adjust_snr_real8(SimInspiralTable *inj, REAL8 target_snr, const char *ifos);
REAL8 network_snr(const char *ifos, SimInspiralTable *inj);
REAL8 snr_in_ifo(const char *ifo, SimInspiralTable *inj);
REAL8 network_snr_real8(const char *ifos, SimInspiralTable *inj);
REAL8 snr_in_ifo_real8(const char *ifo, SimInspiralTable *inj);

REAL8 probability_redshift(REAL8 rshift);
REAL8 luminosity_distance(REAL8 rshift);
REAL8 mean_time_step_sfr(REAL8 zmax, REAL8 rate_local);
REAL8 drawRedshift(REAL8 zmin, REAL8 zmax, REAL8 pzmax);
REAL8 redshift_mass(REAL8 mass, REAL8 z);

/*
 *  *************************************
 *  Defining of the used global variables
 *  *************************************
 */

lalinspiral_time_distribution tDistr;
DistanceDistribution          dDistr;
SkyLocationDistribution       lDistr;
MassDistribution              mDistr;
InclDistribution              iDistr;

SimInspiralTable *simTable;
SimRingdownTable *simRingTable;

char *massFileName = NULL;
char *nrFileName = NULL;
char *sourceFileName = NULL;
char *outputFileName = NULL;
char *exttrigFileName = NULL;
char *IPNSkyPositionsFile = NULL;

INT4 outCompress = 0;
INT4 ninjaMass   = 0;

INT4 logSNR      = 0;
REAL4 minSNR     = -1;
REAL4 maxSNR     = -1;
char *ifos       = NULL;

float mwLuminosity = -1;
REAL4 dmin= -1;
REAL4 dmax= -1;
REAL4 localRate = -1.0;
REAL4 minMass1=-1;
REAL4 maxMass1=-1;
REAL4 minMass2=-1;
REAL4 maxMass2=-1;
REAL4 minMtotal=-1;
REAL4 maxMtotal=-1;
REAL4 meanMass1=-1.0;
REAL4 meanMass2=-1.0;
REAL4 massStdev1=-1.0;
REAL4 massStdev2=-1.0;
REAL4 minMassRatio=-1.0;
REAL4 maxMassRatio=-1.0;
REAL4 inclStd=-1.0;
REAL4 fixed_inc=-1.0;
REAL4 max_inc=LAL_PI/2.0;
REAL4 psi=-1.0;
REAL4 longitude=181.0;
REAL4 latitude=91.0;
REAL4 epsAngle=1e-7;
int spinInjections=-1;
int spinAligned=-1;
REAL4 minSpin1=-1.0;
REAL4 maxSpin1=-1.0;
REAL4 minSpin2=-1.0;
REAL4 maxSpin2=-1.0;
REAL4 minKappa1=-1.0;
REAL4 maxKappa1=1.0;
REAL4 minabsKappa1=0.0;
REAL4 maxabsKappa1=1.0;
REAL4 fixedMass1=-1.0;
REAL4 fixedMass2=-1.0;
INT4  pntMass1=1;
INT4  pntMass2=1;
REAL4 deltaMass1=-1;
REAL4 deltaMass2=-1;
INT4 bandPassInj = 0;
INT4 writeSimRing = 0;
InspiralApplyTaper taperInj = INSPIRAL_TAPER_NONE;
AlignmentType alignInj = notAligned;
REAL8 redshift;

static LALStatus status;
static RandomParams* randParams=NULL;
INT4 numExtTriggers = 0;
ExtTriggerTable   *exttrigHead = NULL;

int num_source;
int numSkyPoints;
int galaxynum;
struct {
  char   name[LIGOMETA_SOURCE_MAX];
  REAL8 ra;
  REAL8 dec;
  REAL8 dist;
  REAL8 lum;
  REAL8 fudge;
} *source_data, *old_source_data,*temparray, *skyPoints;

char MW_name[LIGOMETA_SOURCE_MAX] = "MW";
REAL8* fracVec  =NULL;
REAL8* ratioVec = NULL;
REAL8 norm=0;

int num_mass;
struct {
  REAL8 mass1;
  REAL8 mass2;
} *mass_data;

struct FakeGalaxy{
char name[LIGOMETA_SOURCE_MAX];
REAL8 ra;
REAL8 dec;
REAL8 lum;
REAL8 dist;
REAL8 fudge;
struct FakeGalaxy *next; };
int srcComplete = 0;
int makeCatalog = 0;
REAL8 srcCompleteDist;

int num_nr = 0;
int i = 0;
SimInspiralTable **nrSimArray = NULL;

/*
 *  *********************************
 *  Implementation of the code pieces
 *  *********************************
 */

REAL8 probability_redshift(REAL8 rshift)
{
  REAL8 pz;

  pz = -0.000429072589677+(rshift*(-0.036349728568888+(rshift*(0.860892111762314
     +(rshift*(-0.740935488674010+rshift*(0.265848831356864+rshift*(-0.050041573542298
     +rshift*(0.005184554232421+rshift*(-0.000281450045300+rshift*0.000006400690921))))))))));

  return pz;
}

REAL8 luminosity_distance(REAL8 rshift)
{
  REAL8 dL;
        
        dL = -2.89287707063171+(rshift*(4324.33492012756+(rshift*(3249.74193862773
           +(rshift*(-1246.66339928289+rshift*(335.354613407693+rshift*(-56.1194965448065
       +rshift*(5.20261234121263+rshift*(-0.203151569744028))))))))));
  return dL;
}

REAL8 mean_time_step_sfr(REAL8 zmax, REAL8 rate_local)
{
  REAL8 logzmax,loglambda,step;

  logzmax=log10(zmax);
  loglambda = -0.039563*pow(logzmax,6.)-0.15282*pow(logzmax,5.)-0.017596*pow(logzmax,4.)
            + 0.67193*pow(logzmax,3.)+1.1347*pow(logzmax,2.)-2.3543*logzmax+ 2.0228;
  step=pow(10.,loglambda)/rate_local;

  return step;
}

REAL8 drawRedshift(REAL8 zmin, REAL8 zmax, REAL8 pzmax)
{
  REAL8 test,z,p;
    do
        {
      test = pzmax * XLALUniformDeviate(randParams);
      z = (zmax-zmin) * XLALUniformDeviate(randParams)+zmin;
          p= probability_redshift(z);
        }
        while (test>p);
        
  return z;
}

REAL8 redshift_mass(REAL8 mass, REAL8 z)
{
  REAL8 mz;
  mz= mass * (1.+z);
        
  return mz;
}

REAL8 snr_in_ifo(const char *ifo, SimInspiralTable *inj)
{
  REAL8 this_snr;
  REAL4TimeVectorSeries *tempStrain=NULL;

  AddNumRelStrainModes( &status, &tempStrain, inj);

  this_snr = calculate_ligo_snr_from_strain( tempStrain, inj, ifo);

  XLALDestroyREAL4VectorSequence (tempStrain->data);
  tempStrain->data = NULL;
  LALFree(tempStrain);
  tempStrain = NULL;

  return this_snr;
}


REAL8 snr_in_ifo_real8(const char *ifo, SimInspiralTable *inj)
{
  REAL8       this_snr;
  REAL8TimeSeries *strain = NULL;

  strain   = XLALNRInjectionStrain(ifo, inj);
  this_snr = calculate_ligo_snr_from_strain_real8(strain, ifo);

  XLALDestroyREAL8TimeSeries (strain);

  return this_snr;
}


REAL8 network_snr_real8(const char *ifo_list, SimInspiralTable *inj)
{
  char *tmp;
  char *ifo;

  REAL8 snr_total = 0.0;
  REAL8 this_snr;

  tmp = LALCalloc(1, strlen(ifos) + 1);
  strcpy(tmp, ifo_list);
  ifo = strtok (tmp,",");

  while (ifo != NULL)
  {
    this_snr   = snr_in_ifo_real8(ifo, inj);
    snr_total += this_snr * this_snr;
    ifo        = strtok (NULL, ",");
  }

  LALFree(tmp);

  return sqrt(snr_total);
}


REAL8 network_snr(const char *ifo_list, SimInspiralTable *inj)
{
  char *tmp;
  char *ifo;
  REAL8 snr_total = 0.0;
  REAL8 this_snr;

  tmp = LALCalloc(1, strlen(ifos) + 1);
  strcpy(tmp, ifo_list);

  ifo = strtok (tmp,",");
  while (ifo != NULL)
  {
    this_snr   = snr_in_ifo(ifo, inj);
    snr_total += this_snr * this_snr;
    ifo        = strtok (NULL, ",");
  }

  LALFree(tmp);

  return sqrt(snr_total);
}


void adjust_snr_real8(SimInspiralTable *inj, REAL8 target_snr, const char *ifo_list)
{
  /* Vars for calculating SNRs */
  REAL8 this_snr;
  REAL8 UNUSED low_snr, UNUSED high_snr;
  REAL8 low_dist,high_dist;

  this_snr = network_snr_real8(ifo_list, inj);

  if (this_snr > target_snr)
  {
    high_snr  = this_snr;
    high_dist = inj->distance;

    while (this_snr > target_snr)
    {
      inj-> distance = inj->distance * 3.0;
      this_snr       = network_snr_real8(ifo_list, inj);
    }
    low_snr  = this_snr;
    low_dist = inj->distance;
  } else {
    low_snr  = this_snr;
    low_dist = inj->distance;

    while (this_snr < target_snr)
    {
      inj->distance = (inj->distance) / 3.0;
      this_snr      = network_snr_real8(ifo_list, inj);
    }
    high_snr  = this_snr;
    high_dist = inj->distance;
  }

  while ( abs(target_snr - this_snr) > 1.0 )
  {
    inj->distance = (high_dist + low_dist) / 2.0;
    this_snr = network_snr_real8(ifo_list, inj);

    if (this_snr > target_snr)
    {
      high_snr  = this_snr;
      high_dist = inj->distance;
    } else {
      low_snr  = this_snr;
      low_dist = inj->distance;
    }
  }
}



void adjust_snr(SimInspiralTable *inj, REAL8 target_snr, const char *ifo_list)
{
  /* Vars for calculating SNRs */
  REAL8 this_snr;
  REAL8 UNUSED low_snr, UNUSED high_snr;
  REAL8 low_dist,high_dist;

  this_snr = network_snr(ifo_list, inj);

  if (this_snr > target_snr)
  {
    high_snr  = this_snr;
    high_dist = inj->distance;

    while (this_snr > target_snr)
    {
      inj-> distance = inj->distance * 3.0;
      this_snr       = network_snr(ifo_list, inj);
    }
    low_snr  = this_snr;
    low_dist = inj->distance;
  } else {
    low_snr  = this_snr;
    low_dist = inj->distance;

    while (this_snr < target_snr)
    {
      inj->distance = (inj->distance) / 3.0;
      this_snr      = network_snr(ifo_list, inj);
    }
    high_snr  = this_snr;
    high_dist = inj->distance;
  }

  while ( abs(target_snr - this_snr) > 1.0 )
  {
    inj->distance = (high_dist + low_dist) / 2.0;
    this_snr = network_snr(ifo_list, inj);

    if (this_snr > target_snr)
    {
      high_snr  = this_snr;
      high_dist = inj->distance;
    } else {
      low_snr  = this_snr;
      low_dist = inj->distance;
    }
  }
}



/*
 *
 * code to step forward in the process table
 *
 */
ProcessParamsTable *next_process_param( const char *name, const char *type,
    const char *fmt, ... )
{
  ProcessParamsTable *pp;
  va_list ap;
  pp = calloc( 1, sizeof( *pp ) );
  if ( ! pp )
  {
    perror( "next_process_param" );
    exit( 1 );
  }
  strncpy( pp->program, PROGRAM_NAME, LIGOMETA_PROGRAM_MAX );
  snprintf( pp->param, LIGOMETA_PARAM_MAX, "--%s", name );
  strncpy( pp->type, type, LIGOMETA_TYPE_MAX );
  va_start( ap, fmt );
  vsnprintf( pp->value, LIGOMETA_VALUE_MAX, fmt, ap );
  va_end( ap );
  return pp;
}

/*
 *
 * print-out of the usage
 *
 */
static void print_usage(char *program)
{
  fprintf(stderr,
      "%s [options]\n"\
      "The following options are recognized.  Options not surrounded in []\n"\
      "are required. Defaults are shown in brackets\n", program );
  fprintf(stderr,
      " [--help ]                 display this message\n"\
      " [--verbose]               print progress information\n"\
      " [--user-tag] usertag      set the usertag \n"\
      " [--output ] name          overwrite the standard file naming convention\n"\
      " [--write-compress]        write a compressed xml file\n\n");\
  fprintf(stderr,
      "Waveform details:\n"\
      " [--seed] randomSeed       seed for random number generator (default : 1)\n"\
      "  --f-lower freq           lower cut-off frequency.\n"\
      "  --waveform wfm           set waveform type to wfm\n"\
      "  --amp-order              set PN order in amplitude\n\n");
  fprintf(stderr,
      "Time distribution information:\n"\
      "  --gps-start-time start   GPS start time for injections\n"\
      "  --gps-end-time end       GPS end time for injections\n"\
      "  --ipn-gps-time IPNtime   GPS end time for IPN trigger\n"\
      "  --t-distr timeDist       set the time step distribution of injections\n"\
      "                           fixed: fixed time step\n"\
      "                           uniform: uniform distribution\n"\
      "                           exponential: exponential distribution for Poisson process\n"\
      "  [--time-step] step       space injections by average of step seconds\n"\
      "                           (suggestion : 2630 / pi seconds)\n"\
      "  [--time-interval] int    distribute injections in an interval, int s\n"\
      "                           (default : 0 seconds)\n\n");
  fprintf(stderr,
      "Source distribution information:\n"\
      "  --l-distr  locDist       set the source location distribution,\n"\
      "                           locDist must be one of:\n"\
      "                           source: use locations from source-file\n"\
      "                           exttrig: use external trigger file\n"\
      "                           random: uses random locations\n"\
      "                           fixed: set fixed location\n"\
      "                           ipn: random locations from IPN skypoints\n"\
      " [--longitude] longitude   read longitude if fixed value (degrees)\n"
      " [--latitude] latitude     read latitide if fixed value (degrees)\n"
      "  --d-distr distDist       set the distance distribution of injections\n"\
      "                           source: take distance from galaxy source file\n"\
      "                           uniform: uniform distribution in distance\n"\
      "                           log10: uniform distribution in log10(d) \n"\
      "                           volume: uniform distribution in volume\n"\
      "                           sfr: distribution derived from the SFR\n"\
      " [--local-rate] rho        set the local coalescence rate when --d-dist sfr\n"\
      "                           (suggestion: 1 per Mpc^3 per Myr)\n"\
      "  --i-distr INCDIST        set the inclination distribution, must be either\n"\
      "                           uniform: distribute uniformly over arccos(i)\n"\
      "                           gaussian: gaussian distributed in (i)\n"\
      "                           fixed: no distribution, fixed valued of (i)\n"\
      " --polarization psi        set the polarization angle for all \n"
      "                           injections (degrees)\n"\
      " [--incl-std]  inclStd     std dev for gaussian inclination dist\n"\
      " [--fixed-inc]  fixed_inc  value for the fixed inclination angle (in degrees) if '--i-distr fixed' is chosen.\n"\
      " [--max-inc]  max_inc      value for the maximum inclination angle (in degrees) if '--i-distr uniform' is chosen. \n"\
      " [--source-file] sources   read source parameters from sources\n"\
      "                           requires enable/disable milkyway\n"\
      " [--ipn-file] ipnskypoints read IPN sky points from file\n"\
      " [--sourcecomplete] distance \n"
      "                           complete galaxy catalog out to distance (kPc)\n"\
      " [--make-catalog]          create a text file of the completed galaxy catalog\n"\
      " [--enable-milkyway] lum   enables MW injections, set MW luminosity\n"\
      " [--disable-milkyway]      disables Milky Way injections\n"\
      " [--exttrig-file] exttrig  XML file containing external trigger\n"\
      " [--min-distance] DMIN     set the minimum distance to DMIN kpc\n"\
      " [--max-distance] DMAX     set the maximum distance to DMAX kpc\n"\
      "                           min/max distance required if d-distr not 'source'\n"\
      " [--use-chirp-distance]    Use this option to scale injections using \n"
      "                           chirp distance (relative to a 1.4,1.4)\n"\
      " [--min-snr] SMIN          Sets the minimum network snr\n"\
      " [--max-snr] SMAX          Sets the maximum network snr\n"\
      " [--log-snr]               If set distribute uniformly in log(snr) rather than snr\n"\
      " [--ifos] ifos             Comma-separated list of ifos to include in network SNR\n\n");
  fprintf(stderr,
      "Mass distribution information:\n"\
      " --m-distr massDist        set the mass distribution of injections\n"\
      "                           must be one of:\n"\
      "                           source: using file containing list of mass pairs\n"\
      "                           nrwaves: using xml file with list of NR waveforms\n"\
      "                           (requires setting max/min total masses)\n"\
      "                           totalMass: uniform distribution in total mass\n"\
      "                           componentMass: uniform in m1 and m2\n"\
      "                           gaussian: gaussian mass distribution\n"\
      "                           log: log distribution in component mass\n"\
      "                           totalMassRatio: uniform distribution in total mass and\n"\
      "                           mass ratio m1 / m2\n"\
      "                           logTotalMassUniformMassRatio: log distribution in total mass\n"\
      "                           and uniform in mass ratio\n"\
      "                           totalMassFraction: uniform distribution in total mass and\n"\
      "                           in `mass fraction' m1 / (m1+m2)\n"\
      "                           m1m2SquareGrid: component masses on a square grid\n"\
      "                           fixMasses: fix m1 and m2 to specific values\n"\
      " [--ninja2-mass]           use the NINJA 2 mass-selection algorithm\n"\
      " [--mass-file] mFile       read population mass parameters from mFile\n"\
      " [--nr-file] nrFile        read mass/spin parameters from xml nrFile\n"\
      " [--min-mass1] m1min       set the minimum component mass to m1min\n"\
      " [--max-mass1] m1max       set the maximum component mass to m1max\n"\
      " [--min-mass2] m2min       set the min component mass2 to m2min\n"\
      " [--max-mass2] m2max       set the max component mass2 to m2max\n"\
      " [--min-mtotal] minTotal   sets the minimum total mass to minTotal\n"\
      " [--max-mtotal] maxTotal   sets the maximum total mass to maxTotal\n"\
      " [--fixed-mass1] fixMass1  set mass1 to fixMass1\n"\
      " [--fixed-mass2] fixMass2  set mass2 to fixMass2\n"\
      " [--mean-mass1] m1mean     set the mean value for mass1\n"\
      " [--stdev-mass1] m1std     set the standard deviation for mass1\n"\
      " [--mean-mass2] m2mean     set the mean value for mass2\n"\
      " [--stdev-mass2] m2std     set the standard deviation for mass2\n"\
      " [--min-mratio] minr       set the minimum mass ratio\n"\
      " [--max-mratio] maxr       set the maximum mass ratio\n"\
      " [--mass1-points] m1pnt    set the number of grid points in the m1 direction if '--m-distr=m1m2SquareGrid'\n"\
      " [--mass2-points] m2pnt    set the number of grid points in the m2 direction if '--m-distr=m1m2SquareGrid'\n\n");
  fprintf(stderr,
      "Spin distribution information:\n"\
      "  --disable-spin           disables spinning injections\n"\
      "  --enable-spin            enables spinning injections\n"\
      "                           One of these is required.\n"\
      "  --aligned                enforces the spins to be along the direction\n"\
      "                           of orbital angular momentum.\n"\
      "  [--min-spin1] spin1min   Set the minimum spin1 to spin1min (0.0)\n"\
      "  [--max-spin1] spin1max   Set the maximum spin1 to spin1max (0.0)\n"\
      "  [--min-spin2] spin2min   Set the minimum spin2 to spin2min (0.0)\n"\
      "  [--max-spin2] spin2max   Set the maximum spin2 to spin2max (0.0)\n"\
      "  [--min-kappa1] kappa1min Set the minimum cos(S1.L_N) to kappa1min (-1.0)\n"\
      "  [--max-kappa1] kappa1max Set the maximum cos(S1.L_N) to kappa1max (1.0)\n"\
      "  [--min-abskappa1] abskappa1min \n"\
      "                           Set the minimum absolute value of cos(S1.L_N)\n"\
      "                           to abskappa1min (0.0)\n"\
      "  [--max-abskappa1] abskappa1max \n"\
      "                           Set the maximum absolute value of cos(S1.L_N) \n"\
      "                           to abskappa1max (1.0)\n\n");
  fprintf(stderr,
      "Tapering the injection waveform:\n"\
      "  [--taper-injection] OPT  Taper the inspiral template using option OPT\n"\
      "                            (start|end|startend) \n)"\
      "  [--band-pass-injection]  sets the tapering method of the injected waveform\n\n");
  fprintf(stderr,
      "Output:\n"\
      " [--write-sim-ring]        Writes a sim_ringdown table\n\n");
}


/*
 *
 * functions to read source masses
 *
 */

  void
read_mass_data( char* filename )
{
  char line[256];
  FILE   *fp;
  int n = 0;

  fp=fopen( filename, "r" );
  if ( ! fp )
  {
    perror( "read_mass_data" );
    fprintf( stderr,
        "Error while trying to open file %s\n",
        filename );
    exit( 1 );
  }

  /* count the number of lines in the file */
  num_mass=0;
  while ( fgets( line, sizeof( line ), fp ) )
    ++num_mass;

  /* alloc space for the data */
  mass_data = LALCalloc( num_mass, sizeof(*mass_data) );
  if ( !mass_data )
  {
    fprintf( stderr, "Allocation error for mass_data\n" );
    exit( 1 );
  }

  /* 'rewind' the file */
  rewind( fp );

  /* read the file finally */
  while ( fgets( line, sizeof( line ), fp ) )
  {
    sscanf( line, "%le %le", &mass_data[n].mass1, &mass_data[n].mass2 );
    n++;
  }

  /* close the file */
  fclose( fp );
}

  void
read_nr_data( char* filename )
{
  SimInspiralTable  *nrSimHead = NULL;
  SimInspiralTable  *thisEvent= NULL;
  INT4               j = 0;

  num_nr = SimInspiralTableFromLIGOLw( &nrSimHead, filename, 0, 0 );


  if ( num_nr < 0 )
  {
    fprintf( stderr, "error: unable to read sim_inspiral table from %s\n",
        filename );
    exit( 1 );
  }
  else if ( num_nr == 0 )
  {
    fprintf( stderr, "error: zero events in sim_inspiral table from %s\n",
        filename );
  }

  /* allocate an array of pointers */
  nrSimArray = (SimInspiralTable ** )
    LALCalloc( num_nr, sizeof(SimInspiralTable *) );

  if ( !nrSimArray )
  {
    fprintf( stderr, "Allocation error for nr simulations\n" );
    exit( 1 );
  }

  for( j = 0, thisEvent=nrSimHead; j < num_nr;
      ++j, thisEvent = thisEvent->next )
  {
    nrSimArray[j] = thisEvent;
    if (j > 0)
    {
      nrSimArray[j-1]->next = NULL;
    }
  }
}


/*
 *
 * functions to read source distribution
 *
 */

  void
read_source_data( char* filename )
{
  char line[256];
  FILE *fp;
  int j, k;

  fp = fopen (filename, "r" );
  if ( ! fp )
  {
    perror( "read_source_data" );
    fprintf( stderr, "Could not find file %s\n", filename );
    exit( 1 );
  }

  /* count the number of entries in this file */
  num_source = 0;
  while ( fgets( line, sizeof( line ), fp ) )
    if ( line[0] == '#' )
      continue;
    else
      ++num_source;

  /* rewind the file */
  rewind( fp );

  /* allocate space */
  source_data = LALCalloc( num_source, sizeof( *source_data ) );
  if ( ! source_data )
  {
    fprintf( stderr, "Allocation error for source_data\n" );
    exit( 1 );
  }

  j = 0;
  while ( fgets( line, sizeof( line ), fp ) )
    if ( line[0] == '#' )
      continue;
    else
    {
      char ra_sgn, dec_sgn;
      REAL8 ra_h, ra_m, dec_d, dec_m;
      int c;

      c = sscanf( line, "%s %c%le:%le %c%le:%le %le %le %le",
          source_data[j].name, &ra_sgn, &ra_h, &ra_m, &dec_sgn, &dec_d, &dec_m,
          &source_data[j].dist, &source_data[j].lum, &source_data[j].fudge );
      if ( c != 10 )
      {
        fprintf( stderr, "error parsing source datafile %s\n", sourceFileName );
        exit( 1 );
      }

      /* by convention, overall sign is carried only on hours/degrees entry */
      source_data[j].ra  = ( ra_h + ra_m / 60.0 ) * LAL_PI / 12.0;
      source_data[j].dec = ( dec_d + dec_m / 60.0 ) * LAL_PI / 180.0;

      if ( ra_sgn == '-' )
        source_data[j].ra *= -1;
      if ( dec_sgn == '-' )
        source_data[j].dec *= -1;
      ++j;
    }

  /* close file */
  fclose( fp );

  /* generate ratio and fraction vectors */

  ratioVec = calloc( num_source, sizeof( REAL8 ) );
  fracVec  = calloc( num_source, sizeof( REAL8  ) );
  if ( !ratioVec || !fracVec )
  {
    fprintf( stderr, "Allocation error for ratioVec/fracVec\n" );
    exit( 1 );
  }

  /* MW luminosity might be zero */
  norm = mwLuminosity;

  /* calculate the fractions of the different sources */
  for ( k = 0; k < num_source; ++k )
    norm += ratioVec[k] = source_data[k].lum * source_data[k].fudge;
  fracVec[0] = ratioVec[0] / norm;
  for ( k = 1; k < num_source; ++k )
    fracVec[k] = fracVec[k-1] + ratioVec[k] / norm;
}

/*
 Function to read IPN sky simulations from text file given file - read(file,ra,dec)
*/
void read_IPN_grid_from_file( char *fname )

{

  UINT4              j;                      /* counters */
  char               line[256];              /* string holders */
  FILE               *data;                  /* file object */

  /* read file */
  data = fopen(fname, "r");

  /* check file */
  if ( ! data )
  {
    fprintf( stderr, "Could not find file %s\n", fname );
    exit( 1 );
  }


  /* find number of lines */
  numSkyPoints = 0;
  while ( fgets( line, sizeof( line ), data ) )
    ++numSkyPoints;


  /* seek to start of file again */
  fseek(data, 0, SEEK_SET);  

  /* assign memory for sky points */
  skyPoints = LALCalloc(numSkyPoints, sizeof(*skyPoints));
  if ( ! skyPoints )
  {
    fprintf( stderr, "Allocation error for skyPoints\n" );
    exit( 1 );
  }

  j = 0;
  while ( fgets( line, sizeof( line ), data ) )
    {
      REAL8 ra, dec;
      int c;

      c = sscanf( line, "%le %le", &ra, &dec );
      if ( c != 2 )
      {
        fprintf( stderr, "error parsing IPN sky points datafile %s\n", IPNSkyPositionsFile );
        exit( 1 );
      }

      /* convert to radians */
      skyPoints[j].ra  = ra * ( LAL_PI / 180.0 );  /* from degrees (IPN file) to radians */
      skyPoints[j].dec = dec * ( LAL_PI / 180.0 );
      ++j;
    }


  /* close file */
  fclose( data );
}

/*
*
*
* Function to complete galaxy catalog
*
*/
void sourceComplete() {

/*  Catalog Completion Constants */
REAL8 Mbstar = -20.45;
/* Mbstar = magnitude at which the number of galaxies begins to fall off exponentially, corrected for reddening (to agree with the
lum density of 0.0198) */
REAL8 phistar = 0.0081/0.92; /* normalization constant */
REAL8 alpha = -0.9; /* determines slope at faint end of luminosity function */
REAL8 initDistance = 0.0; /*minimum Distance for galaxy catalog*/
REAL8 DeltaD = 100.0; /* Distance step for stepping through galaxy catalog (kpc) */
REAL8 maxDistance = srcCompleteDist; /*Distance to which you want to correct the catalog (kPc)*/
REAL8 M_min = -12.0; /* minimum blue light magnitude */
REAL8 M_max = -25; /* maximum blue light magnitude */
REAL8 edgestep = 0.1; /* magnitude bin size */

/*  Vectors  */
REAL8Vector *phibins = NULL; /* Magnitude bins for calculating Schechter function */
REAL8Vector *Distance = NULL; /* Distances from initDistance to maxDistance in steps of DeltaD */
REAL8Vector *phi = NULL; /* Schecter magnitude function */
REAL8Vector *phiN = NULL; /* Number of expected galaxies in each magnitude bin */
REAL8Vector *N = NULL; /* Actual number of galaxies in each magnitude bin */
REAL8Vector *pN = NULL; /* Running tally of the fake galaxies added to the catalog */
REAL8Vector *Corrections = NULL; /* Number of galaxies to be added in each magnitude bin */

/* Other Variables */
int edgenum = (int) ceil((M_min-M_max)/edgestep); /* Number of magnitude bins */
char galaxyname[LIGOMETA_SOURCE_MAX] = "Fake"; /* Beginning of name for all added (non-real) galaxies */
int distnum = (maxDistance-initDistance)/DeltaD; /* Number of elements in Distance vector */
int k_at_25Mpc = floor((25000-initDistance)/DeltaD); /*Initial index for Distance vector - no galaxies added before 25Mpc */
int j,k,q; /* Indices for loops */
REAL8 mag; /* Converted blue light luminosity of each galaxy */
int mag_index; /* Index of each galaxy when binning by magnitude */
FILE *fp; /* File for output of corrected galaxy catalog */
REAL8 pow1 = 0.0; /* Used to calculate Schechter function */
REAL8 pow2 = 0.0; /* Used to calculate Schechter function */

REAL8 UNUSED shellLum = 0.0;

/* Parameters for generating random sky positions */
SimInspiralTable *randPositionTable;
static RandomParams* randPositions=NULL;
int rand_skylocation_seed = 3456;

/* Set up linked list for added galaxies*/
struct FakeGalaxy *myFakeGalaxy;
struct FakeGalaxy *head; /*=myFakeGalaxy;*/
struct FakeGalaxy *saved_next;

/* Create the Vectors */
phibins = XLALCreateREAL8Vector(edgenum);
Distance = XLALCreateREAL8Vector(distnum+1);
phi = XLALCreateREAL8Vector(edgenum);
phiN = XLALCreateREAL8Vector(edgenum); N = XLALCreateREAL8Vector(edgenum);
pN = XLALCreateREAL8Vector(edgenum);
Corrections = XLALCreateREAL8Vector(edgenum);

/* Initialize sky location parameters and FakeGalaxy linked list */
randPositionTable = calloc(1, sizeof(SimInspiralTable));
LALCreateRandomParams( &status, &randPositions, rand_skylocation_seed);
galaxynum = 0;
myFakeGalaxy = (struct FakeGalaxy*) calloc(1, sizeof(struct FakeGalaxy));
head = myFakeGalaxy;

/* Initialize the vectors */
for (j=0; j<edgenum; j++)
  {
  phibins->data[j] = M_max+j*edgestep;
  phiN->data[j] = 0;
  N->data[j] = 0;
  pN->data[j] = 0;
  Corrections->data[j] = 0;

  /* Calculate the theoretical blue light magnitude in each magnitude bin */
  pow1 = -1*pow(10, (-0.4*(phibins->data[j]-Mbstar)));
  pow2 = pow(10, (-0.4*(phibins->data[j]-Mbstar)));
  phi->data[j] = 0.92*phistar*exp(pow1)*pow(pow2, alpha+1);
  }

/* Initialize the Distance array */
for (j=0; j<=distnum; j++)
  {
  Distance->data[j] = initDistance+j*DeltaD;
  }


/* Iterate through Distance vector and bin galaxies according to magnitude at each distance */
for (k = k_at_25Mpc; k<distnum; k++)
  {

  /* Reset N to zero before you count the galaxies with distances less than the current Distance */
  for (q = 0; q<edgenum;q++)
    {
    N->data[q]=0;
    }

  /* Count the number of galaxies in the spherical volume with radius Distance->data[k+1] and bin them in magnitude */
  for( q = 0; q<num_source; q++)
    {
    if ( (source_data[q].dist<=Distance->data[k+1]) )
      {
      /* Convert galaxy luminosity to blue light magnitude */
      mag = -2.5*(log10(source_data[q].lum)+7.808);
      /* Calculate which magnitude bin it falls in */
      mag_index = (int) floor((mag-M_max)/edgestep);
      /* Create a histogram array of the number of galaxies in each magnitude bin */
      if (mag_index >= 0 && mag_index<edgenum)
        {
        N->data[mag_index] += 1.0;
        }
      else printf("WARNING GALAXY DOESNT FIT IN BIN\n");
      }
    }

  /* Add galaxies to the catalog based on the difference between the expected number of galaxies and the number of galaxies in the catalog */
  for (j = 0; j<edgenum; j++)
    {
    /* Number of galaxies expected in the spherical volume with radius Distance->data[k+1] */
    phiN->data[j] =edgestep*phi->data[j]*(4.0/3.0)*LAL_PI*(pow(Distance->data[k+1]/1000.0,3));
    /*Difference between the counted number of galaxies and the expected number of galaxies */
    Corrections->data[j] = phiN->data[j] - N->data[j] - pN->data[j];
    /* If there are galaxies missing, add them */
    if (Corrections->data[j]>0.0)
      {
      for (q=0;q<floor(Corrections->data[j]);q++)
        {
        randPositionTable = XLALRandomInspiralSkyLocation( randPositionTable, randPositions);
        myFakeGalaxy->dist = Distance->data[k+1];
        myFakeGalaxy->ra = randPositionTable->longitude;
        myFakeGalaxy->dec = randPositionTable->latitude;
        myFakeGalaxy->fudge = 1;
        sprintf(myFakeGalaxy->name, "%s%d", galaxyname, galaxynum);
        myFakeGalaxy->lum = pow(10.0, (phibins->data[j]/(-2.5)-7.808));
        myFakeGalaxy->next = (struct FakeGalaxy*) calloc(1,sizeof(struct FakeGalaxy));
        myFakeGalaxy = myFakeGalaxy->next;
        galaxynum++;
        pN->data[j] += 1.0;
        }
      }
    }
  }

/*Combine source_data (original catalog) and FakeGalaxies into one array */
temparray = calloc((num_source+galaxynum), sizeof(*source_data));
  if ( !temparray )
  {     fprintf( stderr, "Allocation error for temparray\n" );
    exit( 1 );
  }

for (j=0;j<num_source;j++) {
        temparray[j].dist = source_data[j].dist;
        temparray[j].lum = source_data[j].lum;
        sprintf(temparray[j].name, "%s", source_data[j].name);
        temparray[j].ra = source_data[j].ra;
        temparray[j].dec = source_data[j].dec;
        temparray[j].fudge = source_data[j].fudge;
}
myFakeGalaxy = head;
for (j=num_source;j<(num_source+galaxynum);j++) {
        temparray[j].dist = myFakeGalaxy->dist;
        temparray[j].lum = myFakeGalaxy->lum;
        sprintf(temparray[j].name, "%s", myFakeGalaxy->name);
        temparray[j].ra = myFakeGalaxy->ra;
        temparray[j].dec = myFakeGalaxy->dec;
        temparray[j].fudge = myFakeGalaxy->fudge;
        myFakeGalaxy = myFakeGalaxy->next;
}
myFakeGalaxy->next = NULL;

/*Point old_source_data at source_data */
old_source_data = source_data;

/*Point source_data at the new array*/
source_data = temparray;
shellLum = 0;

if (makeCatalog == 1) {
/* Write the corrected catalog to a file */
fp = fopen("correctedcatalog.txt", "w+");
for (j=0; j<(num_source+galaxynum);j++) {
fprintf(fp, "%s %g %g %g %g %g \n", source_data[j].name, source_data[j].ra, source_data[j].dec, source_data[j].dist, source_data[j].lum, source_data[j].fudge );
}
fclose(fp);
}
/* Recalculate some variables from read_source_data that will have changed due to the addition of fake galaxies */
 ratioVec = (REAL8*) calloc( (num_source+galaxynum), sizeof( REAL8 ) );
 fracVec  = (REAL8*) calloc( (num_source+galaxynum), sizeof( REAL8  ) );
  if ( !ratioVec || !fracVec )
  {
    fprintf( stderr, "Allocation error for ratioVec/fracVec\n" );
    exit( 1 );
  }

  /* MW luminosity might be zero */
  norm = mwLuminosity;

  /* calculate the fractions of the different sources */
  for ( i = 0; i <(num_source+galaxynum); ++i )
    norm += ratioVec[i] = source_data[i].lum * source_data[i].fudge;
  fracVec[0] = ratioVec[0] / norm;
  for ( i = 1; i <(num_source+galaxynum); ++i )
    fracVec[i] = fracVec[i-1] + ratioVec[i] / norm;

/* Free some stuff */
myFakeGalaxy = head;
for (j=0; j<galaxynum; j++) {
        saved_next = myFakeGalaxy->next;
        free(myFakeGalaxy);
        myFakeGalaxy = saved_next;
}
LALFree(old_source_data);
LALFree( skyPoints );
LALDestroyRandomParams( &status, &randPositions);

XLALDestroyREAL8Vector(phibins);
XLALDestroyREAL8Vector(Corrections);
XLALDestroyREAL8Vector(Distance);
XLALDestroyREAL8Vector(phi);
XLALDestroyREAL8Vector(phiN);
XLALDestroyREAL8Vector(N);
XLALDestroyREAL8Vector(pN);
}



/*
 *
 * functions to draw masses from mass distribution
 *
 */

void drawMassFromSource( SimInspiralTable* table )
{
  REAL4 m1, m2, eta;
  int mass_index=0;

  /* choose masses from the mass-list */
  mass_index = (int)( num_mass * XLALUniformDeviate( randParams ) );
  m1 = redshift_mass(mass_data[mass_index].mass1,redshift);
  m2 = redshift_mass(mass_data[mass_index].mass2,redshift);

  eta=m1 * m2 / ( ( m1 + m2 ) * ( m1 + m2 ) );
  table->mass1 = m1;
  table->mass2 = m2;
  table->eta = eta;
  table->mchirp = pow( eta, 0.6) * (m1 + m2);
}


/*
 *
 * functions to draw masses from mass distribution
 *
 */
void drawMassSpinFromNR( SimInspiralTable* table )
{
  int mass_index=0;

  /* choose masses from the mass-list */
  mass_index = (int)( num_nr * XLALUniformDeviate( randParams ) );
  XLALRandomNRInjectTotalMass( table, randParams, minMtotal, maxMtotal,
      nrSimArray[mass_index]);
}


void drawMassSpinFromNRNinja2( SimInspiralTable* inj )
{
  /* For ninja2 we first select a mass, then find */
  /* a waveform that can be injected at that mass */

  int j,k;
  REAL8 startFreq, startFreqHz, massTotal;
  int indx,tmp,*indicies;

  /* Permute the indicies in a random order      */
  /* This lets us check each available waveform  */
  /* once and lets us know when no option works  */
  indicies = (int *) LALCalloc( num_nr, sizeof(int) );

  for ( j = 0; j < num_nr; j++ )
    indicies[j] = j;

  for ( j = 0; j < num_nr; j++ )
  {
    indx           = (int) ( (num_nr-j) * XLALUniformDeviate( randParams ) ) + j;
    tmp            = indicies[j];
    indicies[j]    = indicies[indx];
    indicies[indx] = tmp;
  }

  massTotal = (maxMtotal - minMtotal) * XLALUniformDeviate( randParams ) + minMtotal;

  for ( j = 0; j < num_nr; j++ )
  {
    k           = indicies[j];
    startFreq   = start_freq_from_frame_url(nrSimArray[k]->numrel_data);
    startFreqHz = startFreq / (LAL_TWOPI * massTotal * LAL_MTSUN_SI);

    /* if this startFreqHz makes us happy, inject it */
    if (startFreqHz <= inj->f_lower)
    {
      /* This is a copy of XLALRandomNRInjectTotalMass without  */
      /* the random mass selection.  TODO: refactor that method */
      inj->eta    = nrSimArray[k]->eta;
      inj->mchirp = massTotal * pow(inj->eta, 3.0/5.0);

      /* set mass1 and mass2 */
      inj->mass1 = (massTotal / 2.0) * (1 + pow( (1 - 4 * inj->eta), 0.5) );
      inj->mass2 = (massTotal / 2.0) * (1 - pow( (1 - 4 * inj->eta), 0.5) );

      /* copy over the spin parameters */
      inj->spin1x = nrSimArray[k]->spin1x;
      inj->spin1y = nrSimArray[k]->spin1y;
      inj->spin1z = nrSimArray[k]->spin1z;
      inj->spin2x = nrSimArray[k]->spin2x;
      inj->spin2y = nrSimArray[k]->spin2y;
      inj->spin2z = nrSimArray[k]->spin2z;

      /* copy over the numrel information */
      inj->numrel_mode_min = nrSimArray[k]->numrel_mode_min;
      inj->numrel_mode_max = nrSimArray[k]->numrel_mode_max;
      snprintf(inj->numrel_data, LIGOMETA_STRING_MAX, "%s",
               nrSimArray[k]->numrel_data);

      XLALFree(indicies);
      return;
    }
  }

  /* If we hit the end of the list, oops */
  XLALFree(indicies);
  /* should throw an error here... */
  fprintf(stderr,"No waveform could be injected at MTotal=%f Msun\n", massTotal/LAL_MTSUN_SI);
}

/*
 *
 * functions to draw sky location from source distribution
 *
 */
void drawFromSource( REAL8 *rightAscension,
    REAL8 *declination,
    REAL8 *distance,
    CHAR   name[LIGOMETA_SOURCE_MAX] )
{
  REAL4 u;
  int j;

  u=XLALUniformDeviate( randParams );

  /* draw from the source table */
  for ( j = 0; j < num_source; ++j )
  {
    if ( u < fracVec[j] )
    {
      /* put the parameters */
      *rightAscension = source_data[j].ra;
      *declination    = source_data[j].dec;
      *distance = source_data[j].dist/1000.0;
      memcpy( name, source_data[j].name,
          sizeof(CHAR) * LIGOMETA_SOURCE_MAX );
      return;
    }
  }

  /* now then, draw from MilkyWay
   * WARNING: This sets location AND distance */
  XLALRandomInspiralMilkywayLocation( rightAscension, declination, distance,
      randParams );
  memcpy( name, MW_name, sizeof(CHAR) * 30 );

}

/*
 *
 * functions to draw IPN sky location from IPN simulation points
 *
 */
void drawFromIPNsim( REAL8 *rightAscension,
    REAL8 *declination )
{
  REAL4 u;
  INT4 j;
  
  u=XLALUniformDeviate( randParams );
  j=( int ) (u*numSkyPoints);  
 

  /* draw from the IPN source table */
    if ( j < numSkyPoints )
    {
      /* put the parameters */
      *rightAscension = skyPoints[j].ra;
      *declination    = skyPoints[j].dec;
      return;
    }
}


/*
 *
 * functions to draw sky location from exttrig source file
 *
 */
void drawLocationFromExttrig( SimInspiralTable* table )
{
  LIGOTimeGPS timeGRB;  /* real time of the GRB */
  REAL4 ra_rad, de_rad;
  REAL8 gmst1, gmst2;

  /* convert the position (stored as degree) to radians first */
  ra_rad = exttrigHead->event_ra  * LAL_PI_180;
  de_rad = exttrigHead->event_dec * LAL_PI_180;

  /* populate the time structures */
  timeGRB.gpsSeconds     = exttrigHead->start_time;
  timeGRB.gpsNanoSeconds = exttrigHead->start_time_ns;

  gmst1 = XLALGreenwichMeanSiderealTime(&timeGRB);
  gmst2 = XLALGreenwichMeanSiderealTime(&table->geocent_end_time);

  /* populate the table */
  table->longitude = ra_rad- gmst1 + gmst2;
  table->latitude  = de_rad;
}


/*
 *
 * generate all parameters (sky position and angles) for a random inspiral
 *
 */
int main( int argc, char *argv[] )
{
  LIGOTimeGPS gpsStartTime = {-1,0};
  LIGOTimeGPS gpsEndTime = {-1,0};
  LIGOTimeGPS IPNgpsTime = {-1,0};
  LIGOTimeGPS currentGpsTime;
  long gpsDuration;

  REAL8 meanTimeStep = -1;
  REAL8 timeInterval = 0;
  REAL4 fLower = -1;
  UINT4 useChirpDist = 0;
  REAL4 minMass10, maxMass10, minMass20, maxMass20, minMtotal0, maxMtotal0, meanMass10, meanMass20, massStdev10, massStdev20; /* masses at z=0 */
  REAL8 pzmax=0; /* maximal value of the probability distribution of the redshift */
  INT4 ncount;
  size_t ninj;
  int rand_seed = 1;

  /* waveform */
  CHAR waveform[LIGOMETA_WAVEFORM_MAX];
  CHAR dummy[256];
  INT4 amp_order = -1;
  /* xml output data */
  CHAR                  fname[256];
  CHAR                 *userTag = NULL;
  MetadataTable         proctable;
  MetadataTable         procparams;
  MetadataTable         injections;
  MetadataTable         ringparams;
  ProcessParamsTable   *this_proc_param;
  LIGOLwXMLStream       xmlfp;

  REAL8 drawnDistance = 0.0;
  REAL8 drawnRightAscension = 0.0;
  REAL8 drawnDeclination = 0.0;
  CHAR  drawnSourceName[LIGOMETA_SOURCE_MAX];
  REAL8 IPNgmst1 = 0.0;
  REAL8 IPNgmst2 = 0.0;

  REAL8 targetSNR;


  status=blank_status;

  /* getopt arguments */
  struct option long_options[] =
  {
    {"help",                          no_argument, 0,                'h'},
    {"verbose",                 no_argument,       &vrbflg,           1 },
    {"source-file",             required_argument, 0,                'f'},
    {"mass-file",               required_argument, 0,                'm'},
    {"nr-file",                 required_argument, 0,                'c'},
    {"exttrig-file",            required_argument, 0,                'E'},
    {"f-lower",                 required_argument, 0,                'F'},
    {"gps-start-time",          required_argument, 0,                'a'},
    {"gps-end-time",            required_argument, 0,                'b'},
    {"ipn-gps-time",            required_argument, 0,                '"'},
    {"t-distr",                 required_argument, 0,                '('},
    {"time-step",               required_argument, 0,                't'},
    {"time-interval",           required_argument, 0,                'i'},
    {"seed",                    required_argument, 0,                's'},
    {"waveform",                required_argument, 0,                'w'},
    {"amp-order",               required_argument, 0,                'q'},
    {"user-tag",                required_argument, 0,                'Z'},
    {"userTag",                 required_argument, 0,                'Z'},
    {"m-distr",                 required_argument, 0,                'd'},
    {"min-mass1",               required_argument, 0,                'j'},
    {"max-mass1",               required_argument, 0,                'k'},
    {"min-mass2",               required_argument, 0,                'J'},
    {"max-mass2",               required_argument, 0,                'K'},
    {"min-mtotal",              required_argument, 0,                'A'},
    {"max-mtotal",              required_argument, 0,                'L'},
    {"fixed-mass1",             required_argument, 0,                ']'},
    {"fixed-mass2",             required_argument, 0,                '['},
    {"mean-mass1",              required_argument, 0,                'n'},
    {"mean-mass2",              required_argument, 0,                'N'},
    {"ninja2-mass",             no_argument,       &ninjaMass,         1},
    {"mass1-points",            required_argument, 0,                ':'},
    {"mass2-points",            required_argument, 0,                ';'},    
    {"stdev-mass1",             required_argument, 0,                'o'},
    {"stdev-mass2",             required_argument, 0,                'O'},
    {"min-mratio",              required_argument, 0,                'x'},
    {"max-mratio",              required_argument, 0,                'y'},
    {"min-distance",            required_argument, 0,                'p'},
    {"max-distance",            required_argument, 0,                'r'},
    {"use-chirp-distance",      no_argument,       0,                ','},
    {"min-snr",                 required_argument, 0,                '1'},
    {"max-snr",                 required_argument, 0,                '2'},
    {"log-snr",                 no_argument,       &logSNR,            1},
    {"ifos",                    required_argument, 0,                '3'},
    {"d-distr",                 required_argument, 0,                'e'},
    {"local-rate",              required_argument, 0,                ')'},
    {"l-distr",                 required_argument, 0,                'l'},
    {"longitude",               required_argument, 0,                'v'},
    {"latitude",                required_argument, 0,                'z'},
    {"i-distr",                 required_argument, 0,                'I'},
    {"incl-std",                required_argument, 0,                'B'},
    {"fixed-inc",               required_argument, 0,                'C'},
    {"max-inc",                 required_argument, 0,               1001},
    {"polarization",            required_argument, 0,                'S'},
    {"sourcecomplete",          required_argument, 0,                'H'},
    {"make-catalog",            no_argument,       0,                '.'},
    {"enable-milkyway",         required_argument, 0,                'M'},
    {"disable-milkyway",        no_argument,       0,                'D'},
    {"min-spin1",               required_argument, 0,                'g'},
    {"min-kappa1",              required_argument, 0,                'Q'},
    {"max-kappa1",              required_argument, 0,                'R'},
    {"min-abskappa1",           required_argument, 0,                'X'},
    {"max-abskappa1",           required_argument, 0,                'Y'},
    {"max-spin1",               required_argument, 0,                'G'},
    {"min-spin2",               required_argument, 0,                'u'},
    {"max-spin2",               required_argument, 0,                'U'},
    {"output",                  required_argument, 0,                'P'},
    {"version",                 no_argument,       0,                'V'},
    {"enable-spin",             no_argument,       0,                'T'},
    {"disable-spin",            no_argument,       0,                'W'},
    {"aligned",                 no_argument,       0,                '@'},
    {"write-compress",          no_argument,       &outCompress,       1},
    {"taper-injection",         required_argument, 0,                '*'},
    {"band-pass-injection",     no_argument,       0,                '}'},
    {"write-sim-ring",          no_argument,       0,                '{'},
    {"ipn-file",                required_argument, 0,                '^'},
    {0, 0, 0, 0}
  };
  int c;

  /* set up initial debugging values */
  lal_errhandler = LAL_ERR_EXIT;
  set_debug_level( "1" );

  /* create the process and process params tables */
  proctable.processTable = (ProcessTable *)
    calloc( 1, sizeof(ProcessTable) );
  XLALGPSTimeNow(&(proctable.processTable->start_time));
  XLALPopulateProcessTable(proctable.processTable, PROGRAM_NAME, LALAPPS_VCS_IDENT_ID,
      LALAPPS_VCS_IDENT_STATUS, LALAPPS_VCS_IDENT_DATE, 0);
  snprintf( proctable.processTable->comment, LIGOMETA_COMMENT_MAX, " " );
  this_proc_param = procparams.processParamsTable = (ProcessParamsTable *)
    calloc( 1, sizeof(ProcessParamsTable) );

  /* clear the waveform field */
  memset( waveform, 0, LIGOMETA_WAVEFORM_MAX * sizeof(CHAR) );

  /* parse the arguments */
  while ( 1 )
  {
    /* getopt_long stores long option here */
    int option_index = 0;
    long int gpsinput;
    size_t optarg_len;

    c = getopt_long_only( argc, argv,
        "hf:m:a:b:t:s:w:i:M:*", long_options, &option_index );

    /* detect the end of the options */
    if ( c == - 1 )
    {
      break;
    }

    switch ( c )
    {
      case 0:
        /* if this option set a flag, do nothing else now */
        if ( long_options[option_index].flag != 0 )
        {
          break;
        }
        else
        {
          fprintf( stderr, "error parsing option %s with argument %s\n",
              long_options[option_index].name, optarg );
          exit( 1 );
        }
        break;

      case 'f':
        optarg_len = strlen( optarg ) + 1;
        sourceFileName = calloc( 1, optarg_len * sizeof(char) );
        memcpy( sourceFileName, optarg, optarg_len * sizeof(char) );
        this_proc_param = this_proc_param->next =
          next_process_param( long_options[option_index].name, "string",
              "%s", optarg );
        break;

      case 'm':
        optarg_len = strlen( optarg ) + 1;
        massFileName = calloc( 1, optarg_len * sizeof(char) );
        memcpy( massFileName, optarg, optarg_len * sizeof(char) );
        this_proc_param = this_proc_param->next =
          next_process_param( long_options[option_index].name, "string",
              "%s", optarg );
        break;

      case 'c':
        optarg_len = strlen( optarg ) + 1;
        nrFileName = calloc( 1, optarg_len * sizeof(char) );
        memcpy( nrFileName, optarg, optarg_len * sizeof(char) );
        this_proc_param = this_proc_param->next =
          next_process_param( long_options[option_index].name, "string",
              "%s", optarg );
        break;

      case 'E':
        optarg_len = strlen( optarg ) + 1;
        exttrigFileName = calloc( 1, optarg_len * sizeof(char) );
        memcpy( exttrigFileName, optarg, optarg_len * sizeof(char) );
        this_proc_param = this_proc_param->next =
          next_process_param( long_options[option_index].name, "string",
              "%s", optarg );
        break;

      case 'F':
        fLower = atof( optarg );
        this_proc_param = this_proc_param->next =
          next_process_param( long_options[option_index].name, "float",
              "%f", fLower );
        break;

      case 'a':
        gpsinput = atol( optarg );
        if ( gpsinput < 441417609 )
        {
          fprintf( stderr, "invalid argument to --%s:\n"
              "GPS start time is prior to "
              "Jan 01, 1994  00:00:00 UTC:\n"
              "(%ld specified)\n",
              long_options[option_index].name, gpsinput );
          exit( 1 );
        }
        gpsStartTime.gpsSeconds = gpsinput;
        gpsStartTime.gpsNanoSeconds = 0;
        this_proc_param = this_proc_param->next =
          next_process_param( long_options[option_index].name, "int",
              "%ld", gpsinput );
        break;

      case 'b':
        gpsinput = atol( optarg );
        if ( gpsinput < 441417609 )
        {
          fprintf( stderr, "invalid argument to --%s:\n"
              "GPS start time is prior to "
              "Jan 01, 1994  00:00:00 UTC:\n"
              "(%ld specified)\n",
              long_options[option_index].name, gpsinput );
          exit( 1 );
        }
        gpsEndTime.gpsSeconds = gpsinput;
        gpsEndTime.gpsNanoSeconds = 0;
        this_proc_param = this_proc_param->next =
          next_process_param( long_options[option_index].name, "int",
              "%ld", gpsinput );
        break;

      case '"':
        gpsinput = atol( optarg );
        if ( gpsinput < 441417609 )
        {
          fprintf( stderr, "invalid argument to --%s:\n"
              "GPS start time is prior to "
              "Jan 01, 1994  00:00:00 UTC:\n"
              "(%ld specified)\n",
              long_options[option_index].name, gpsinput );
          exit( 1 );
        }
        IPNgpsTime.gpsSeconds = gpsinput;
        this_proc_param = this_proc_param->next =
          next_process_param( long_options[option_index].name, "int",
              "%ld", gpsinput );
        break;

      case 's':
        rand_seed = atoi( optarg );
        this_proc_param = this_proc_param->next =
          next_process_param( long_options[option_index].name, "int",
              "%d", rand_seed );
        break;
                        
      case '(':
                optarg_len = strlen( optarg ) + 1;
                memcpy( dummy, optarg, optarg_len );

                if (!strcmp(dummy, "fixed"))
                {
                  tDistr=LALINSPIRAL_FIXED_TIME_DIST;
                }
                else if (!strcmp(dummy, "uniform"))
                {
                  tDistr=LALINSPIRAL_UNIFORM_TIME_DIST;
                }
                else if(!strcmp(dummy, "exponential"))
                {
                  tDistr=LALINSPIRAL_EXPONENTIAL_TIME_DIST;
                }
                else
                {
                  tDistr=LALINSPIRAL_UNKNOWN_TIME_DIST;
                  fprintf( stderr, "invalid argument to --%s:\n"
                          "unknown time distribution: %s must be one of\n"
                          "fixed, uniform or exponential\n",
                          long_options[option_index].name, optarg );
                  exit( 1 );
                }
                break;

      case ')':
            localRate = atof( optarg );
            this_proc_param = this_proc_param->next =
            next_process_param( long_options[option_index].name,
                        "float", "%le", localRate );
            break;

      case 't':
        {
          meanTimeStep = atof( optarg );
          this_proc_param = this_proc_param->next =
            next_process_param( long_options[option_index].name, "float",
                "%le", meanTimeStep );
        }
        break;

      case 'i':
        timeInterval = atof( optarg );
        this_proc_param = this_proc_param->next =
          next_process_param( long_options[option_index].name,
              "float", "%le", timeInterval );
        break;

      case 'w':
        snprintf( waveform, LIGOMETA_WAVEFORM_MAX, "%s", optarg );
        this_proc_param = this_proc_param->next =
          next_process_param( long_options[option_index].name, "string",
              "%s", optarg );
        break;

      case 'q':
        amp_order = atof( optarg );
        this_proc_param = this_proc_param->next =
          next_process_param( long_options[option_index].name,
            "int", "%ld", amp_order );
      break;

      case 'M':
        /* set the luminosity of the Milky Way */
        mwLuminosity = atof( optarg );
        if ( mwLuminosity < 0 )
        {
          fprintf( stderr, "invalid argument to --%s:\n"
              "Milky Way luminosity must be positive"
              "(%f specified)\n",
              long_options[option_index].name, mwLuminosity );
          exit( 1 );
        }

        this_proc_param = this_proc_param->next =
          next_process_param( long_options[option_index].name, "float",
              "%le", mwLuminosity );
        break;

      case 'D':
        /* set the luminosity of the Milky Way */
        this_proc_param = this_proc_param->next =
          next_process_param( long_options[option_index].name, "string",
              "" );
        mwLuminosity = 0;
        break;

      case 'Z':
        /* create storage for the usertag */
        optarg_len = strlen( optarg ) + 1;
        userTag = (CHAR *) calloc( optarg_len, sizeof(CHAR) );
        memcpy( userTag, optarg, optarg_len );

        this_proc_param = this_proc_param->next = (ProcessParamsTable *)
          calloc( 1, sizeof(ProcessParamsTable) );
        snprintf( this_proc_param->program, LIGOMETA_PROGRAM_MAX, "%s",
            PROGRAM_NAME );
        snprintf( this_proc_param->param, LIGOMETA_PARAM_MAX, "--userTag" );
        snprintf( this_proc_param->type, LIGOMETA_TYPE_MAX, "string" );
        snprintf( this_proc_param->value, LIGOMETA_VALUE_MAX, "%s",
            optarg );
        break;

      case 'd':
        optarg_len = strlen( optarg ) + 1;
        memcpy( dummy, optarg, optarg_len );
        this_proc_param = this_proc_param->next = (ProcessParamsTable *)
          calloc( 1, sizeof(ProcessParamsTable) );
        snprintf( this_proc_param->program, LIGOMETA_PROGRAM_MAX, "%s",
            PROGRAM_NAME );
        snprintf( this_proc_param->param, LIGOMETA_PARAM_MAX, "--m-distr" );
        snprintf( this_proc_param->type, LIGOMETA_TYPE_MAX, "string" );
        snprintf( this_proc_param->value, LIGOMETA_VALUE_MAX, "%s",
            optarg );

        if (!strcmp(dummy, "source"))
        {
          mDistr=massFromSourceFile;
        }
        else if (!strcmp(dummy, "nrwaves"))
        {
          mDistr=massFromNRFile;
        }
        else if (!strcmp(dummy, "totalMass"))
        {
          mDistr=uniformTotalMass;
        }
        else if (!strcmp(dummy, "componentMass"))
        {
          mDistr=uniformComponentMass;
        }
        else if (!strcmp(dummy, "gaussian"))
        {
          mDistr=gaussianMassDist;
        }
        else if (!strcmp(dummy, "log"))
        {
          mDistr=logComponentMass;
        }
        else if (!strcmp(dummy, "totalMassRatio"))
        {
          mDistr=uniformTotalMassRatio;
        }
        else if (!strcmp(dummy, "logTotalMassUniformMassRatio"))
        {
          mDistr=logMassUniformTotalMassRatio;
        }
        else if (!strcmp(dummy, "m1m2SquareGrid"))
        {
          mDistr=m1m2SquareGrid;
        }
        else if (!strcmp(dummy, "fixMasses"))
        {
          mDistr=fixMasses;
        }
        else if (!strcmp(dummy, "totalMassFraction"))
        {
          mDistr=uniformTotalMassFraction;
        }
        else
        {
          fprintf( stderr, "invalid argument to --%s:\n"
              "unknown mass distribution: %s must be one of\n"
              "(source, nrwaves, totalMass, componentMass, gaussian, log,\n"
              "totalMassRatio, totalMassFraction, logTotalMassUniformMassRatio,\n"
              "m1m2SquareGrid)\n",
              long_options[option_index].name, optarg );
          exit( 1 );
        }
        break;

      case 'j':
        minMass1 = atof( optarg );
        this_proc_param = this_proc_param->next =
          next_process_param( long_options[option_index].name,
              "float", "%le", minMass1 );
        break;

      case 'k':
        maxMass1 = atof( optarg );
        this_proc_param = this_proc_param->next =
          next_process_param( long_options[option_index].name,
              "float", "%le", maxMass1 );
        break;

      case 'J':
        minMass2 = atof( optarg );
        this_proc_param = this_proc_param->next =
          next_process_param( long_options[option_index].name,
              "float", "%le", minMass2 );
        break;

      case 'K':
        maxMass2 = atof( optarg );
        this_proc_param = this_proc_param->next =
          next_process_param( long_options[option_index].name,
              "float", "%le", maxMass2 );
        break;

      case 'A':
        minMtotal = atof( optarg );
        this_proc_param = this_proc_param->next =
          next_process_param( long_options[option_index].name,
              "float", "%le", minMtotal );
        break;

      case 'L':
        maxMtotal = atof( optarg );
        this_proc_param = this_proc_param->next =
          next_process_param( long_options[option_index].name,
              "float", "%le", maxMtotal );
        break;

      case 'n':
        meanMass1 = atof( optarg );
        this_proc_param = this_proc_param->next =
          next_process_param( long_options[option_index].name,
              "float", "%le", meanMass1 );
        break;

      case 'N':
        meanMass2 = atof( optarg );
        this_proc_param = this_proc_param->next =
          next_process_param( long_options[option_index].name,
              "float", "%le", meanMass2 );
        break;

      case 'o':
        massStdev1 = atof( optarg );
        this_proc_param = this_proc_param->next =
          next_process_param( long_options[option_index].name,
              "float", "%le", massStdev1 );
        break;

      case 'O':
        massStdev2 = atof( optarg );
        this_proc_param = this_proc_param->next =
          next_process_param( long_options[option_index].name,
              "float", "%le", massStdev2 );
        break;

      case 'x':
        minMassRatio = atof( optarg );
        this_proc_param = this_proc_param->next =
          next_process_param( long_options[option_index].name,
              "float", "%le", minMassRatio );
        break;

      case 'y':
        maxMassRatio = atof( optarg );
        this_proc_param = this_proc_param->next =
          next_process_param( long_options[option_index].name,
              "float", "%le", maxMassRatio );
        break;

      case ':':
        pntMass1 = atof( optarg );
        this_proc_param = this_proc_param->next =
          next_process_param( long_options[option_index].name,
              "int", "%d", pntMass1 );
        break;

      case ';':
        pntMass2 = atof( optarg );
        this_proc_param = this_proc_param->next =
          next_process_param( long_options[option_index].name,
              "int", "%d", pntMass2 );
        break;
      
      case ']':
        fixedMass1 = atof( optarg );
        this_proc_param = this_proc_param->next =
          next_process_param( long_options[option_index].name,
              "float", "%d", fixedMass1 );
        break;
      
      case '[':
        fixedMass2 = atof( optarg );
        this_proc_param = this_proc_param->next =
          next_process_param( long_options[option_index].name,
              "float", "%d", fixedMass2 );
        break;

      case 'p':
        /* minimum distance from earth */
        dmin = (REAL4) atof( optarg );
        if ( dmin <= 0 )
        {
          fprintf( stderr, "invalid argument to --%s:\n"
              "minimum distance must be > 0: "
              "(%f kpc specified)\n",
              long_options[option_index].name, dmin );
          exit( 1 );
        }
        this_proc_param = this_proc_param->next =
          next_process_param( long_options[option_index].name,
              "float", "%e", dmin );
        break;

      case 'r':
        /* max distance from earth */
        dmax = (REAL4) atof( optarg );
        if ( dmax <= 0 )
        {
          fprintf( stderr, "invalid argument to --%s:\n"
              "maximum distance must be greater than 0: "
              "(%f kpc specified)\n",
              long_options[option_index].name, dmax );
          exit( 1 );
        }
        this_proc_param = this_proc_param->next =
          next_process_param( long_options[option_index].name,
              "float", "%e", dmax );
        break;

      case ',':
        /* Distribute injections in chirp distance not distance*/
        this_proc_param = this_proc_param->next =
          next_process_param( long_options[option_index].name, "string",
              "" );
        useChirpDist = 1;
        break;

      case 'e':
        optarg_len = strlen( optarg ) + 1;
        memcpy( dummy, optarg, optarg_len );
        this_proc_param = this_proc_param->next = (ProcessParamsTable *)
          calloc( 1, sizeof(ProcessParamsTable) );
        snprintf( this_proc_param->program, LIGOMETA_PROGRAM_MAX, "%s",
            PROGRAM_NAME );
        snprintf( this_proc_param->param,LIGOMETA_PARAM_MAX,"--d-distr" );
        snprintf( this_proc_param->type, LIGOMETA_TYPE_MAX, "string" );
        snprintf( this_proc_param->value, LIGOMETA_VALUE_MAX, "%s",
            optarg );

        if (!strcmp(dummy, "source"))
        {
          dDistr=distFromSourceFile;
        }
        else if (!strcmp(dummy, "uniform"))
        {
          dDistr=uniformDistance;
        }
        else if (!strcmp(dummy, "log10"))
        {
          dDistr=uniformLogDistance;
        }
        else if (!strcmp(dummy, "volume"))
        {
          dDistr=uniformVolume;
        }
        else if (!strcmp(dummy, "sfr"))
        {
          dDistr=sfr;
        }
        else
        {
          fprintf( stderr, "invalid argument to --%s:\n"
              "unknown source distribution: "
              "%s, must be one of (uniform, log10, volume, source)\n",
              long_options[option_index].name, optarg );
          exit( 1 );
        }

        break;

      case 'l':
        optarg_len = strlen( optarg ) + 1;
        memcpy( dummy, optarg, optarg_len );
        this_proc_param = this_proc_param->next = (ProcessParamsTable *)
          calloc( 1, sizeof(ProcessParamsTable) );
        snprintf( this_proc_param->program, LIGOMETA_PROGRAM_MAX, "%s",
            PROGRAM_NAME );
        snprintf( this_proc_param->param, LIGOMETA_PARAM_MAX, "--l-distr" );
        snprintf( this_proc_param->type, LIGOMETA_TYPE_MAX, "string" );
        snprintf( this_proc_param->value, LIGOMETA_VALUE_MAX, "%s",
            optarg );

        if (!strcmp(dummy, "source"))
        {
          lDistr=locationFromSourceFile;
        }
        else if (!strcmp(dummy, "exttrig"))
        {
          lDistr=locationFromExttrigFile;
        }
        else if (!strcmp(dummy, "random"))
        {
          lDistr=uniformSkyLocation;
        }
        else if(!strcmp(dummy, "fixed"))
        {
          lDistr=fixedSkyLocation;
        }
        else if(!strcmp(dummy, "ipn"))
        {
          lDistr=locationFromIPNFile;
        }
	else
        {
          fprintf( stderr, "invalid argument to --%s:\n"
              "unknown location distribution: "
              "%s must be one of (source, random)\n",
              long_options[option_index].name, optarg );
          exit( 1 );
        }

        break;

      case 'H':
        /* Turn on galaxy catalog completion function */
        srcComplete = 1;
        srcCompleteDist = (REAL8) atof( optarg );
        break;

      case '.':
        /* Create a text file of completed catalog */
        makeCatalog = 1;
        break;

      case 'v':
        /* fixed location (longitude) */
        longitude =  atof( optarg )*LAL_PI_180 ;
        if (longitude <= (  LAL_PI + epsAngle ) && \
            longitude >= ( -LAL_PI - epsAngle ))
        {
          this_proc_param = this_proc_param->next =
            next_process_param( long_options[option_index].name,
                "float", "%e", longitude );
        }
        else
        {
          fprintf(stderr,"invalid argument to --%s:\n"
                  "%s must be between -180. and 180. degrees\n",
                  long_options[option_index].name, optarg );
          exit( 1 );
        }
        break;

      case 'z':
        /* fixed location (latitude) */
        latitude = (REAL4) atof( optarg )*LAL_PI_180;
        if (latitude <= (  LAL_PI/2. + epsAngle ) && \
            latitude >= ( -LAL_PI/2. - epsAngle ))
        {
          this_proc_param = this_proc_param->next =
            next_process_param( long_options[option_index].name,
                "float", "%e", latitude );
        }
        else
        {
          fprintf(stderr,"invalid argument to --%s:\n"
                  "%s must be between -90. and 90. degrees\n",
                  long_options[option_index].name, optarg );
          exit( 1 );
        }
        break;

      case 'I':
        optarg_len = strlen( optarg ) + 1;
        memcpy( dummy, optarg, optarg_len );
        this_proc_param = this_proc_param->next = (ProcessParamsTable *)
          calloc( 1, sizeof(ProcessParamsTable) );
        snprintf( this_proc_param->program, LIGOMETA_PROGRAM_MAX, "%s",
            PROGRAM_NAME );
        snprintf( this_proc_param->param, LIGOMETA_PARAM_MAX, "--i-distr" );
        snprintf( this_proc_param->type, LIGOMETA_TYPE_MAX, "string" );
        snprintf( this_proc_param->value, LIGOMETA_VALUE_MAX, "%s",
            optarg );

        if (!strcmp(dummy, "uniform"))
        {
          iDistr=uniformInclDist;
        }
        else if (!strcmp(dummy, "gaussian"))
        {
          iDistr=gaussianInclDist;
        }
        else if (!strcmp(dummy, "fixed"))
        {
          iDistr=fixedInclDist;
        }
        else
        {
          fprintf( stderr, "invalid argument to --%s:\n"
              "unknown inclination distribution: "
              "%s must be one of (uniform, gaussian, fixed)\n",
              long_options[option_index].name, optarg );
          exit( 1 );
        }

        break;


      case 'B':
        /* gaussian width for inclination */
        inclStd = (REAL4) atof( optarg );
        if ( inclStd <= 0 )
        {
          fprintf( stderr, "invalid argument to --%s:\n"
              "inclination gaussian width must be greater than 0: "
              "(%f specified)\n",
              long_options[option_index].name, dmax );
          exit( 1 );
        }
        this_proc_param = this_proc_param->next =
          next_process_param( long_options[option_index].name,
              "float", "%e", inclStd );
        break;

      case 'C':
        /* fixed angle of inclination */
        fixed_inc = (REAL4) atof( optarg )/180.*LAL_PI;
        this_proc_param = this_proc_param->next =
          next_process_param( long_options[option_index].name,
              "float", "%e", fixed_inc );
        break;

     case 1001:
        /* maximum  angle of inclination */
        max_inc = (REAL4) atof( optarg )/180.*LAL_PI;
        if ( (atof(optarg) < 0.) || (atof(optarg) >= 180.) ) {
          fprintf( stderr, "invalid argument to --%s:\n"
              "maximum inclination angle must be between 0 and 180 degrees:"
              "(%s specified)\n",
              long_options[option_index].name, optarg );
          exit( 1 );
        }
        this_proc_param = this_proc_param->next =
          next_process_param( long_options[option_index].name,
              "float", "%e", max_inc );
        break;

      case 'S':
        /* set the polarization angle */
        psi = (REAL4) atof( optarg )/180.*LAL_PI;
        if ( (atof(optarg) < 0.) || (atof(optarg) >= 360.) ) {
          fprintf( stderr, "invalid argument to --%s:\n"
              "polarization angle must be between 0 and 360 degrees: "
              "(%s specified)\n",
              long_options[option_index].name, optarg );
          exit( 1 );
        }
        this_proc_param = this_proc_param->next =
          next_process_param( long_options[option_index].name,
              "float", "%e", psi );
        break;

      case 'P':
        optarg_len = strlen( optarg ) + 1;
        outputFileName = calloc( 1, optarg_len * sizeof(char) );
        memcpy( outputFileName, optarg, optarg_len * sizeof(char) );
        this_proc_param = this_proc_param->next =
          next_process_param( long_options[option_index].name, "string",
              "%s", optarg );
        break;

      case 'g':
        minSpin1 = atof( optarg );
        this_proc_param = this_proc_param->next =
          next_process_param( long_options[option_index].name,
              "float", "%le", minSpin1 );
        break;

      case 'G':
        maxSpin1 = atof( optarg );
        this_proc_param = this_proc_param->next =
          next_process_param( long_options[option_index].name,
              "float", "%le", maxSpin1 );
        break;

      case 'Q':
        minKappa1 = atof( optarg );
        this_proc_param = this_proc_param->next =
          next_process_param( long_options[option_index].name,
              "float", "%le", minKappa1 );
        break;

      case 'R':
        maxKappa1 = atof( optarg );
        this_proc_param = this_proc_param->next =
          next_process_param( long_options[option_index].name,
              "float", "%le", maxKappa1 );
        break;

      case 'X':
        minabsKappa1 = atof( optarg );
        this_proc_param = this_proc_param->next =
          next_process_param( long_options[option_index].name,
              "float", "%le", minabsKappa1 );
        break;

      case 'Y':
        maxabsKappa1 = atof( optarg );
        this_proc_param = this_proc_param->next =
          next_process_param( long_options[option_index].name,
              "float", "%le", maxabsKappa1 );
        break;

      case 'u':
        minSpin2 = atof( optarg );
        this_proc_param = this_proc_param->next =
          next_process_param( long_options[option_index].name,
              "float", "%le", minSpin2 );
        break;

      case 'U':
        maxSpin2 = atof( optarg );
        this_proc_param = this_proc_param->next =
          next_process_param( long_options[option_index].name,
              "float", "%le", maxSpin2 );
        break;

      case 'V':
        /* print version information and exit */
        fprintf( stdout, "LIGO/LSC inspiral injection engine\n");
        XLALOutputVersionString(stderr, 0);
        exit( 0 );
        break;

      case 'T':
        /* enable spinning injections */
        this_proc_param = this_proc_param->next =
          next_process_param( long_options[option_index].name, "string",
              "" );
        spinInjections = 1;
        break;

      case 'W':
        /* disable spinning injections */
        this_proc_param = this_proc_param->next =
          next_process_param( long_options[option_index].name, "string",
              "" );
        spinInjections = 0;
        break;

      case '@':
        /* enforce aligned spins */
        this_proc_param = this_proc_param->next =
          next_process_param( long_options[option_index].name, "string",
              "" );
        spinAligned = 1;
        break;

      case '}':
        /* enable band-passing */
        this_proc_param = this_proc_param->next =
          next_process_param( long_options[option_index].name, "string",
              "" );
        bandPassInj = 1;
        break;

      case '{':
        /* write out a sim_ringdown table */
        this_proc_param = this_proc_param->next =
          next_process_param( long_options[option_index].name, "string",
              "" );
        writeSimRing = 1;
        break;

      case '*':
        /* Set injection tapering */
        if ( ! strcmp( "start", optarg ) )
        {
            taperInj = INSPIRAL_TAPER_START;
        }
        else if ( ! strcmp( "end", optarg ) )
        {
            taperInj = INSPIRAL_TAPER_END;
        }
        else if ( ! strcmp( "startend", optarg ) )
        {
            taperInj = INSPIRAL_TAPER_STARTEND;
        }
        else
        {
            fprintf( stderr, "invalid argument to --%s:\n"
                    "unknown option specified: %s\n"
                    "(Must be one of start|end|startend)\n",
                    long_options[option_index].name, optarg );
        }
        this_proc_param = this_proc_param->next =
                next_process_param( long_options[option_index].name,
                        "string", optarg );
        break;

      case '1':
        minSNR = atof( optarg );

        if ( minSNR < 2 )
        {
          fprintf(stderr,"invalid argument to --%s:\n"
                  "%s must be greater than 2\n",
                  long_options[option_index].name, optarg );

          exit( 1 );
        }
        this_proc_param = this_proc_param->next =
          next_process_param( long_options[option_index].name,
              "float", "%le", minSNR );

        break;
      case '2':
        maxSNR = atof( optarg );
        if ( maxSNR < 2 )
        {
          fprintf(stderr,"invalid argument to --%s:\n"
                  "%s must be greater than 2\n",
                  long_options[option_index].name, optarg );

          exit( 1 );
        }

        this_proc_param = this_proc_param->next =
          next_process_param( long_options[option_index].name,
              "float", "%le", maxSNR );
        break;
      case '3':
        optarg_len = strlen( optarg ) + 1;
        ifos       = calloc( 1, optarg_len * sizeof(char) );
        memcpy( ifos, optarg, optarg_len * sizeof(char) );
        this_proc_param = this_proc_param->next =
          next_process_param( long_options[option_index].name, "string",
              "%s", optarg );
        break;

      case 'h':
        print_usage(argv[0]);
        exit( 0 );
        break;

      case '?':
        print_usage(argv[0]);
        exit( 1 );
        break;

      case '^':
        optarg_len = strlen( optarg ) + 1;
        IPNSkyPositionsFile = calloc( 1, optarg_len * sizeof(char) );
        memcpy( IPNSkyPositionsFile, optarg, optarg_len * sizeof(char) );
        this_proc_param = this_proc_param->next =
          next_process_param( long_options[option_index].name, "string",
              "%s", optarg );
        break;


      default:
        fprintf( stderr, "unknown error while parsing options\n" );
        print_usage(argv[0]);
        exit( 1 );
    }
  }

  /* must set MW flag */
  if ( mwLuminosity < 0  && dDistr == distFromSourceFile )
  {
    fprintf( stderr,
        "Must specify either --enable-milkyway LUM or --disable-milkyway\n"\
        " when using --d-distr=source\n" );
    exit( 1 );
  }

  if (gpsStartTime.gpsSeconds==-1 || gpsEndTime.gpsSeconds==-1)
  {
    fprintf( stderr,
        "Must specify both --gps-start-time and --gps-end-time.\n");
    exit( 1 );
  }

  gpsDuration=gpsEndTime.gpsSeconds-gpsStartTime.gpsSeconds;

  if ( dDistr == unknownDistanceDist )
  {
    fprintf(stderr,"Must specify a distance distribution (--d-distr).\n");
    exit( 1 );
  }

  if ( lDistr == unknownLocationDist )
  {
    fprintf(stderr,"Must specify a location distribution (--l-distr).\n");
    exit( 1 );
  }

  if ( lDistr == fixedSkyLocation && longitude == 181. )
  {
    fprintf(stderr,
        "Must specify both --longitude and --latitude when using \n"\
        "--l-distr=fixed\n");
    exit( 1 );
  }

  if ( lDistr == fixedSkyLocation && latitude == 91. )
  {
    fprintf(stderr,
        "Must specify both --longitude and --latitude when using \n"\
        "--l-distr=fixed\n");
    exit( 1 );
  }

  if ( mDistr == unknownMassDist )
  {
    fprintf(stderr,"Must specify a mass distribution (--m-distr).\n");
    exit( 1 );
  }

  if ( iDistr == unknownInclDist )
  {
    fprintf(stderr,"Must specify an inclination distribution (--i-distr).\n");
    exit( 1 );
  }

  /* if using source file, check that file and MW choice selected */
  if ( dDistr==distFromSourceFile || lDistr==locationFromSourceFile )
  {
    if ( ! sourceFileName )
    {
      fprintf( stderr,
          "Must specify --source-file when using --d-distr source \n" );
      exit( 1 );
    }

    /* read the source distribution here */
    read_source_data( sourceFileName );

   /* complete the galaxy catalog */
   if (srcComplete == 1)
    {
    sourceComplete();
    }
  }

  /* if using IPN sky points file, check that file exists and read it */
  if ( lDistr==locationFromIPNFile )
  {
    if ( ! IPNSkyPositionsFile )
    {
      fprintf( stderr,
          "Must specify --ipn-file when using IPN sky points distribution \n" );
      exit( 1 );
    }

    /* read the source distribution here */
   read_IPN_grid_from_file( IPNSkyPositionsFile );
  }
  /* If we're distributing over snr make sure we have everything */
  if ( minSNR > -1 || maxSNR > -1 || logSNR || ifos )
  {
    if ( minSNR == -1 || maxSNR == -1 || ifos == NULL )
    {
      fprintf( stderr,
        "Must provide all of --min-snr, --max-snr and --ifos to distribute by SNR\n" );
      exit( 1 );
    }

    if ( maxSNR <= minSNR )
    {
      fprintf( stderr, "max SNR must be greater than min SNR\n");
      exit( 1 );
    }

    if ( logSNR )
    {
      minSNR = log(minSNR);
      maxSNR = log(maxSNR);
    }
  }

  /* check if the source file is specified for distance but NOT for
     location */
  if ( dDistr==distFromSourceFile && lDistr!=locationFromSourceFile )
  {
    fprintf( stderr,
        "WARNING: source file specified for distance "
        "but NOT for location. This might give strange distributions\n" );
  }

  /* check if the location file is specified for location but NOT for
   * distances: GRB case */
  if ( dDistr!=distFromSourceFile && lDistr==locationFromSourceFile &&
      mwLuminosity>0.0 )
  {
    fprintf( stderr,
        "WARNING: source file specified for locations "
        "but NOT for distances, while Milky Way injections "
        "are allowed. This might give strange distributions\n" );
  }


  /* check selection of masses */
  if ( !massFileName && mDistr==massFromSourceFile )
  {
    fprintf( stderr,
        "Must specify either a file contining the masses (--mass-file) "
        "or choose another mass-distribution (--m-distr).\n" );
    exit( 1 );
  }
  if ( !nrFileName && mDistr==massFromNRFile )
  {
    fprintf( stderr,
        "Must specify either a file contining the masses (--nr-file) "
        "or choose another mass-distribution (--m-distr).\n" );
    exit( 1 );
  }


  /* read the masses from the mass file here */
  if ( massFileName && mDistr==massFromSourceFile )
  {
    read_mass_data( massFileName );
  }

  if ( nrFileName && mDistr==massFromNRFile )
  {
    read_nr_data ( nrFileName );
  }

  /* read in the data from the external trigger file */
  if ( lDistr == locationFromExttrigFile && !exttrigFileName )
  {
    fprintf( stderr,
        "If --l-distr exttrig is specified, must specify "
        "external trigger XML file using --exttrig-file.\n");
    exit( 1 );
  }
  if ( lDistr == locationFromExttrigFile && exttrigFileName )
  {
    numExtTriggers=LALExtTriggerTableFromLIGOLw( &exttrigHead, exttrigFileName,
        0, 1);
    fprintf(stderr,
              "Number of triggers read from the external trigger file: %d\n",
               numExtTriggers);

    if (numExtTriggers>1)
    {
      fprintf(stderr,
                "WARNING: Only 1 external trigger expected in the file '%s'",
                 exttrigFileName );
    }
    if (numExtTriggers==0)
    {
      fprintf(stderr,
                "ERROR: No external trigger found in file '%s'",
                 exttrigFileName );

      exit(1);
    }
  }

  /* check inclination distribution */
  if ( ( iDistr == gaussianInclDist ) && ( inclStd < 0.0 ) )
  {
    fprintf( stderr,
        "Must specify width for gaussian inclination distribution; \n"
        "use --incl-std.\n" );
    exit( 1 );
  }
  if ( ( iDistr == fixedInclDist ) && ( fixed_inc < 0. ) )
  {
    fprintf( stderr,
        "Must specify an inclination if you want it fixed; \n"
        "use --fixed-inc.\n" );
    exit( 1 );
  }

  /* require --f-lower be explicit */
  if ( fLower <= 0.0 )
  {
    fprintf( stderr, "--f-lower must be specified and non-zero\n" );
    exit( 1 );
  }


  /* check for gaussian mass distribution parameters */
  if ( mDistr==gaussianMassDist && (meanMass1 <= 0.0 || massStdev1 <= 0.0 ||
        meanMass2 <= 0.0 || massStdev2 <= 0.0))
  {
    fprintf( stderr,
        "Must specify --mean-mass1/2 and --stdev-mass1/2 if choosing \n"
        " --m-distr=gaussian\n" );
    exit( 1 );
  }

  /* check if the mass area is properly specified */
  if ( (mDistr!=gaussianMassDist && mDistr!=fixMasses) && 
      (minMass1 <=0.0 || minMass2 <=0.0 || maxMass1 <=0.0 || maxMass2 <=0.0) )
  {
    fprintf( stderr,
        "Must specify --min-mass1/2 and --max-mass1/2 if choosing"
        " --m-distr not gaussian or fixMasses\n" );
    exit( 1 );
  }

  /* check if the maximum total mass is properly specified */
  if ( mDistr!=gaussianMassDist && maxMtotal<(minMass1 + minMass2 ))
  {
    fprintf( stderr,
        "Maximum total mass must be larger than minMass1+minMass2\n");
    exit( 1 );
  }

  /* check if total mass is specified */
  if ( mDistr != fixMasses && maxMtotal<0.0)
  {
    fprintf( stderr,
        "Must specify --max-mtotal.\n" );
    exit( 1 );
  }
  if ( mDistr != fixMasses && minMtotal<0.0)
  {
    fprintf( stderr,
        "Must specify --min-mtotal.\n" );
    exit( 1 );
  }

  /* check if mass ratios are specified */
  if ( (mDistr==uniformTotalMassRatio || mDistr==logMassUniformTotalMassRatio
        || mDistr==uniformTotalMassFraction)
      && (minMassRatio < 0.0 || maxMassRatio < 0.0) )
  {
    fprintf( stderr,
        "Must specify --min-mass-ratio and --max-mass-ratio if choosing \n"
        " --m-distr=totalMassRatio or --m-distr=logTotalMassUniformMassRatio \n"
        " or --m-distr=totalMassFraction\n");
    exit( 1 );
  }

  if ( dDistr!=distFromSourceFile && (dmin<0.0 || dmax<0.0) )
  {
    fprintf( stderr,
        "Must specify --min-distance and --max-distance if \n"
        "--d-distr is not source.\n" );
    exit( 1 );
  }

  if ( dDistr==sfr && (dmax<0.2 || dmax>1.0) )
  {
    fprintf( stderr,
        "Maximal redshift can only take values between 0.2 and 1.\n" );
    exit( 1 );
  }

  /* check if number of grid points is specified */
  if ( mDistr==m1m2SquareGrid )
  {
    if ( pntMass1<2 || pntMass2<2 )
    {
    fprintf( stderr, "--mass1-points and --mass2-points must be specified "
        "and >= 2 if --m-distr=m1m2SquareGrid \n" );
    exit( 1 );
    }
    else
    {
      deltaMass1 = ( maxMass1 - minMass1 ) / (REAL4) ( pntMass1 -1 );
      deltaMass2 = ( maxMass2 - minMass2 ) / (REAL4) ( pntMass2 -1 );
    }
  }

  /* check if fixed-mass1 and fixed-mass2 are specified */
  if ( mDistr==fixMasses && ( fixedMass1<0.0 || fixedMass2<0.0 ) )
  {
    fprintf( stderr, "--fixed-mass1 and --fixed-mass2 must be specified "
        "and >= 0 if --m-distr=fixMasses\n" );
    exit( 1 );
  }

  /* check if waveform is specified */
  if ( !*waveform )
  {
    fprintf( stderr, "No waveform specified (--waveform).\n" );
    exit( 1 );
  }

  if ( spinInjections==-1 && mDistr != massFromNRFile )
  {
    fprintf( stderr,
        "Must specify --disable-spin or --enable-spin\n"
        "unless doing NR injections\n" );
    exit( 1 );
  }

  if ( spinInjections==0 && spinAligned==1 )
  {
    fprintf( stderr,
        "Must enable spin to obtain aligned spin injections.\n" );
    exit( 1 );
  }

  if ( spinInjections==1 && strncmp(waveform, "IMRPhenomB", 10)==0 && spinAligned==-1 )
  {
    fprintf( stderr,
        "Spinning IMRPhenomB injections must have the --aligned option.\n" );
    exit( 1 );
  }

  if ( spinInjections==1 && spinAligned==1 && strncmp(waveform, "IMRPhenomB", 10)
    && strncmp(waveform, "SpinTaylor", 10) )
  {
    fprintf( stderr,
        "Sorry, I only know to make spin aligned injections for \n"
        "IMRPhenomB, SpinTaylor and SpinTaylorFrameless waveforms.\n" );
    exit( 1 );
  }

  if ( spinInjections==1 )
  {
    /* check that spins are in range 0 - 1 */
    if (minSpin1 < 0. || minSpin2 < 0. || maxSpin1 > 1. || maxSpin2 >1.)
    {
      fprintf( stderr,
          "Spins can only take values between 0 and 1.\n" );
      exit( 1 );
    }

    /* check max and mins are the correct way around */
    if (minSpin1 > maxSpin1 || minSpin2 > maxSpin2 )
    {
      fprintf( stderr,
          "Minimal spins must be less than maximal spins.\n" );
      exit( 1 );
    }

    /* check that selection criteria for kappa are unique */
    if ( (minKappa1 > -1.0 || maxKappa1 < 1.0) &&
        (minabsKappa1 > 0.0 || maxabsKappa1 < 1.0) )
    {
      fprintf( stderr,
          "Either the options [--min-kappa1,--max-kappa1] or\n"
          "[--min-abskappa1,--max-abskappa1] can be specified\n" );
      exit( 1 );
    }

    /* check that kappa is in range */
    if (minKappa1 < -1.0 || maxKappa1 > 1.0)
    {
      fprintf( stderr,
          "Kappa can only take values between -1 and +1\n" );
      exit( 1 );
    }
    /* check that kappa min-max are set correctly */
    if (minKappa1 > maxKappa1)
    {
      fprintf( stderr,
          "Minimal kappa must be less than maximal kappa\n" );
      exit( 1 );
    }
    /* check that abskappa is in range */
    if (minabsKappa1 < 0.0 || maxabsKappa1 > 1.0)
    {
      fprintf( stderr,
      "The absolute value of kappa can only take values between 0 and +1\n" );
      exit( 1 );
    }
    /* check that kappa min-max are set correctly */
    if (minabsKappa1 > maxabsKappa1)
    {
      fprintf( stderr,
          "Minimal kappa must be less than maximal kappa\n" );
      exit( 1 );
    }
  }
        
  if( dDistr==sfr && localRate > 0.)
  {
    /* calculate mean time step from the SFR  */
        meanTimeStep = mean_time_step_sfr(dmax,localRate);
  }

  if (meanTimeStep<=0)
  {
    fprintf( stderr,
             "Minimum time step value must be larger than zero\n" );
    exit( 1 );
  }

  if (timeInterval > 0. && tDistr == LALINSPIRAL_EXPONENTIAL_TIME_DIST)
  {
    fprintf( stderr,
         "time interval must be zero\n" );
    exit( 1 );
  }


  if ( userTag && outCompress )
  {
    snprintf( fname, sizeof(fname), "HL-INJECTIONS_%d_%s-%d-%ld.xml.gz",
        rand_seed, userTag, gpsStartTime.gpsSeconds, gpsDuration );
  }
  else if ( userTag && !outCompress )
  {
    snprintf( fname, sizeof(fname), "HL-INJECTIONS_%d_%s-%d-%ld.xml",
        rand_seed, userTag, gpsStartTime.gpsSeconds, gpsDuration );
  }
  else if ( !userTag && outCompress )
  {
    snprintf( fname, sizeof(fname), "HL-INJECTIONS_%d-%d-%ld.xml.gz",
        rand_seed, gpsStartTime.gpsSeconds, gpsDuration );
  }
  else
  {
    snprintf( fname, sizeof(fname), "HL-INJECTIONS_%d-%d-%ld.xml",
        rand_seed, gpsStartTime.gpsSeconds, gpsDuration );
  }
  if ( outputFileName )
  {
    snprintf( fname, sizeof(fname), "%s",
        outputFileName);
  }

  /* increment the random seed by the GPS start time:*/
  rand_seed += gpsStartTime.gpsSeconds;

  /* set up the LAL random number generator */
  LALCreateRandomParams( &status, &randParams, rand_seed );

  this_proc_param = procparams.processParamsTable;
  procparams.processParamsTable = procparams.processParamsTable->next;
  free( this_proc_param );

  /* create the first injection */
  simTable = injections.simInspiralTable = (SimInspiralTable *)
    calloc( 1, sizeof(SimInspiralTable) );

  simRingTable = ringparams.simRingdownTable = (SimRingdownTable *)
    calloc( 1, sizeof(SimRingdownTable) );

  /* set redshift to zero */
  redshift=0.;

  /* set mass distribution parameters to their value at z = 0 */
  minMass10 = minMass1;
  maxMass10 = maxMass1;
  minMass20 = minMass2;
  maxMass20 = maxMass2;
  minMtotal0 = minMtotal;
  maxMtotal0 = maxMtotal;
  meanMass10 = meanMass1;
  meanMass20 = meanMass2;
  massStdev10 = massStdev1;
  massStdev20 = massStdev2;

  /* calculate the maximal value of the probability distribution of the redshift */        
  if (dDistr == sfr)
  {
    pzmax = probability_redshift(dmax);
  }

  /* loop over parameter generation until end time is reached */
  ninj = 0;
  ncount = 0;
  currentGpsTime = gpsStartTime;
  while ( 1 )
  {
    /* increase counter */
    ninj++;

    /* store time in table */
    simTable=XLALRandomInspiralTime( simTable, randParams,
        currentGpsTime, timeInterval );

    /* populate waveform and other parameters */
    memcpy( simTable->waveform, waveform,
        sizeof(CHAR) * LIGOMETA_WAVEFORM_MAX );
    simTable->f_lower = fLower;
    simTable->amp_order = amp_order;

    /* draw redshift */
    if (dDistr==sfr)
    {
          redshift= drawRedshift(dmin,dmax,pzmax);        

      minMass1 = redshift_mass(minMass10, redshift);
      maxMass1 = redshift_mass(maxMass10, redshift);
      meanMass1 = redshift_mass(meanMass10, redshift);
      massStdev1 = redshift_mass(massStdev10, redshift);
      minMass2 = redshift_mass(minMass20, redshift);
      maxMass2 = redshift_mass(maxMass20, redshift);
      meanMass2 = redshift_mass(meanMass20, redshift);
      massStdev2 = redshift_mass(massStdev20, redshift);
      minMtotal = redshift_mass(minMtotal0, redshift);
      maxMtotal = redshift_mass(maxMtotal0, redshift);
    }

    /* populate masses */
    if ( mDistr==massFromSourceFile )
    {
      drawMassFromSource( simTable );
    }
    else if ( mDistr==massFromNRFile )
    {
      if (ninjaMass)
        drawMassSpinFromNRNinja2( simTable );
      else
        drawMassSpinFromNR( simTable );
    }
    else if ( mDistr==gaussianMassDist )
    {
      simTable=XLALGaussianInspiralMasses( simTable, randParams,
          minMass1, maxMass1,
          meanMass1, massStdev1,
          minMass2, maxMass2,
          meanMass2, massStdev2);
    }
    else if ( mDistr==uniformTotalMassRatio )
    {
      simTable=XLALRandomInspiralTotalMassRatio(simTable, randParams,
          mDistr, minMtotal, maxMtotal, minMassRatio, maxMassRatio );
    }
    else if ( mDistr==logMassUniformTotalMassRatio )
    {
      simTable=XLALRandomInspiralTotalMassRatio(simTable, randParams,
          mDistr, minMtotal, maxMtotal, minMassRatio, maxMassRatio );
    }
    else if ( mDistr==m1m2SquareGrid )
    {
      simTable=XLALm1m2SquareGridInspiralMasses( simTable, minMass1, minMass2,
          minMtotal, maxMtotal, deltaMass1, deltaMass2, pntMass1, pntMass2, 
          ninj, &ncount);
    }
    else if ( mDistr==fixMasses )
    {
      simTable=XLALFixedInspiralMasses( simTable, fixedMass1, fixedMass2);
    }
    else if ( mDistr==uniformTotalMassFraction )
    {
      simTable=XLALRandomInspiralTotalMassFraction(simTable, randParams,
          mDistr, minMtotal, maxMtotal, minMassRatio, maxMassRatio );
    }
    else {
      simTable=XLALRandomInspiralMasses( simTable, randParams, mDistr,
          minMass1, maxMass1,
          minMass2, maxMass2,
          minMtotal, maxMtotal);
    }
    
    /* draw location and distances */
    drawFromSource( &drawnRightAscension, &drawnDeclination, &drawnDistance,
        drawnSourceName );
    drawFromIPNsim( &drawnRightAscension, &drawnDeclination );

    /* populate distances */
    if ( dDistr == distFromSourceFile )
    {
      if ( dmax > 0 )
      {
        while ( drawnDistance > dmax/1000.0 )
        {
          drawFromSource( &drawnRightAscension, &drawnDeclination,
                          &drawnDistance, drawnSourceName );
        }
      }
      simTable->distance = drawnDistance;
    }
    else if (dDistr == sfr )
    {
       /* fit of luminosity distance  between z=0-1, in Mpc for h0=0.7, omega_m=0.3, omega_v=0.7*/
       simTable->distance = luminosity_distance(redshift);
    }
    else
    {
      simTable=XLALRandomInspiralDistance(simTable, randParams,
          dDistr, dmin/1000.0, dmax/1000.0);
    }
    /* Scale by chirp mass if desired, relative to a 1.4,1.4 object */
    if (useChirpDist)
    {
      REAL4 scaleFac;
      scaleFac = simTable->mchirp/(2.8*pow(0.25,0.6));
      simTable->distance = simTable->distance*pow(scaleFac,5./6.);
    }

    /* populate location */
    if ( lDistr == locationFromSourceFile )
    {
      simTable->longitude = drawnRightAscension;
      simTable->latitude  = drawnDeclination;
      memcpy( simTable->source, drawnSourceName,
          sizeof(CHAR) * LIGOMETA_SOURCE_MAX );
    }
    else if ( lDistr == locationFromExttrigFile )
    {
      drawLocationFromExttrig( simTable );
    }
    else if ( lDistr == fixedSkyLocation)
    {
      simTable->longitude = longitude;
      simTable->latitude = latitude;
    }
    else if (lDistr == uniformSkyLocation)
    {
      simTable=XLALRandomInspiralSkyLocation(simTable, randParams);
    }
    else if ( lDistr == locationFromIPNFile )
    {
      IPNgmst1 = XLALGreenwichMeanSiderealTime(&IPNgpsTime);
      IPNgmst2 = XLALGreenwichMeanSiderealTime(&simTable->geocent_end_time);
      simTable->longitude = drawnRightAscension - IPNgmst1 + IPNgmst2;
      simTable->latitude  = drawnDeclination;
    }
    else
    {
      fprintf( stderr,
               "Unknown location distribution specified. Possible choices: "
               "source, exttrig, random or fixed\n" );
      exit( 1 );
    }

    /* populate polarization, inclination, and coa_phase */
    do
    {
      simTable=XLALRandomInspiralOrientation(simTable, randParams,
                                             iDistr, inclStd);
    } while ( (fabs(cos(simTable->inclination))<cos(max_inc)) );

    /* override inclination */
    if ( iDistr == fixedInclDist )
    {
      simTable->inclination = fixed_inc;
    }

    /* override polarization angle */
    if ( psi != -1.0 )
    {
      simTable->polarization = psi;
    }

    /* populate spins, if required */
    if (spinInjections)
    {
      if (spinAligned==1)
      {
        if (strncmp(waveform, "IMRPhenomB", 10)==0)
          alignInj = alongzAxis;
        else if (strncmp(waveform, "SpinTaylor", 10)==0)
          alignInj = inxzPlane;
        else
        {
          fprintf( stderr, "Unknown waveform type for aligned spin injections.\n" );
          exit( 1 );
        }
      }
      else
        alignInj = notAligned;
      simTable = XLALRandomInspiralSpins( simTable, randParams,
          minSpin1, maxSpin1,
          minSpin2, maxSpin2,
          minKappa1, maxKappa1,
          minabsKappa1, maxabsKappa1,
          alignInj );
    }

    if ( ifos != NULL )
    {
        targetSNR = minSNR + (maxSNR - minSNR) * XLALUniformDeviate( randParams );
        if ( logSNR )
          targetSNR = exp(targetSNR);

        adjust_snr(simTable, targetSNR, ifos);

        /* TODO: for NINJA2, decide whether to call the above or
        adjust_snr_real8(simTable, targetSNR, ifos);
        */
    }

    /* populate the site specific information */
    LALPopulateSimInspiralSiteInfo( &status, simTable );

    /* populate the taper options */
    {
        switch (taperInj)
        {
            case INSPIRAL_TAPER_NONE:
                 snprintf( simTable->taper, LIGOMETA_WAVEFORM_MAX,
                         "%s", "TAPER_NONE");
                 break;
            case INSPIRAL_TAPER_START:
                 snprintf( simTable->taper, LIGOMETA_WAVEFORM_MAX,
                         "%s", "TAPER_START");
                 break;
            case INSPIRAL_TAPER_END:
                 snprintf( simTable->taper, LIGOMETA_WAVEFORM_MAX,
                         "%s", "TAPER_END");
                 break;
            case INSPIRAL_TAPER_STARTEND:
                 snprintf( simTable->taper, LIGOMETA_WAVEFORM_MAX,
                         "%s", "TAPER_STARTEND");
                 break;
            default: /* Never reach here */
                 fprintf( stderr, "unknown error while populating sim_inspiral taper options\n" );
                 exit(1);
        }

    }

    /* populate the bandpass options */
    simTable->bandpass = bandPassInj;


    /* populate the sim_ringdown table */
   if ( writeSimRing )
   {
       memcpy( simRingTable->waveform, "Ringdown",
          sizeof(CHAR) * LIGOMETA_WAVEFORM_MAX );
       memcpy( simRingTable->coordinates, "EQUATORIAL",
          sizeof(CHAR) * LIGOMETA_WAVEFORM_MAX );
       simRingTable->geocent_start_time = simTable->geocent_end_time;
       simRingTable->h_start_time = simTable->h_end_time;
       simRingTable->l_start_time = simTable->l_end_time;
       simRingTable->v_start_time = simTable->v_end_time;
       simRingTable->start_time_gmst = simTable->end_time_gmst;
       simRingTable->longitude = simTable->longitude;
       simRingTable->latitude = simTable->latitude;
       simRingTable->distance = simTable->distance;
       simRingTable->inclination = simTable->inclination;
       simRingTable->polarization = simTable->polarization;
       simRingTable->phase = 0;
       simRingTable->mass = XLALNonSpinBinaryFinalBHMass(simTable->eta, simTable->mass1, simTable->mass2);
       /* The final spin calc has been generalized so as to allow initially spinning systems*/
       /* simRingTable->spin = XLALNonSpinBinaryFinalBHSpin(simTable->eta); */
       simRingTable->spin = XLALSpinBinaryFinalBHSpin(simTable->eta, simTable->mass1, simTable->mass2,
          simTable->spin1x, simTable->spin2x,simTable->spin1y, simTable->spin2y, simTable->spin1z, simTable->spin2z);
       simRingTable->frequency = XLALBlackHoleRingFrequency( simRingTable->mass, simRingTable->spin);
       simRingTable->quality = XLALBlackHoleRingQuality(simRingTable->spin);
       simRingTable->epsilon = 0.01;
       simRingTable->amplitude = XLALBlackHoleRingAmplitude( simRingTable->frequency, simRingTable->quality, simRingTable->distance, simRingTable->epsilon );
       simRingTable->eff_dist_h = simTable->eff_dist_h;
       simRingTable->eff_dist_l = simTable->eff_dist_l;
       simRingTable->eff_dist_v = simTable->eff_dist_v;
       simRingTable->hrss = XLALBlackHoleRingHRSS( simRingTable->frequency, simRingTable->quality, simRingTable->amplitude, 2., 0. );
       // need hplus & hcross in each detector to populate these
       simRingTable->hrss_h = 0.; //XLALBlackHoleRingHRSS( simRingTable->frequency, simRingTable->quality, simRingTable->amplitude, 0., 0. );
       simRingTable->hrss_l = 0.; //XLALBlackHoleRingHRSS( simRingTable->frequency, simRingTable->quality, simRingTable->amplitude, 0., 0. );
       simRingTable->hrss_v = 0.; //XLALBlackHoleRingHRSS( simRingTable->frequency, simRingTable->quality, simRingTable->amplitude, 0., 0. );
    }

    /* increment current time, avoiding roundoff error;
       check if end of loop is reached */
    if (tDistr == LALINSPIRAL_EXPONENTIAL_TIME_DIST)
    {
      XLALGPSAdd( &currentGpsTime, -(REAL8 )meanTimeStep * log( XLALUniformDeviate(randParams) ) );
    }
    else
    {
      currentGpsTime = gpsStartTime;
      XLALGPSAdd( &currentGpsTime, ninj * meanTimeStep );
    }
    if ( XLALGPSCmp( &currentGpsTime, &gpsEndTime ) >= 0 )
      break;

  /* allocate and go to next SimInspiralTable */
    simTable = simTable->next = (SimInspiralTable *)
      calloc( 1, sizeof(SimInspiralTable) );
    simRingTable = simRingTable->next = (SimRingdownTable *)
      calloc( 1, sizeof(SimRingdownTable) );

  }


  /* destroy the structure containing the random params */
  LAL_CALL(  LALDestroyRandomParams( &status, &randParams ), &status);

  /* If we read from an external trigger file, free our external trigger.
     exttrigHead is guaranteed to have no children to free. */
  if ( exttrigHead != NULL ) {
    LALFree(exttrigHead);
  }

  /* destroy the NR data */
  if ( num_nr )
  {
    for( i = 0; i < num_nr; i++ )
    {
      LALFree( nrSimArray[i] );
    }
    LALFree( nrSimArray );
  }

  memset( &xmlfp, 0, sizeof(LIGOLwXMLStream) );


  LAL_CALL( LALOpenLIGOLwXMLFile( &status, &xmlfp, fname ), &status );
  XLALGPSTimeNow(&(proctable.processTable->end_time));
  LAL_CALL( LALBeginLIGOLwXMLTable( &status, &xmlfp, process_table ),
      &status );
  LAL_CALL( LALWriteLIGOLwXMLTable( &status, &xmlfp, proctable,
        process_table ), &status );
  LAL_CALL( LALEndLIGOLwXMLTable ( &status, &xmlfp ), &status );

  if ( procparams.processParamsTable )
  {
    LAL_CALL( LALBeginLIGOLwXMLTable( &status, &xmlfp, process_params_table ),
        &status );
    LAL_CALL( LALWriteLIGOLwXMLTable( &status, &xmlfp, procparams,
          process_params_table ), &status );
    LAL_CALL( LALEndLIGOLwXMLTable ( &status, &xmlfp ), &status );
  }

  if ( injections.simInspiralTable )
  {
    LAL_CALL( LALBeginLIGOLwXMLTable( &status, &xmlfp, sim_inspiral_table ),
        &status );
    LAL_CALL( LALWriteLIGOLwXMLTable( &status, &xmlfp, injections,
          sim_inspiral_table ), &status );
    LAL_CALL( LALEndLIGOLwXMLTable ( &status, &xmlfp ), &status );
  }

  if ( writeSimRing )
  {
    if ( ringparams.simRingdownTable )
    {
      LAL_CALL( LALBeginLIGOLwXMLTable( &status, &xmlfp, sim_ringdown_table ),
          &status );
      LAL_CALL( LALWriteLIGOLwXMLTable( &status, &xmlfp, ringparams,
          sim_ringdown_table ), &status );
      LAL_CALL( LALEndLIGOLwXMLTable ( &status, &xmlfp ), &status );
    }
  }

  LAL_CALL( LALCloseLIGOLwXMLFile ( &status, &xmlfp ), &status );

  if (source_data)
    LALFree(source_data);
  if (mass_data)
    LALFree(mass_data);
  if (skyPoints)
    LALFree(skyPoints);

  LALCheckMemoryLeaks();
  return 0;
}

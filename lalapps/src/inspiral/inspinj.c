/*
 *  Copyright (C) 2007 Chad Hanna, Alexander Dietz, Duncan Brown, Gareth Jones, Jolien Creighton, Nickolas Fotopoulos, Patrick Brady, Stephen Fairhurst
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
 * Author: Brown, D. A., Creighton, J. D. E. and Dietz A. 
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
#include <lal/LIGOMetadataUtils.h>
#include <lal/LIGOLwXMLRead.h>
#include <lal/LIGOLwXML.h>
#include <lal/Random.h>
#include <lal/AVFactories.h>
#include <lal/InspiralInjectionParams.h>
#include <processtable.h>
#include <lal/lalGitID.h>
#include <lalappsGitID.h>

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
LALSnprintf( this_proc_param->program, LIGOMETA_PROGRAM_MAX, "%s", \
    PROGRAM_NAME ); \
LALSnprintf( this_proc_param->param, LIGOMETA_PARAM_MAX, "--%s", \
    long_options[option_index].name ); \
LALSnprintf( this_proc_param->type, LIGOMETA_TYPE_MAX, "%s", pptype ); \
LALSnprintf( this_proc_param->value, LIGOMETA_VALUE_MAX, format, ppvalue );


/* 
 *  *********************************
 *  Definition of the prototypes 
 *  *********************************
 */
ProcessParamsTable *next_process_param( const char *name, const char *type,
    const char *fmt, ... );
void read_mass_data( char *filename );
void read_nr_data( char* filename );
void read_source_data( char* filename );
void drawFromSource( REAL8 *rightAscension,
    REAL8 *declination,
    REAL8 *distance,
    CHAR  name[LIGOMETA_SOURCE_MAX] );
void drawLocationFromExttrig( SimInspiralTable* table );
void drawMassFromSource( SimInspiralTable* table );
void drawMassSpinFromNR( SimInspiralTable* table );

/* 
 *  *************************************
 *  Defining of the used global variables
 *  *************************************
 */

DistanceDistribution    dDistr;
SkyLocationDistribution lDistr;
MassDistribution        mDistr;
InclDistribution        iDistr;

SimInspiralTable *simTable;

char *massFileName = NULL;
char *nrFileName = NULL;
char *sourceFileName = NULL;
char *outputFileName = NULL;
char *exttrigFileName = NULL;

INT4 outCompress = 0;

float mwLuminosity = -1;
REAL4 dmin= -1;
REAL4 dmax= -1;
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
REAL4 fixed_inc=0.0;
REAL4 psi=0.0;
REAL4 longitude=181.0;
REAL4 latitude=91.0;
REAL4 epsAngle=1e-7;
int spinInjections=-1;
REAL4 minSpin1=-1.0;
REAL4 maxSpin1=-1.0;
REAL4 minSpin2=-1.0;
REAL4 maxSpin2=-1.0;
REAL4 minKappa1=-1.0;
REAL4 maxKappa1=1.0;
REAL4 minabsKappa1=0.0;
REAL4 maxabsKappa1=1.0;
INT4 bandPassInj = 0;
InspiralApplyTaper taperInj = INSPIRAL_TAPER_NONE;


static LALStatus status;
static RandomParams* randParams=NULL;
INT4 numExtTriggers = 0;
ExtTriggerTable   *exttrigHead = NULL;

int num_source;
struct {
  char   name[LIGOMETA_SOURCE_MAX];
  REAL8 ra;
  REAL8 dec;
  REAL8 dist;
  REAL8 lum;
  REAL8 fudge;
} *source_data;

char MW_name[LIGOMETA_SOURCE_MAX] = "MW";
REAL8* fracVec  =NULL;
REAL8* ratioVec = NULL;
REAL8 norm=0;

int num_mass;
struct {
  REAL8 mass1;
  REAL8 mass2;
} *mass_data;

int num_nr = 0;
int i = 0;
SimInspiralTable **nrSimArray = NULL;

/* 
 *  *********************************
 *  Implementation of the code pieces  
 *  *********************************
 */

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
  LALSnprintf( pp->param, LIGOMETA_PARAM_MAX, "--%s", name );
  strncpy( pp->type, type, LIGOMETA_TYPE_MAX );
  va_start( ap, fmt );
  LALVsnprintf( pp->value, LIGOMETA_VALUE_MAX, fmt, ap );
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
      " [--write-compress]        write a compressed xml file\n"\
      "\n"\
      "Waveform details:\n"\
      " [--seed] randomSeed       seed for random number generator (default : 1)\n"\
      "  --f-lower freq           lower cut-off frequency.\n"\
      "  --waveform wfm           set waveform type to wfm\n"\
      "  --amp-order              set PN order in amplitude\n"\
      "\n"\
      "Time distribution information:\n"\
      "  --gps-start-time start   GPS start time for injections\n"\
      "  --gps-end-time end       GPS end time for injections\n"\
      "  [--time-step] step       space injections by average of step seconds\n"\
      "                           (default : 2630 / pi seconds)\n"\
      "  [--time-interval] int    distribute injections in an interval, int s\n"\
      "                           (default : 0 seconds)\n"\
      "\n"\
      "Source distribution information:\n"\
      "  --l-distr  locDist       set the source location distribution,\n"\
      "                           locDist must be one of:\n"\
      "                           source: use locations from source-file\n"\
      "                           exttrig: use external trigger file\n"\
      "                           random: uses random locations\n"\
      "                           fixed: set fixed location\n"\
      " [--longitude] longitude   read longitude if fixed value (degrees)\n"
      " [--latitude] latitude     read latitide if fixed value (degrees)\n"
      "  --d-distr distDist       set the distance distribution of injections\n"\
      "                           source: take distance from galaxy source file\n"\
      "                           uniform: uniform distribution in distance\n"\
      "                           log10: uniform distribution in log10(d) \n"\
      "                           volume: uniform distribution in volume\n"\
      "  --i-distr INCDIST        set the inclination distribution, must be either\n"\
      "                           uniform: distribute uniformly over arccos(i)\n"\
      "                           gaussian: gaussian distributed in arccos(i)\n"\
      "                           fixed: no distribution, fixed valued of (i)\n"\
      " --polarization psi        set the polarization angle for all \n"
      "                           injections (degrees)\n"\
      " [--inclStd]  incStd       std dev for gaussian inclination dist\n"\
      " [--fixed-inc]  fixed_inc  read inclination dist if fixed value (degrees)\n"\
      " [--source-file] sources   read source parameters from sources\n"\
      "                           requires enable/disable milkyway\n"\
      " [--enable-milkyway] lum   enables MW injections, set MW luminosity\n"\
      " [--disable-milkyway]      disables Milky Way injections\n"\
      " [--exttrig-file] exttrig  XML file containing external trigger\n"\
      " [--min-distance] DMIN     set the minimum distance to DMIN kpc\n"\
      " [--max-distance] DMAX     set the maximum distance to DMAX kpc\n"\
      "                           min/max distance required if d-distr not 'source'\n"\
      "\n"\
      "Mass distribution information:\n"\
      " --m-distr massDist        set the mass distribution of injections\n"\
      "                           must be one of:\n"\
      "                           source: using file containing list of mass pairs\n"\
      "                           nrwaves: using xml file with list of NR waveforms\n"\
      "                           (requires setting max/min total masses)\n"\
      "                           totalMass: uniform distribution in total mass\n"\
      "                           componentMass: uniform in m1 and m2\n"\
      "                           gaussian: gaussian mass distribution\n"\
      "                           log: log distribution in comonent mass\n"\
      "                           totalMassRatio: uniform distribution in total mass ratio\n"\
      " [--mass-file] mFile       read population mass parameters from mFile\n"\
      " [--nr-file] nrFile        read mass/spin parameters from xml nrFile\n"\
      " [--min-mass1] m1min       set the minimum component mass to m1min\n"\
      " [--max-mass1] m1max       set the maximum component mass to m1max\n"\
      " [--min-mass2] m2min       set the min component mass2 to m2min\n"\
      " [--max-mass2] m2max       set the max component mass2 to m2max\n"\
      " [--min-mtotal] minTotal   sets the minimum total mass to minTotal\n"\
      " [--max-mtotal] maxTotal   sets the maximum total mass to maxTotal\n"\
      " [--mean-mass1] m1mean     set the mean value for mass1\n"\
      " [--stdev-mass1] m1std     set the standard deviation for mass1\n"\
      " [--mean-mass2] m2mean     set the mean value for mass2\n"\
      " [--stdev-mass2] m2std     set the standard deviation for mass2\n"\
      " [--min-mratio] minr       set the minimum mass ratio\n"\
      " [--max-mratio] maxr       set the maximum mass ratio\n"\
      "\n"\
      "Spin distribution information:\n"\
      "  --disable-spin           disables spinning injections\n"\
      "  --enable-spin            enables spinning injections\n"\
      "                           One of these is required.\n"\
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
      "                           to abskappa1max (1.0)\n"\
      "\n"\
      "Tapering the injection waveform:\n"\
      "  [--taper-injection] OPT  Taper the inspiral template using option OPT\n"\
      "                            (start|end|startend) \n)"\
      "  [--band-pass-injection]  sets the tapering method of the injected waveform\n"\
      "\n");
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
  INT4               i = 0;

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

  for( i = 0, thisEvent=nrSimHead; i < num_nr; 
      ++i, thisEvent = thisEvent->next )
  {
    nrSimArray[i] = thisEvent;
    if (i > 0)
    {
      nrSimArray[i-1]->next = NULL;
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
  int i;

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

  i = 0;
  while ( fgets( line, sizeof( line ), fp ) )
    if ( line[0] == '#' )
      continue;
    else
    {
      char ra_sgn, dec_sgn;
      REAL8 ra_h, ra_m, dec_d, dec_m;
      int c;

      c = sscanf( line, "%s %c%le:%le %c%le:%le %le %le %le",
          source_data[i].name, &ra_sgn, &ra_h, &ra_m, &dec_sgn, &dec_d, &dec_m,
          &source_data[i].dist, &source_data[i].lum, &source_data[i].fudge );
      if ( c != 10 )
      {
        fprintf( stderr, "error parsing source datafile %s\n", sourceFileName );
        exit( 1 );
      }

      /* by convention, overall sign is carried only on hours/degrees entry */
      source_data[i].ra  = ( ra_h + ra_m / 60.0 ) * LAL_PI / 12.0;
      source_data[i].dec = ( dec_d + dec_m / 60.0 ) * LAL_PI / 180.0;

      if ( ra_sgn == '-' )
        source_data[i].ra *= -1;
      if ( dec_sgn == '-' )
        source_data[i].dec *= -1;
      ++i;
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
  for ( i = 0; i < num_source; ++i )
    norm += ratioVec[i] = source_data[i].lum * source_data[i].fudge;
  fracVec[0] = ratioVec[0] / norm;
  for ( i = 1; i < num_source; ++i )
    fracVec[i] = fracVec[i-1] + ratioVec[i] / norm;
}


/*
 *
 * functions to draw masses from mass distribution
 *
 */

void drawMassFromSource( SimInspiralTable* table )
{ 
  REAL4 m1, m2, eta;
  int index=0;

  /* choose masses from the mass-list */  
  index = (int)( num_mass * XLALUniformDeviate( randParams ) );
  m1=mass_data[index].mass1;
  m2=mass_data[index].mass2;

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
  int index=0;

  /* choose masses from the mass-list */  
  index = (int)( num_nr * XLALUniformDeviate( randParams ) );
  XLALRandomNRInjectTotalMass( table, randParams, minMtotal, maxMtotal, 
      nrSimArray[index]);
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
  int i;

  u=XLALUniformDeviate( randParams );

  /* draw from the source table */
  for ( i = 0; i < num_source; ++i )
  {
    if ( u < fracVec[i] )
    {
      /* put the parameters */
      *rightAscension = source_data[i].ra;
      *declination    = source_data[i].dec;
      *distance = source_data[i].dist/1000.0;
      memcpy( name, source_data[i].name,
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
 * functions to draw sky location from exttrig source file
 *
 */
void drawLocationFromExttrig( SimInspiralTable* table )
{
  LIGOTimeGPS timeGRB;  /* real time of the GRB */
  LALMSTUnitsAndAcc unitsAndAcc; 
  REAL4 ra_rad, de_rad;
  REAL8 gmst1, gmst2;  

  /* convert the position (stored as degree) to radians first */
  ra_rad = exttrigHead->event_ra  * LAL_PI_180;
  de_rad = exttrigHead->event_dec * LAL_PI_180;

  /* set units and accuracy for GMST calculation*/
  unitsAndAcc.accuracy = LALLEAPSEC_STRICT;
  unitsAndAcc.units = MST_RAD;

  /* populate the time structures */
  timeGRB.gpsSeconds     = exttrigHead->start_time;
  timeGRB.gpsNanoSeconds = exttrigHead->start_time_ns;

  LALGPStoGMST1( &status, &gmst1, &timeGRB, &unitsAndAcc );
  LALGPStoGMST1( &status, &gmst2, &table->geocent_end_time, &unitsAndAcc );

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
  LIGOTimeGPS gpsStartTime;
  LIGOTimeGPS gpsEndTime;
  LIGOTimeGPS currentGpsTime;
  long gpsDuration;

  REAL8 meanTimeStep = 2630 / LAL_PI; /* seconds between injections     */
  REAL8 timeInterval = 0;
  REAL4 fLower = -1;
  REAL4 eps=0.01;  /* needed for some awkward spinning injections */

  size_t ninj;
  int rand_seed = 1;

  /* waveform */
  CHAR waveform[LIGOMETA_WAVEFORM_MAX];
  CHAR dummy[256];
  INT4 amp_order = -1;
  /* xml output data */
  CHAR                  fname[256];
  CHAR                 *userTag = NULL;
  LALLeapSecAccuracy    accuracy = LALLEAPSEC_LOOSE;
  MetadataTable         proctable;
  MetadataTable         procparams;
  MetadataTable         injections;
  ProcessParamsTable   *this_proc_param;
  LIGOLwXMLStream       xmlfp;

  REAL8 drawnDistance = 0.0;
  REAL8 drawnRightAscension = 0.0;
  REAL8 drawnDeclination = 0.0;
  CHAR  drawnSourceName[LIGOMETA_SOURCE_MAX];

  status=blank_status;
  gpsStartTime.gpsSeconds=-1;
  gpsEndTime.gpsSeconds=-1;

  /* getopt arguments */
  /* available letters: H */
  struct option long_options[] =
  {
    {"help",                          no_argument, 0,                'h'},
    {"source-file",             required_argument, 0,                'f'},
    {"mass-file",               required_argument, 0,                'm'},
    {"nr-file",                 required_argument, 0,                'c'},
    {"exttrig-file",            required_argument, 0,                'E'},
    {"f-lower",                 required_argument, 0,                'F'},
    {"gps-start-time",          required_argument, 0,                'a'},
    {"gps-end-time",            required_argument, 0,                'b'},
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
    {"mean-mass1",              required_argument, 0,                'n'},
    {"mean-mass2",              required_argument, 0,                'N'},
    {"stdev-mass1",             required_argument, 0,                'o'},
    {"stdev-mass2",             required_argument, 0,                'O'},
    {"min-mratio",              required_argument, 0,                'x'},
    {"max-mratio",              required_argument, 0,                'y'},
    {"min-distance",            required_argument, 0,                'p'},
    {"max-distance",            required_argument, 0,                'r'},
    {"d-distr",                 required_argument, 0,                'e'},
    {"l-distr",                 required_argument, 0,                'l'},
    {"longitude",               required_argument, 0,                'v'},
    {"latitude",                required_argument, 0,                'z'},
    {"i-distr",                 required_argument, 0,                'I'},
    {"inclStd",                 required_argument, 0,                'B'},
    {"fixed-inc",               required_argument, 0,                'C'},
    {"polarization",            required_argument, 0,                'S'},
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
    {"write-compress",          no_argument,       &outCompress,       1},
    {"taper-injection",         required_argument, 0,                '*'},
    {"band-pass-injection",     no_argument,       0,                '}'},
    {0, 0, 0, 0}
  };
  int c;

  /* set up inital debugging values */
  lal_errhandler = LAL_ERR_EXIT;
  set_debug_level( "1" );

  /* create the process and process params tables */
  proctable.processTable = (ProcessTable *) 
    calloc( 1, sizeof(ProcessTable) );
  LAL_CALL( LALGPSTimeNow ( &status, &(proctable.processTable->start_time),
        &accuracy ), &status );
  if (strcmp(CVS_REVISION,"$Revi" "sion$"))
    {
      LAL_CALL( populate_process_table( &status, proctable.processTable, 
					PROGRAM_NAME, CVS_REVISION,
					CVS_SOURCE, CVS_DATE ), &status );
    }
  else
    {
      LAL_CALL( populate_process_table( &status, proctable.processTable, 
					PROGRAM_NAME, lalappsGitCommitID,
					lalappsGitGitStatus,
					lalappsGitCommitDate ), &status );
    }
  LALSnprintf( proctable.processTable->comment, LIGOMETA_COMMENT_MAX, " " );
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
        if ( gpsinput > 999999999 )
        {
          fprintf( stderr, "invalid argument to --%s:\n"
              "GPS start time is after " 
              "Sep 14, 2011  01:46:26 UTC:\n"
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
        if ( gpsinput > 999999999 )
        {
          fprintf( stderr, "invalid argument to --%s:\n"
              "GPS start time is after " 
              "Sep 14, 2011  01:46:26 UTC:\n"
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

      case 's':
        rand_seed = atoi( optarg );
        this_proc_param = this_proc_param->next = 
          next_process_param( long_options[option_index].name, "int", 
              "%d", rand_seed );
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
        LALSnprintf( waveform, LIGOMETA_WAVEFORM_MAX * sizeof(CHAR), "%s",
            optarg );
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
        LALSnprintf( this_proc_param->program, LIGOMETA_PROGRAM_MAX, "%s", 
            PROGRAM_NAME );
        LALSnprintf( this_proc_param->param, LIGOMETA_PARAM_MAX, "--userTag" );
        LALSnprintf( this_proc_param->type, LIGOMETA_TYPE_MAX, "string" );
        LALSnprintf( this_proc_param->value, LIGOMETA_VALUE_MAX, "%s",
            optarg );
        break;

      case 'd':
        optarg_len = strlen( optarg ) + 1;
        memcpy( dummy, optarg, optarg_len );
        this_proc_param = this_proc_param->next = (ProcessParamsTable *)
          calloc( 1, sizeof(ProcessParamsTable) );
        LALSnprintf( this_proc_param->program, LIGOMETA_PROGRAM_MAX, "%s", 
            PROGRAM_NAME );
        LALSnprintf( this_proc_param->param, LIGOMETA_PARAM_MAX, "--m-distr" );
        LALSnprintf( this_proc_param->type, LIGOMETA_TYPE_MAX, "string" );
        LALSnprintf( this_proc_param->value, LIGOMETA_VALUE_MAX, "%s",
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
        else
        {
          fprintf( stderr, "invalid argument to --%s:\n"
              "unknown mass distribution: %s must be one of\n"
              "(source, nrwaves, totalMass, componentMass, gaussian, log, totalMassRatio)\n", 
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
              "float", "%le", minMassRatio );
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

      case 'e':
        optarg_len = strlen( optarg ) + 1;
        memcpy( dummy, optarg, optarg_len );
        this_proc_param = this_proc_param->next = (ProcessParamsTable *)
          calloc( 1, sizeof(ProcessParamsTable) );
        LALSnprintf( this_proc_param->program, LIGOMETA_PROGRAM_MAX, "%s", 
            PROGRAM_NAME );
        LALSnprintf( this_proc_param->param,LIGOMETA_PARAM_MAX,"--d-distr" );
        LALSnprintf( this_proc_param->type, LIGOMETA_TYPE_MAX, "string" );
        LALSnprintf( this_proc_param->value, LIGOMETA_VALUE_MAX, "%s",
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
        LALSnprintf( this_proc_param->program, LIGOMETA_PROGRAM_MAX, "%s", 
            PROGRAM_NAME );
        LALSnprintf( this_proc_param->param, LIGOMETA_PARAM_MAX, "--l-distr" );
        LALSnprintf( this_proc_param->type, LIGOMETA_TYPE_MAX, "string" );
        LALSnprintf( this_proc_param->value, LIGOMETA_VALUE_MAX, "%s",
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
        else
        {
          fprintf( stderr, "invalid argument to --%s:\n"
              "unknown location distribution: "
              "%s must be one of (source, random)\n", 
              long_options[option_index].name, optarg );
          exit( 1 );
        }

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
        LALSnprintf( this_proc_param->program, LIGOMETA_PROGRAM_MAX, "%s", 
            PROGRAM_NAME );
        LALSnprintf( this_proc_param->param, LIGOMETA_PARAM_MAX, "--i-distr" );
        LALSnprintf( this_proc_param->type, LIGOMETA_TYPE_MAX, "string" );
        LALSnprintf( this_proc_param->value, LIGOMETA_VALUE_MAX, "%s",
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

      case 'S':
        /* set the polarization angle */
        psi = (REAL4) atof( optarg )/180.*LAL_PI;
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
        fprintf( stdout, "LIGO/LSC inspiral injection engine\n"
            "The CBC group \n"
            "CVS Version: " CVS_ID_STRING "\n"
            "CVS Tag: " CVS_NAME_STRING "\n" );
	fprintf( stdout, lalappsGitID );
        exit( 0 );
        break;

      case 'T':
        /* enable spining injections */
        this_proc_param = this_proc_param->next = 
          next_process_param( long_options[option_index].name, "string", 
              "" );
        spinInjections = 1;
        break;

      case 'W':
        /* disable spining injections */
        this_proc_param = this_proc_param->next = 
          next_process_param( long_options[option_index].name, "string", 
              "" );
        spinInjections = 0;
        break;

      case '}':
        /* enable band-passing */
        this_proc_param = this_proc_param->next = 
          next_process_param( long_options[option_index].name, "string", 
              "" );
        bandPassInj = 1;
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

      case 'h':
        print_usage(argv[0]);
        exit( 0 );
        break;

      case '?':
        print_usage(argv[0]);
        exit( 1 );
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
        "Must specify either a file contining the masses (--mass-file) "\
        "or choose another mass-distribution (--m-distr).\n" );
    exit( 1 );
  }
  if ( !nrFileName && mDistr==massFromNRFile )
  {
    fprintf( stderr, 
        "Must specify either a file contining the masses (--nr-file) "\
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
        "If --l-distr exttrig is specified, must specify " \
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
  if ( iDistr==gaussianInclDist && inclStd<0.0 )
  {
    fprintf( stderr, 
        "Must specify width for gaussian inclination distribution, "\
        "use --inclStd.\n" );
    exit( 1 );
  }

  /* require --f-lower be explicit */
  if ( fLower <= 0.0 )
  {
    fprintf( stderr, "--f-lower must be specified and non-zero\n" );
    exit( 1 );
  }


  /* check for gaussian mass distribution parameters */
  if ( mDistr==gaussianMassDist && (meanMass1 < 0.0 || massStdev1 < 0.0 || 
        meanMass2 < 0.0 || massStdev2 < 0.0))
  {
    fprintf( stderr, 
        "Must specify --mean-mass1/2 and --stdev-mass1/2 if choosing"
        " --m-distr=gaussian\n" );
    exit( 1 );
  }

  /* check if the mass area is properly specified */
  if ( mDistr!=gaussianMassDist && (minMass1 <0.0 || minMass2 <0.0 || 
        maxMass1 <0.0 || maxMass2 <0.0) )
  {
    fprintf( stderr, 
        "Must specify --min-mass1/2 and --max-mass1/2 if choosing"
        " --m-distr not gaussian\n" );
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
  if ( maxMtotal<0.0)
  {
    fprintf( stderr, 
        "Must specify --max-mtotal.\n" );
    exit( 1 );
  }
  if ( minMtotal<0.0)
  {
    fprintf( stderr, 
        "Must specify --min-mtotal.\n" );
    exit( 1 );
  }

  /* check if mass ratios are specified */
  if ( mDistr==uniformTotalMassRatio && (minMassRatio < 0.0 || maxMassRatio < 0.0) )
  {
    fprintf( stderr,
        "Must specify --min-mass-ratio and --max-mass-ratio if choosing"
        " --m-distr=totalMassRatio\n");
    exit( 1 );
  }

  if ( dDistr!=distFromSourceFile && (dmin<0.0 || dmax<0.0) )
  {
    fprintf( stderr, 
        "Must specify --min-distance and --max-distance if "
        "--d-distr is not source.\n" );
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
        "Must specify --disable-spin or --enable-spin\n"\
        "Unless doing NR injections\n" );
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
          "Either the options [--min-kappa1,--max-kappa1] or\n"\
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

  
  
  if ( userTag && outCompress )
  {
    LALSnprintf( fname, sizeof(fname), "HL-INJECTIONS_%d_%s-%d-%d.xml.gz",
        rand_seed, userTag, gpsStartTime.gpsSeconds, gpsDuration );
  }
  else if ( userTag && !outCompress )
  {
    LALSnprintf( fname, sizeof(fname), "HL-INJECTIONS_%d_%s-%d-%d.xml", 
        rand_seed, userTag, gpsStartTime.gpsSeconds, gpsDuration );
  }
  else if ( !userTag && outCompress )
  {
    LALSnprintf( fname, sizeof(fname), "HL-INJECTIONS_%d-%d-%d.xml.gz",
        rand_seed, gpsStartTime.gpsSeconds, gpsDuration );
  }
  else
  {
    LALSnprintf( fname, sizeof(fname), "HL-INJECTIONS_%d-%d-%ld.xml", 
        rand_seed, gpsStartTime.gpsSeconds, gpsDuration );
  }
  if ( outputFileName ) 
  {
    LALSnprintf( fname, sizeof(fname), "%s", 
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

  /* loop over parameter generation until end time is reached */
  ninj = 0;
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

    /* populate masses */
    if ( mDistr==massFromSourceFile )
    {
      drawMassFromSource( simTable );
    }
    else if ( mDistr==massFromNRFile )
    {
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
          minMtotal, maxMtotal, minMassRatio, maxMassRatio );
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
    else
    {
      simTable=XLALRandomInspiralDistance(simTable, randParams, 
          dDistr, dmin/1000.0, dmax/1000.0);
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
    else
    {
      simTable=XLALRandomInspiralSkyLocation(simTable, randParams); 
    }

    /* populate orientations */
    if ( iDistr == fixedInclDist )
    {
      simTable->inclination = fixed_inc;
    }
    else
    {                           
      do {
	simTable=XLALRandomInspiralOrientation(simTable, randParams,
					       iDistr, inclStd);
      } while ( ! strcmp(waveform, "SpinTaylorthreePointFivePN") &&
		( simTable->inclination < eps ||
		  simTable->inclination > LAL_PI-eps) );
    }

    /* set polarization angle */
    simTable->polarization = psi;

    /* populate spins, if required */
    if (spinInjections)
    {
      simTable = XLALRandomInspiralSpins( simTable, randParams, 
          minSpin1, maxSpin1,
          minSpin2, maxSpin2,
          minKappa1, maxKappa1,
          minabsKappa1, maxabsKappa1);
    }

    /* populate the site specific information */
    LALPopulateSimInspiralSiteInfo( &status, simTable );

    /* populate the taper options */
    {
        switch (taperInj)
        {
            case INSPIRAL_TAPER_NONE:
                 LALSnprintf( simTable->taper, LIGOMETA_WAVEFORM_MAX, 
                         "%s", "TAPER_NONE"); 
                 break;
            case INSPIRAL_TAPER_START:
                 LALSnprintf( simTable->taper, LIGOMETA_WAVEFORM_MAX, 
                         "%s", "TAPER_START"); 
                 break;
            case INSPIRAL_TAPER_END:
                 LALSnprintf( simTable->taper, LIGOMETA_WAVEFORM_MAX, 
                         "%s", "TAPER_END"); 
                 break;
            case INSPIRAL_TAPER_STARTEND:
                 LALSnprintf( simTable->taper, LIGOMETA_WAVEFORM_MAX, 
                         "%s", "TAPER_STARTEND"); 
                 break;
            default: /* Never reach here */
                 fprintf( stderr, "unknown error while populating sim_inspiral taper options\n" );
                 exit (1);
        }

    }
    
    /* populate the bandpass options */
    simTable->bandpass = bandPassInj;

    /* increment current time, avoiding roundoff error;
       check if end of loop is reached */
    currentGpsTime = gpsStartTime;
    XLALGPSAdd(&currentGpsTime, ninj * meanTimeStep);
    if ( XLALGPSCmp( &currentGpsTime, &gpsEndTime ) >= 0 )
      break;

    /* allocate and go to next SimInspiralTable */
    simTable = simTable->next = (SimInspiralTable *)
      calloc( 1, sizeof(SimInspiralTable) );
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
  LAL_CALL( LALGPSTimeNow ( &status, &(proctable.processTable->end_time),
        &accuracy ), &status );
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

  LAL_CALL( LALCloseLIGOLwXMLFile ( &status, &xmlfp ), &status );

  LALCheckMemoryLeaks();
  return 0;
}

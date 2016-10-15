/*
*  Copyright (C) 2007 Badri Krishnan
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
 * \file
 * \ingroup lalapps_pulsar_Hough
 * \author Badri Krishnan
 * \brief
 * program for comparing two sfts in a frequency band
 */

#include <lal/SFTfileIO.h>


/* Default parameters. */



#define MAXFILENAMELENGTH 512
/* defaults chosen for L1 */

#define FILE1 "/nfs/morbo/geo600/hannover/sft/S2-LIGO/S2_L1_Funky-v3Calv5DQ30MinSFTs/CAL_SFT.734359206"
#define FILE2 "/nfs/morbo/geo600/hannover/sft/S2-LIGO-clean/S2_L1_Funky-v3Calv5DQ30MinSFTs-clean/CLEAN_SFT.734359206"
#define STARTFREQ 150.0
#define BANDFREQ 1.0



/* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< */
/* vvvvvvvvvvvvvvvvvvvvvvvvvvvvvv------------------------------------ */
int main(int argc, char *argv[]){ 
  static  LALStatus   status;  /* LALStatus pointer */ 
  SFTtype     *sft1, *sft2;
  REAL8       startFreq, bandFreq; 
  /*  CHAR file1[MAXFILENAMELENGTH], file2[MAXFILENAMELENGTH]; */
  CHAR *file1, *file2;
  INT4 arg;

  /* set defaults */
  file1 = FILE1;
  file2 = FILE2;
  startFreq = STARTFREQ;
  bandFreq = BANDFREQ;

  /********************************************************/  
  /* Parse argument list.  i stores the current position. */
  /********************************************************/
  arg = 1;
  while ( arg < argc ) {
    /* Parse debuglevel option. */
    if ( !strcmp( argv[arg], "-d" ) ) {
      if ( argc > arg + 1 ) {
        arg++;
      }
    }  
    /* parse first sft filename */
    else if ( !strcmp( argv[arg], "-A" ) ) {
      if ( argc > arg + 1 ) {
        arg++;
        file1 = argv[arg++];
      } 
    }  
    /* parse second sft filename */
    else if ( !strcmp( argv[arg], "-B" ) ) {
      if ( argc > arg + 1 ) {
        arg++;
        file2 = argv[arg++];
      }
    }  
    /* parse start freq */
    else if ( !strcmp( argv[arg], "-f" ) ) {
      if ( argc > arg + 1 ) {
        arg++;
        startFreq = atof(argv[arg++]);
      }
    }  
    /* parse bandwidth  */
    else if ( !strcmp( argv[arg], "-b" ) ) {
      if ( argc > arg + 1 ) {
        arg++;
        bandFreq = atof(argv[arg++]);
      }
    }  
    /* Unrecognized option. */
    else {
      printf("unknown argument\n");
      printf("options are: \n");
      printf("-d LALdebuglevel -A firstsftfile -B secondsftfile -f startfrequency -b freqband\n");
      exit(0);
    }
  } 

  /* End of argument parsing loop. */
  /******************************************************************/   


  
  sft1 = NULL;
  sft2 = NULL;
  LALReadSFTfile (&status, &sft1, startFreq, startFreq + bandFreq, file1);
  REPORTSTATUS( &status);

  LALReadSFTfile (&status, &sft2, startFreq, startFreq + bandFreq, file2);
  /*REPORTSTATUS( &status);*/
  {
    UINT4 j, nBins;
    REAL8 diff;
    COMPLEX8 *data1, *data2;
    nBins = sft1->data->length;

    for (j=0; j<nBins; j++)
      {
	data1 = sft1->data->data + j;
	data2 = sft2->data->data + j;
	diff = (data1->re - data2->re)*(data1->re - data2->re) + (data1->im - data2->im)*(data1->im - data2->im);
	printf("%1.3e\n", sqrt(diff));
      }
  }

  XLALDestroySFT ( sft1);
  /*REPORTSTATUS( &status);*/

  XLALDestroySFT ( sft2);
  /*REPORTSTATUS( &status);*/

  LALCheckMemoryLeaks(); 
  /*REPORTSTATUS( &status);*/

  return status.statusCode;
}

/* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< */














/*
*  Copyright (C) 2007  Holger Pletsch, Reinhard Prix, Xavier Siemens
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
 * \ingroup lalapps_pulsar
 * \author X. Siemens
 * \brief Noise Estimator
 */

/*********************************************************************************/
/*                                   Noise Estimator                             */
/*                                                                               */
/*			               X. Siemens                                */
/*                                                                               */
/*                                UWM - June 2004                                */
/*********************************************************************************/

#include "FindSh.h"
#include "clusters.h"
UINT4 SFTno,RealSFTno;                /* Global variables that keep track of no of SFTs */
INT4 ifmin,band;
REAL4 *p;
REAL8 *po;
char filelist[MAXFILES][MAXFILENAMELENGTH];
REAL8 N,deltaT;
REAL8 medianbias=1.0;
UINT4 numbins=50;
static LALStatus status;

int main(int argc,char *argv[]) 
{


  LALRngMedBias (&status, &medianbias, numbins);

  /* Reads command line arguments into the CommandLineArgs struct. 
     In the absence of command line arguments it sets some defaults */
  if (ReadCommandLine(argc,argv,&CommandLineArgs)) return 1;

  /* Reads in SFT directory for filenames and total number of files */
  if (ReadSFTDirectory(CommandLineArgs)) return 2;

  /* Find the PSD */
  if (ComputePSD()) return 2;

  /* Free memory*/
  if (Freemem()) return 8;

  return 0;

}

/*******************************************************************************/

int ComputePSD(void)
{
  FILE *fp;
  INT4 offset;
  UINT4 ndeltaf;
  UINT4 i, j;
  size_t errorcode;
  REAL8 ShAve=0.0,SpAve=0.0,ShInv;
  REAL8Vector *Sp=NULL, *RngMdnSp=NULL;   /* |SFT|^2 and its rngmdn  */
  ndeltaf = band+1;

  LALDCreateVector(&status, &Sp, (UINT4) ndeltaf);
  LALDCreateVector(&status, &RngMdnSp, (UINT4) ndeltaf);

  ShInv=0.0;

  for (i=0;i<SFTno;i++)       /* Loop over SFTs */          
    {

      /* Set to zero the values */
      for (j=0;j<ndeltaf;j++){
	RngMdnSp->data[j] = 0.0;
	Sp->data[j]       = 0.0;
      }

     fp=fopen(filelist[i],"r");
      if (fp==NULL) 
	{
	  fprintf(stderr,"Weird... %s doesn't exist!\n",filelist[i]);
	  return 1;
	}

      /* Read in the header from the file */
      errorcode=fread((void*)&header,sizeof(header),1,fp);
      if (errorcode!=1) 
	{
	  fprintf(stderr,"No header in data file %s\n",filelist[i]);
	  return 1;
	}
      
      /* Check that data is correct endian order */
      if (header.endian!=1.0)
	{
	  fprintf(stderr,"First object in file %s is not (double)1.0!\n",filelist[i]);
	  fprintf(stderr,"It could be a file format error (big/little\n");
	  fprintf(stderr,"endian) or the file might be corrupted\n\n");
	  return 2;
	}
    
      /* Check that the time base is positive */
      if (header.tbase<=0.0)
	{
	  fprintf(stderr,"Timebase %f from data file %s non-positive!\n",
		  header.tbase,filelist[i]);
	  return 3;
	}
      
      offset=(ifmin-header.firstfreqindex)*2*sizeof(REAL4);
      errorcode=fseek(fp,offset,SEEK_CUR);


      errorcode=fread((void*)p,2*ndeltaf*sizeof(REAL4),1,fp);  
      if (errorcode!=1)
	{
	  printf("Dang! Error reading data in SFT file %s!\n",filelist[i]);
	  return 1;
	}
      fclose(fp);

      /*Loop over frequency bins in each SFT      */
      for (j=0;j<ndeltaf;j++)
	 {
	   int jre=2*j;
	   int jim=jre+1;

	   Sp->data[j]=(REAL8)(p[jre]*p[jre]+p[jim]*p[jim]);
	 }

      SpAve=0.0;
      for (j=0;j<ndeltaf;j++)
	 {
	   SpAve=SpAve+Sp->data[j];/*/ndeltaf;*/
	 }
      SpAve=SpAve/ndeltaf;

      /* Compute running median */
      EstimateFloor(&status, Sp, numbins, RngMdnSp);

      /* Average */
      ShAve=0.0;
      for (j=0;j<ndeltaf;j++){
	ShAve=ShAve+RngMdnSp->data[j]/ndeltaf/medianbias;
      }

      ShInv=ShInv+1.0/(2*deltaT*ShAve/N)/SFTno;

    }

  fprintf(stdout,"%15.14e\n",sqrt(1/ShInv));

  LALDDestroyVector(&status, &RngMdnSp);
  LALDDestroyVector(&status, &Sp);
  
  return 0;

}

/*******************************************************************************/

int ReadSFTDirectory(struct CommandLineArgsTag CLA)
{
  char command[256];
  size_t errorcode;
  FILE *fp;
  UINT4 filenum=0;
  UINT4 j;
  glob_t globbuf;


  /* check out what's in SFT directory */
  strcpy(command,CLA.directory);
  strcat(command,"/*SFT*");
  globbuf.gl_offs = 1;
  glob(command, GLOB_ERR|GLOB_MARK, NULL, &globbuf);

  /* read file names -- MUST NOT FORGET TO PUT ERROR CHECKING IN HERE !!!! */
  while (filenum < globbuf.gl_pathc) 
    {
      strcpy(filelist[filenum],globbuf.gl_pathv[filenum]);
      filenum++;
      if (filenum > MAXFILES)
	{
	  fprintf(stderr,"Too many files in directory! Exiting... \n");
	  return 1;
	}
    }
  globfree(&globbuf);

  SFTno=filenum;  /* Global variable that keeps track of no of SFTs*/


  /* open FIRST file and get info from it*/

  fp=fopen(filelist[0],"r");
  if (fp==NULL) 
    {
      fprintf(stderr,"Weird... %s doesn't exist!\n",filelist[0]);
      return 1;
    }
  /* Read in the header from the file */
  errorcode=fread((void*)&header,sizeof(header),1,fp);
  if (errorcode!=1) 
    {
      fprintf(stderr,"No header in data file %s\n",filelist[0]);
      return 1;
    }

  /* Check that data is correct endian order */
  if (header.endian!=1.0)
    {
      fprintf(stderr,"First object in file %s is not (double)1.0!\n",filelist[0]);
      fprintf(stderr,"It could be a file format error (big/little\n");
      fprintf(stderr,"endian) or the file might be corrupted\n\n");
      return 2;
    }
    
  /* Check that the time base is positive */
  if (header.tbase<=0.0)
    {
      fprintf(stderr,"Timebase %f from data file %s non-positive!\n",
	      header.tbase,filelist[0]);
      return 3;
    }
  fclose(fp);

  /* Allocate pointers for SFT data -- to be used later */
  p=(REAL4 *)LALMalloc(2*header.nsamples*sizeof(REAL4));
  po=(REAL8 *)LALMalloc(SFTno*sizeof(REAL8));

  for (j=0;j<SFTno;j++)
    {
      po[j]=0.0;   
    }
 
 deltaT=1.0/CLA.s;  /* deltaT is the time resolution of the original data */
 N=header.tbase*CLA.s; /* the number of time data points in one sft  */

 if(CLA.s == 0.0)
   {
     deltaT=header.tbase/(2.0*header.nsamples); /* deltaT is the time resolution of the original data */
     N=2.0*header.nsamples; /* the number of time data points in one sft */
   }

  ifmin=floor(CLA.f0*header.tbase);
  band=ceil(CLA.b*header.tbase);

      if (ifmin<header.firstfreqindex || 
	  ifmin+band > header.firstfreqindex+header.nsamples) 
	{
	  fprintf(stderr,"Freq index range %d->%d not in %d to %d\n",
		ifmin,ifmin+band,header.firstfreqindex,
		  header.firstfreqindex+header.nsamples);
	  return 4;
	}

  return 0;
}


/*******************************************************************************/

int ReadCommandLine(int argc,char *argv[],struct CommandLineArgsTag *CLA) 
{
  INT4 c, errflg = 0;
  optarg = NULL;
  
  /* Initialize default values */
  CLA->directory=NULL;
  CLA->f0=0.0;
  CLA->b=0.0;
  CLA->s=0.0;
  CLA->nbins=50;
 
  /* Scan through list of command line arguments */
  while (!errflg && ((c = getopt(argc, argv,"hb:D:r:I:C:o:f:b:s:n:"))!=-1))
    switch (c) {
    case 'D':
      /* SFT directory */
      CLA->directory=optarg;
      break;
    case 'f':
      /* calibrated sft output directory */
      CLA->f0=atof(optarg);
      break;
    case 'b':
      /* calibrated sft output directory */
      CLA->b=atof(optarg);
      break;
    case 's':
      /* sampling rate */
      CLA->s=atof(optarg);
      break;
    case 'n':
      /* number of bins for computing the running median */
      CLA->nbins=atof(optarg);
      break;
    case 'h':
      /* print usage/help message */
      fprintf(stderr,"Arguments are:\n");
      fprintf(stderr,"\t-D\tSTRING\t(Directory where SFTs are located)\n");
      fprintf(stderr,"\t-f\tFLOAT\t(Starting Frequency in Hz; default is 0.0 Hz)\n");
      fprintf(stderr,"\t-b\tFLOAT\t(Frequenzy band in Hz)\n");
      fprintf(stderr,"\t-s\tFLOAT\t(Sampling Rate in Hz; if = 0 then header info is used)\n");
      fprintf(stderr,"\t-n\tINT\t(Number of bins to compute running median; default is 50)\n");
      fprintf(stderr,"(e.g.: ./FindSh -D ../KnownPulsarDemo/data/ -f 1010.7 -b 3.0 -n 50) \n");
      exit(0);
      break;
    default:
      /* unrecognized option */
      errflg++;
      fprintf(stderr,"Unrecognized option argument %c\n",c);
      exit(1);
      break;
    }
  if(CLA->b == 0.0)
    {
      fprintf(stderr,"No frequency band specified; input band with -b option.\n");
      fprintf(stderr,"For help type ./FindSh -h \n");
      return 1;
    }  
  if(CLA->nbins <= 0)
    {
      fprintf(stderr,"Number of frequency bins must greater than zero; input number with -n option.\n");
      fprintf(stderr,"For help type ./FindSh -h \n");
      return 1;
    }      
  else
    {
      numbins=CLA->nbins;
    }

  if(CLA->directory == NULL)
    {
      fprintf(stderr,"No directory specified; input directory with -D option.\n");
      fprintf(stderr,"For help type ./FindSh -h \n");
      return 1;
    }      
  return errflg;
}


/*******************************************************************************/


int Freemem()
{

  LALFree(po);
  LALFree(p);

  LALCheckMemoryLeaks();
  
  return 0;
}

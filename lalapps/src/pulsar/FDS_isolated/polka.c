/*
*  Copyright (C) 2007 Bruce Allen, Bernd Machenschalk, Reinhard Prix, Xavier Siemens
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
 * \ingroup pulsarApps
 * \author X. Siemens
 * \brief takes in two Fstats file to look for coincidence
 */

/*********************************************************************************/
/*                     polka - the pulsar koinzidenz analysis code               */
/*                                                                               */
/*			               X. Siemens                                */
/*                   (takes in two Fstats file to look for coincidence)          */
/*                                                                               */
/*                                  UWM - March  2004                            */
/*********************************************************************************/


#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include <lal/LALDatatypes.h>
#include <lal/LALMalloc.h>
#include <lal/LALConstants.h>
#include <lal/LALStatusMacros.h>
#include <lal/ConfigFile.h>

#include <lalapps.h>

#include "getopt.h"

/* some error codes and messages */
#define POLKAC_ENULL  		1
#define POLKAC_ENONULL 		2
#define POLKAC_ESYS 		3
#define POLKAC_EINVALIDFSTATS	4
#define POLKAC_EMEM		5

#define POLKAC_MSGENULL 	"Arguments contained an unexpected null pointer"
#define POLKAC_MSGENONULL	"Input pointer was not NULL"
#define POLKAC_MSGESYS		"System call failed (probably file IO"
#define POLKAC_MSGEINVALIDFSTATS "Invalid Fstats file"
#define POLKAC_MSGEMEM		"Sorry, ran out of memory... bye."

#ifndef USE_BOINC
#define USE_BOINC 0
#endif

#if USE_BOINC
#include "boinc_api.h"
#include "filesys.h"

#define fopen boinc_fopen
extern CHAR *Outputfilename;
#endif

struct PolkaCommandLineArgsTag 
{
  char *FstatsFile1; /* Names of Fstat files to be read in */
  char *FstatsFile2;
  char *FstatsFile3; /* Names of Fstat files to be read in to compute false alarm */
  char *FstatsFile4;
  char *OutputFile;
  REAL8 Deltaf;      /* Size of coincidence window in Hz */
  REAL8 DeltaAlpha;  /* Size of coincidence window in radians */
  REAL8 DeltaDelta;  /* Size of coincidence window in radians */
  REAL8 fmin;        /* Minimum frequency of candidate in first IFO */
  REAL8 fmax;        /* Maximum frequency of candidate in first IFO */
  UINT4 EAH;         /* Einstein at home flag for alternative output */ 
} PolkaCommandLineArgs;

typedef struct CandidateTag 
{
  UINT4 length;	   /* number of candidates in list */
  REAL8 *f;        /* Frequency */
  REAL8 *Alpha;    /* longitude */
  REAL8 *Delta;    /* latitude */
  REAL8 *F;        /* Maximum value of F for the cluster */
  REAL8 *fa;       /* false alarm probability for that candidate */
  UINT4 *Ctag;     /* tag for candidate if it's been found in coincidence */
  INT4  *CtagCounter;     /* contains the cumulative sum of coincident candidates so far */
} CandidateList;

typedef struct CoincidentCandidateTag 
{
  REAL8 f1;		/* Frequencies */
  REAL8 f2;
  REAL8 Alpha1;		/* longitude */
  REAL8 Alpha2;
  REAL8 Delta1;		/* latitude */
  REAL8 Delta2;
  REAL8 F1;		/* Maximum value of F for the cluster */
  REAL8 F2;
  REAL8 fa;		/* false alarm probabilities for that candidate */
  REAL8 fa1;
  REAL8 fa2;       	
} CoincidentCandidate;

typedef struct CoincidentPairsTag 
{
  UINT4 c1;             /* number in Fstats file that corresponds to first member of pair */
  UINT4 c2;             /* number in Fstats file that corresponds to second member of pair */
  REAL8 fa;             /* joint false alarm for that pair */
} CoincidentPairs;


int ReadCommandLine(int argc,char *argv[],struct PolkaCommandLineArgsTag *CLA);
int ReadCandidateFiles(struct PolkaCommandLineArgsTag CLA);
int compare1F(const void *ip, const void *jp);
int compare2F(const void *ip, const void *jp);
int compare3F(const void *ip, const void *jp);
int compare4F(const void *ip, const void *jp);
int compare2f(const void *ip, const void *jp);
int compareCCfa(const void *ip, const void *jp);
int compareCPfa(const void *ip, const void *jp);
void locate(double xx[], int n, double x, int *j, int *indices);
void ReadOneCandidateFile (LALStatus *stat, CandidateList *CList, const char *fname);

extern INT4 lalDebugLevel;

CandidateList CList1, CList2, CList3, CList4; /* treat up to 4 candidate files */

CoincidentCandidate *CC;
CoincidentPairs *CP;

#ifndef FALSE
#define FALSE (1==0)
#endif
#ifndef TRUE
#define TRUE  (1==1)
#endif

/* main() mapped to polka() if using boinc */
#if USE_BOINC
int polka(int argc,char *argv[])
#else
int main(int argc,char *argv[]) 
#endif
{
  INT4 *indices1F=NULL,*indices2f=NULL,*indices2F=NULL,*indicesCCfa=NULL,*indices3F=NULL,*indices4F=NULL;
  REAL8 MaxAngularDistance;
  UINT4 i;
  UINT4 numCoincidences = 0;
  FILE *fpOut;
  BOOLEAN haveFile3 = FALSE;
  BOOLEAN haveFile4 = FALSE;
#if USE_BOINC
  static char resolved_filename[256];
#endif
  lalDebugLevel = 1;

  /* Reads command line arguments */
  if (ReadCommandLine(argc,argv,&PolkaCommandLineArgs)) return 1;

  /* define some shortcuts for convenience */
  haveFile3 = (PolkaCommandLineArgs.FstatsFile3 != NULL);
  haveFile4 = (PolkaCommandLineArgs.FstatsFile4 != NULL);

  /* Reads in candidare files */
  if (ReadCandidateFiles(PolkaCommandLineArgs)) return 2;

  /* Prepare list of coincident candidates */

  /* create arrays of indices */
  if (!(indices1F=(INT4 *)LALMalloc(sizeof(INT4) * CList1.length))){
    fprintf(stderr,"Unable to allocate index array in main\n");
    return 1;
  }
  if (!(indices2F=(INT4 *)LALMalloc(sizeof(INT4) * CList2.length))){
    fprintf(stderr,"Unable to allocate index array in main\n");
    return 1;
  }
  if (!(indices2f=(INT4 *)LALMalloc(sizeof(INT4) * CList2.length))){
    fprintf(stderr,"Unable to allocate index array in main\n");
    return 1;
  }

  /* populate arrays of indices */
  for (i=0;i<CList1.length;i++) indices1F[i]=i;
  for (i=0;i<CList2.length;i++) indices2F[i]=i;
  for (i=0;i<CList2.length;i++) indices2f[i]=i;

  /* sort arrays of indices in DECREASING order*/
  qsort((void *)indices1F, (size_t)CList1.length, sizeof(int), compare1F);
  qsort((void *)indices2F, (size_t)CList2.length, sizeof(int), compare2F);
  qsort((void *)indices2f, (size_t)CList2.length, sizeof(int), compare2f);

  if ( haveFile3 )
    {
      if (!(indices3F=(INT4 *)LALMalloc(sizeof(INT4)*CList3.length))){
	fprintf(stderr,"Unable to allocate index array in main\n");
	return 1;
      }
      for (i=0;i<CList3.length;i++) indices3F[i]=i;
      qsort((void *)indices3F, (size_t)CList3.length, sizeof(int), compare3F);
    }      
  if ( haveFile4 )
    {
      if (!(indices4F=(INT4 *)LALMalloc(sizeof(INT4)*CList4.length))){
	fprintf(stderr,"Unable to allocate index array in main\n");
	return 1;
      }
      for (i=0;i<CList4.length;i++) indices4F[i]=i;
      qsort((void *)indices4F, (size_t)CList4.length, sizeof(int), compare4F);
    }      

  numCoincidences = 0; /* kounts koinzident events */
  MaxAngularDistance=sqrt(pow(PolkaCommandLineArgs.DeltaAlpha,2)+pow(PolkaCommandLineArgs.DeltaDelta,2))+1e-8;

  /* go through list */

  if (CList1.length !=0 && CList2.length !=0 )
    {
      for (i=0; i < CList1.length; i++)
	{
	  REAL8 f1min,f1max,difff;
	  REAL8 f1,Alpha1,Delta1,F1;
	  int if2min,if2max,f;

	  /* Minimum and maximum frequencies acceptable for coincidence */
	  f1 = CList1.f[indices1F[i]];
	  /* if candidate frequency does not lie within bounds specified by user go to next in list */
	  if(f1 < PolkaCommandLineArgs.fmin || f1 > PolkaCommandLineArgs.fmax) continue;

	  f1min=f1-PolkaCommandLineArgs.Deltaf;
	  f1max=f1+PolkaCommandLineArgs.Deltaf;
	  
	  /* Find nearest index to f1min and f1max; function explained below */
	  locate(CList2.f,CList2.length,f1min,&if2min,indices2f);
	  locate(CList2.f,CList2.length,f1max,&if2max,indices2f);

	  /* look for repeats in the frequency and make sure we include them in our search for coincidences */
	  /* This potential problem was pointed out by Peter Shawhan during the code review */ 
	  /* look only until second to last */
	  while ( (if2min < (int)CList2.length-2) && (CList2.f[indices2f[if2min]] == CList2.f[indices2f[if2min+1]]) )
	    if2min++;
	  /* then check last one separately to avoid seg faults */
	  if ( (if2min == (int)CList2.length-2) && (CList2.f[indices2f[if2min]] == CList2.f[indices2f[if2min+1]]))
	    if2min++;
	  /* look only until second  */
	  while ( (if2max > 1) && (CList2.f[indices2f[if2max]] == CList2.f[indices2f[if2max-1]]) )
	    if2max--;
	  /* then check first one separately to avoid seg faults */
	  if ( (if2max == 1) && (CList2.f[indices2f[if2max]] == CList2.f[indices2f[if2max-1]]))
	    if2max--;

	  /* alpha */
	  Alpha1=CList1.Alpha[indices1F[i]];
	  /* delta */
	  Delta1=CList1.Delta[indices1F[i]];
	  /* F */
	  F1=CList1.F[indices1F[i]];
      
	  for (f=if2max; f <= if2min; f++)
	    {
	      REAL8 Alpha2=CList2.Alpha[indices2f[f]],Delta2=CList2.Delta[indices2f[f]];
	      REAL8 n1[3],n2[3],AngularDistance;
	      REAL8 cosAngularDistance;
	  
	      n1[0]=cos(Alpha1)*cos(Delta1);
	      n1[1]=sin(Alpha1)*cos(Delta1);
	      n1[2]=sin(Delta1);
	      
	      n2[0]=cos(Alpha2)*cos(Delta2);
	      n2[1]=sin(Alpha2)*cos(Delta2);
	      n2[2]=sin(Delta2);
	      
	      cosAngularDistance=n1[0]*n2[0]+n1[1]*n2[1]+n1[2]*n2[2];
	      if (cosAngularDistance  >  1.0) cosAngularDistance =  1.0;
	      if (cosAngularDistance  < -1.0) cosAngularDistance = -1.0;
	      
	      AngularDistance=acos(cosAngularDistance);

	      difff=fabs(f1 - CList2.f[indices2f[f]]);
	      
	      /* check difference in frequencies because we're not guaranteed 
		 sufficient closeness at the edges of array */
	      if ( difff <= PolkaCommandLineArgs.Deltaf) 
		{
		  if ( AngularDistance <= MaxAngularDistance )
		    {	
		      int j;
		      CoincidentCandidate *thisCC;
		      CoincidentPairs *thisCP;

		      /* tag the candidates that have been found in coincidence */
		      CList1.Ctag[indices1F[i]]=1;
		      CList2.Ctag[indices2f[f]]=1;
		  
		      /* seems we found a coincident candidate: let's make space for it to be stored */
		      numCoincidences ++;
		      if ( (CC = LALRealloc ( CC, numCoincidences * sizeof(CoincidentCandidate) )) == NULL) {
			fprintf (stderr, "Error: ran out of memory ... goodbye.\n");
			return 1;	/* got lazy here.. */
		      }
		      if ( (CP = LALRealloc ( CP, numCoincidences * sizeof(CoincidentPairs) )) == NULL) {
			fprintf (stderr, "Error: ran out of memory ... goodbye.\n");
			return 1;	/* got lazy here.. */
		      }
		      
		      thisCC = &(CC[ numCoincidences - 1]);	/* point to current new coincidences */
		      thisCP = &(CP[ numCoincidences - 1]);	/* point to current new coincidences */
		      
		      thisCC->f1 = f1;
		      thisCC->Alpha1 = Alpha1;
		      thisCC->Delta1 =Delta1;
		      thisCC->F1 = F1;

		      if ( haveFile3 )
			{
			  locate(CList3.F,CList3.length,thisCC->F1,&j,indices3F);
			  thisCC->fa1=(REAL8)(j+1)/(REAL8)CList3.length;
			}
		      else thisCC->fa1=(REAL8)(i+1)/(REAL8)CList1.length;

		      thisCC->f2=CList2.f[indices2f[f]];
		      thisCC->Alpha2=CList2.Alpha[indices2f[f]];
		      thisCC->Delta2=CList2.Delta[indices2f[f]];
		      thisCC->F2=CList2.F[indices2f[f]];

		      if ( haveFile4 )
			{
			  locate(CList4.F,CList4.length,thisCC->F2,&j,indices4F);
			  thisCC->fa2=(REAL8)(j+1)/(REAL8)CList4.length;
			}else{
			  locate(CList2.F,CList2.length,thisCC->F2,&j,indices2F);
			  thisCC->fa2=(REAL8)(j+1)/(REAL8)CList2.length;
			}
		      
		      thisCC->fa = thisCC->fa1 * thisCC->fa2;
		      
		      thisCP->c1=indices1F[i];
		      thisCP->c2=indices2f[f];
		      thisCP->fa=thisCC->fa;

		    }
		}
	    }
      /* next candidate for 1st ifo */
	}     
    }


  /* allocate space */
  if (numCoincidences != 0){ 
    if (!(indicesCCfa=(INT4 *)LALMalloc(sizeof(INT4) * numCoincidences))){
      fprintf(stderr,"Unable to allocate index array in main\n");
      return 1;
    }
  }

  for (i=0; i < numCoincidences; i++) 
    indicesCCfa[i]=i;

  /* open and write the file */
#if USE_BOINC
  if (boinc_resolve_filename(PolkaCommandLineArgs.OutputFile, resolved_filename, sizeof(resolved_filename))) {
    fprintf(stderr,
	    "Can't resolve file \"%s\"\n"
	    "If running a non-BOINC test, create [INPUT] or touch [OUTPUT] file\n",
	    PolkaCommandLineArgs.OutputFile);
    boinc_finish(2);
  }
  fpOut=fopen(resolved_filename,"w");
#else
  fpOut=fopen(PolkaCommandLineArgs.OutputFile,"w"); 	 
#endif
  if (!PolkaCommandLineArgs.EAH)
    {
      /* sort in increasing probability of joint false alarm */
      qsort((void *)indicesCCfa, (size_t)numCoincidences, sizeof(int), compareCCfa);
      for (i=0; i < numCoincidences; i++) 
	{
	  UINT4 k = indicesCCfa[i];  /* print out ordered by joint significance */
	  fprintf(fpOut,"%1.15e %e %e %e %e %1.15e %e %e %e %e %e\n",
		  CC[k].f1, CC[k].Alpha1, CC[k].Delta1,
		  CC[k].F1, CC[k].fa1,
		  CC[k].f2, CC[k].Alpha2, CC[k].Delta2,
		  CC[k].F2, CC[k].fa2, CC[k].fa);
	}
    }else{
      fprintf(fpOut,"%%1\n");
      {
	int k=-1;    
	for (i=0; i < CList1.length; i++)
	  {
	    if (CList1.Ctag[i]) 
	      {
		k++;
		fprintf(fpOut,"%16.12f %10.8f %10.8f %20.17f\n",
			CList1.f[i],CList1.Alpha[i],CList1.Delta[i],CList1.F[i]);
		CList1.CtagCounter[i]=k;
	      }
	  }
      }
      fprintf(fpOut,"%%2\n");
      {
	int k=-1;
	for (i=0; i < CList2.length; i++)
	  {
	    if (CList2.Ctag[i]) 
	      {
		k++;
		fprintf(fpOut,"%16.12f %10.8f %10.8f %20.17f\n",
			CList2.f[i],CList2.Alpha[i],CList2.Delta[i],CList2.F[i]);  
		CList2.CtagCounter[i]=k;
	      }    
	  }
      }
      fprintf(fpOut,"%%coincidences\n");
      /* sort in increasing probability of joint false alarm */
      qsort((void *)indicesCCfa, (size_t)numCoincidences, sizeof(int), compareCPfa);
      
      for (i=0; i < numCoincidences; i++) 
	{
	  UINT4 k = indicesCCfa[i];  /* print out ordered by joint significance */
	  fprintf(fpOut,"%d %d %e\n",
		  CList1.CtagCounter[CP[k].c1],CList2.CtagCounter[CP[k].c2],CP[k].fa);
	}

/*       for (i=0; i < numCoincidences; i++)  */
/* 	{ */
/* 	  UINT4 k = indicesCCfa[i];   */
/* 	  fprintf(stdout,"%1.15le %le %le %le %le %1.15le %le %le %le %le %le\n",  */
/* 		  CList1.f[CP[k].c1],CList1.Alpha[CP[k].c1],CList1.Delta[CP[k].c1],CList1.F[CP[k].c1],CList1.fa[CP[k].c1], */
/* 		  CList2.f[CP[k].c2],CList2.Alpha[CP[k].c2],CList2.Delta[CP[k].c2],CList2.F[CP[k].c2],CList2.fa[CP[k].c2],CP[k].fa); */
/* 	} */
      /* cat polka_out-short | awk '{print $1" "$2" "$3" "$4" "$6" "$7" "$8" "$9" "$11}' > la3 */

    }
  fprintf(fpOut,"%%DONE\n");	
#if USE_BOINC
  /* write end marker */
  Outputfilename=resolved_filename;
#endif
  fclose(fpOut);

  LALFree(indices1F);
  LALFree(indices2F);
  LALFree(indices2f);

  /* freeing a CList is a bit tedious, so we use a macro */
#define freeCList(x) do { LALFree((x).f); LALFree((x).Alpha); LALFree((x).Delta); LALFree((x).F); LALFree((x).fa); LALFree((x).Ctag);LALFree((x).CtagCounter);} while(0)
  
  freeCList(CList1);
  freeCList(CList2);
  if (haveFile3 ) {
    freeCList(CList3);
    LALFree(indices3F);  
  }
  if (haveFile4 ) 
    {
      freeCList(CList4);
      LALFree(indices4F);  
    }
  
  if (numCoincidences != 0){ 
    LALFree ( CC );
    LALFree ( CP );
    LALFree(indicesCCfa);

  }
  
  LALCheckMemoryLeaks(); 

  return 0;

}

/*******************************************************************************/
/* Explanation of locate

the function locate returns the lower value of the two indices of the array xx
that bound the value given x.

consider xx to be in descending order:

fmax                                      fmin
   |     |     |     |     |     |     |     |
j: 0     1     2     3     4     5     6     7

            ^           ^            ^
	 f2+Df	        f2        f2-Df

locate will return if2max=1 and if2min=5

In this example:
 
1) at the beginning jl=0 and ju=8

2) since ju-jl > 1:
   jm=(0+8)/2=4
   since f2 > xx[4]: then ju=4

3) since ju-jl > 1:
   jm=(0+4)/2=2
   since f2 < xx[2]: then jl=2

4) since ju-jl > 1:
   jm=(2+4)/2=3
   since f2 < xx[3]: then jl=3

5) since ju-jl=1   
   return j=jl=3

*/

void locate(double xx[], int n, double x, int *j, int *indices) 
     /* locates x in array of xx */
{ 

 int ju,jm,jl; 


 if( x <= xx[indices[n-1]] ) 
   {
     *j=n-1;
     return;
   }
 if( x >= xx[indices[0]] ) 
   {
     *j=0;
     return;
   }

 jl=0;  
 ju=n;

 while (ju-jl > 1) 
   {
     jm=(ju+jl)/2;
     if ( x <= xx[indices[jm]] ) 
       jl=jm;
     else ju=jm; 
   }
 
 *j=jl;

}


/*******************************************************************************/

/* Sorting function to sort 1st candidate into DECREASING order of F */
int compare1F(const void *ip, const void *jp)
{
  REAL8 di, dj;

  di=CList1.F[*(const int *)ip];
  dj=CList1.F[*(const int *)jp];

  if (di<dj)
    return 1;
  
  if (di==dj)
    return (ip < jp);

  return -1;
}
/*******************************************************************************/

/* Sorting function to sort 1st candidate into DECREASING order of F */
int compare2F(const void *ip, const void *jp)
{
  REAL8 di, dj;

  di=CList2.F[*(const int *)ip];
  dj=CList2.F[*(const int *)jp];

  if (di<dj)
    return 1;
  
  if (di==dj)
    return (ip < jp);

  return -1;
}

/*******************************************************************************/

/* Sorting function to sort 1st candidate into DECREASING order of F */
int compare3F(const void *ip, const void *jp)
{
  REAL8 di, dj;

  di=CList3.F[*(const int *)ip];
  dj=CList3.F[*(const int *)jp];

  if (di<dj)
    return 1;
  
  if (di==dj)
    return (ip < jp);

  return -1;
}
/*******************************************************************************/

/* Sorting function to sort 1st candidate into DECREASING order of F */
int compare4F(const void *ip, const void *jp)
{
  REAL8 di, dj;

  di=CList4.F[*(const int *)ip];
  dj=CList4.F[*(const int *)jp];

  if (di<dj)
    return 1;
  
  if (di==dj)
    return (ip < jp);

  return -1;
}

/*******************************************************************************/

/* Sorting function to sort second candidate list into DECREASING order of f */
int compare2f(const void *ip, const void *jp)
{
  REAL8 di, dj;

  di=CList2.f[*(const int *)ip];
  dj=CList2.f[*(const int *)jp];

  if (di<dj)
    return 1;
  
  if (di==dj)
    return (ip < jp);

  return -1;
}
/*******************************************************************************/

/* Sorting function to sort second candidate list into increasing order of fa */
int compareCCfa(const void *ip, const void *jp)
{
  REAL8 di, dj;

  di=CC[*(const int *)ip].fa;
  dj=CC[*(const int *)jp].fa;

  if (di<dj)
    return -1;
  
  if (di==dj)
    return (ip > jp);

  return 1;
}


/*******************************************************************************/

/* Sorting function to sort pair list into increasing order of fa */
int compareCPfa(const void *ip, const void *jp)
{
  REAL8 di, dj;

  di=CP[*(const int *)ip].fa;
  dj=CP[*(const int *)jp].fa;

  if (di<dj)
    return -1;
  
  if (di==dj)
    return (ip > jp);

  return 1;
}


/*******************************************************************************/

int ReadCandidateFiles(struct PolkaCommandLineArgsTag CLA)
{
  LALStatus status = blank_status;	/* initialize status */

  ReadOneCandidateFile (&status, &CList1, CLA.FstatsFile1);
  if (status.statusCode != 0) {
    REPORTSTATUS (&status);
    return 1;
  }
  ReadOneCandidateFile (&status, &CList2, CLA.FstatsFile2);
  if (status.statusCode != 0) {
    REPORTSTATUS (&status);
    return 1;
  }

  if (CLA.FstatsFile3)
    {
      ReadOneCandidateFile (&status, &CList3, CLA.FstatsFile3);
      if (status.statusCode != 0) {
	REPORTSTATUS (&status);
	return 1;
      }
    }

  if (CLA.FstatsFile4)
    {
      ReadOneCandidateFile (&status, &CList4, CLA.FstatsFile4);
      if (status.statusCode != 0) {
	REPORTSTATUS (&status);
	return 1;
      }
    }

  return 0;

} /* ReadCandidateFiles() */

/*******************************************************************************/


int ReadCommandLine(int argc,char *argv[],struct PolkaCommandLineArgsTag *CLA) 
{
  INT2 errflg = 0;
  INT4 c; 
  INT4 option_index = 0;

  const char *optstring = "h1:2:3:4:f:a:d:m:M:o:s:e:b";
  struct option long_options[] =
    {
      {"fstatsfile1", 		required_argument, 0, 	'1'},
      {"fstatsfile2", 		required_argument, 0, 	'2'},
      {"frequency-window", 	required_argument, 0, 	'f'},
      {"delta-window", 		required_argument, 0, 	'd'},
      {"alpha-window", 		required_argument, 0, 	'a'},
      {"fmin",   		required_argument, 0, 	's'},
      {"fmax",   		required_argument, 0, 	'e'},
      {"outputfile", 		required_argument, 0, 	'o'},
      {"EAHoutput", 		no_argument, 0, 	'b'},
      {"help", 			no_argument, 0, 	'h'},
      {0, 0, 0, 0}
    };

  /* Initialize default values */
  CLA->FstatsFile1=NULL;
  CLA->FstatsFile2=NULL;
  CLA->FstatsFile3=NULL;
  CLA->FstatsFile4=NULL;
  CLA->OutputFile=NULL;
  CLA->Deltaf=0.0;
  CLA->DeltaAlpha=0;
  CLA->DeltaDelta=0;
  CLA->fmin=0;
  CLA->fmax=0;
  CLA->EAH=0;

  /* reset gnu getopt */
  optind = 0;

  /* Scan through list of command line arguments */
  while (1)
    {
      c = getopt_long(argc, argv, optstring, long_options, &option_index);      
      if (c == -1) 
	break;
      switch (c) {
      case '1':
	/* SFT directory */
	CLA->FstatsFile1=optarg;
	break;
      case '2':
	/* calibration files directory */
	CLA->FstatsFile2=optarg;
	break;
      case '3':
	/* SFT directory */
	CLA->FstatsFile3=optarg;
	break;
      case '4':
	/* calibration files directory */
	CLA->FstatsFile4=optarg;
	break;
      case 'o':
	/* calibration files directory */
	CLA->OutputFile=optarg;
	break;
      case 'f':
	/* Spin down order */
	CLA->Deltaf=atof(optarg);
	break;
      case 'a':
	/* Spin down order */
	CLA->DeltaAlpha=atof(optarg);
	break;
      case 's':
	/* Spin down order */
	CLA->fmin=atof(optarg);
	break;
      case 'e':
	/* Spin down order */
	CLA->fmax=atof(optarg);
	break;
      case 'd':
	/* Spin down order */
	CLA->DeltaDelta=atof(optarg);
	break;
      case 'b':
	/* Spin down order */
	CLA->EAH=1;
	break;
      case 'h':
	/* print usage/help message */
	fprintf(stderr,"Arguments are (defaults):\n");
	fprintf(stderr,"\t--fstatsfile1 (-1)\tSTRING\tFirst candidates Fstats file\n");
	fprintf(stderr,"\t--fstatsfile1 (-2)\tSTRING\tSecond candidates Fstats file\n");
	fprintf(stderr,"\t--fstatsfile3 (-3)\tSTRING\tFstats used to compute false alarm for -1\n");
	fprintf(stderr,"\t--fstatsfile4 (-4)\tSTRING\tFstats used to compute false alarm for -2\n");
	fprintf(stderr,"\t--outputfile  (-o)\tSTRING\tName of ouput candidates file\n");
	fprintf(stderr,"\t--frequency-window (-f)\tFLOAT\tFrequency window in Hz (0.0)\n");
	fprintf(stderr,"\t--alpha-window (-a)\tFLOAT\tAlpha window in radians (0.0)\n");
	fprintf(stderr,"\t--delta-window (-d)\tFLOAT\tDelta window in radians (0.0)\n");
	fprintf(stderr,"\t--fmin (-s)\tFLOAT\t Minimum frequency of candidate in 1st IFO\n");
	fprintf(stderr,"\t--fmax (-e)\tFLOAT\t Maximum frequency of candidate in 1st IFO\n");
	fprintf(stderr,"\t--EAHoutput (-b)\tFLAG\t Einstein at home output flag. \n");
	fprintf(stderr,"\t--help        (-h)\t\tThis message\n");
	exit(0);
	break;
      default:
	/* unrecognized option */
	errflg++;
	fprintf(stderr,"Unrecognized option argument %c\n",c);
	exit(1);
	break;
      }
    }

  if(CLA->FstatsFile1 == NULL)
    {
      fprintf(stderr,"No 1st candidates file specified; input with -1 option.\n");
      fprintf(stderr,"For help type ./polka -h \n");
      return 1;
    }      
  if(CLA->FstatsFile2 == NULL)
    {
      fprintf(stderr,"No 2nd candidates file specified; input with -2 option.\n");
      fprintf(stderr,"For help type ./polka -h \n");
      return 1;
    }      
  if(CLA->OutputFile == NULL)
    {
      fprintf(stderr,"No ouput filename specified; input with -o option.\n");
      fprintf(stderr,"For help type ./polka -h \n");
      return 1;
    }      

  if(CLA->fmin == 0.0)
    {
      fprintf(stderr,"No minimum frequency specified.\n");
      fprintf(stderr,"For help type ./polka -h \n");
      return 1;
    }      

  if(CLA->fmax == 0.0)
    {
      fprintf(stderr,"No maximum frequency specified.\n");
      fprintf(stderr,"For help type ./polka -h \n");
      return 1;
    }      

  return errflg;
}

/*******************************************************************************/

#define DONE_MARKER "%DONE"
/* read and parse the given candidate 'Fstats'-file fname into the candidate-list CList */
void 
ReadOneCandidateFile (LALStatus *stat, CandidateList *CList, const char *fname)
{
  UINT4 i;
  UINT4 numlines;
  REAL8 dmp;
  LALParsedDataFile *Fstats =NULL;	/* pre-parsed contents of Fstats-file */
  const CHAR *thisline;
  CandidateList cands;
  
  INITSTATUS(stat);
  ATTATCHSTATUSPTR (stat);
 
  ASSERT ( fname, stat, POLKAC_ENULL, POLKAC_MSGENULL);
  ASSERT ( CList, stat, POLKAC_ENULL, POLKAC_MSGENULL);
  ASSERT ( CList->f == NULL && CList->Alpha == NULL && CList->Delta == NULL 
	   && CList->F == NULL && CList->fa == NULL && CList->Ctag == NULL && CList->CtagCounter == NULL, 
 	   stat, POLKAC_ENONULL, POLKAC_MSGENONULL);

  /* ------ Open and read candidate file ------ */
  TRY ( LALParseDataFile (stat->statusPtr, &Fstats, fname), stat);

  numlines = Fstats->lines->nTokens; /* how many lines of data */

  if ( numlines == 0) 
    {
      XLALPrintError ("ERROR: File '%s' is empty and is not properly terminated by '%s' marker!\n\n", fname, DONE_MARKER);
      TRY (LALDestroyParsedDataFile ( stat->statusPtr, &Fstats ), stat);
      ABORT (stat, POLKAC_EINVALIDFSTATS, POLKAC_MSGEINVALIDFSTATS);
    }

  /* check validity of this Fstats-file */
  thisline = Fstats->lines->tokens[numlines-1];	/* get last line */
  if ( strcmp(thisline, DONE_MARKER ) ) 
    {
      XLALPrintError ("ERROR: File '%s' is not properly terminated by '%s' marker!\n\n", fname, DONE_MARKER);
      TRY (LALDestroyParsedDataFile ( stat->statusPtr, &Fstats ), stat);
      ABORT (stat, POLKAC_EINVALIDFSTATS, POLKAC_MSGEINVALIDFSTATS);
    }
  else
    numlines --; 	/* avoid stepping on DONE-marker */
  
  /* reserve memory for fstats-file contents */
  cands.f     = LALCalloc (numlines, sizeof(REAL8));
  cands.Alpha = LALCalloc (numlines, sizeof(REAL8));
  cands.Delta = LALCalloc (numlines, sizeof(REAL8));
  cands.F     = LALCalloc (numlines, sizeof(REAL8));
  cands.fa    = LALCalloc (numlines, sizeof(REAL8));
  cands.Ctag  = LALCalloc (numlines, sizeof(UINT4));
  cands.CtagCounter  = LALCalloc (numlines, sizeof(INT4));


  if ( !cands.f || !cands.Alpha || !cands.Delta || !cands.F || !cands.fa || !cands.Ctag || !cands.CtagCounter )
    {
      TRY( LALDestroyParsedDataFile ( stat->statusPtr, &Fstats ), stat);
      ABORT (stat, POLKAC_EMEM, POLKAC_MSGEMEM);
    }

  for (i=0; i < numlines; i++)
    {
      int read;
      
      cands.Ctag[i]=0;
      cands.CtagCounter[i]=-1;

      thisline = Fstats->lines->tokens[i];
      read = sscanf (thisline, 
		     "%" LAL_REAL8_FORMAT " %" LAL_REAL8_FORMAT " %" LAL_REAL8_FORMAT " %" LAL_REAL8_FORMAT 
		     " %" LAL_REAL8_FORMAT " %" LAL_REAL8_FORMAT " %" LAL_REAL8_FORMAT, 
		     &(cands.f[i]), &(cands.Alpha[i]), &(cands.Delta[i]), &dmp, &dmp, &dmp, &(cands.F[i]) );
      if ( read != 7 )
	{
	  XLALPrintError ("Failed to parse line %d in file '%s' \n", i+1, fname);
	  TRY (LALDestroyParsedDataFile ( stat->statusPtr, &Fstats ), stat);
	  LALFree (cands.f);
	  LALFree (cands.Alpha);
	  LALFree (cands.Delta);
	  LALFree (cands.F);
	  LALFree (cands.fa);
	  LALFree (cands.Ctag);
	  LALFree (cands.CtagCounter);
	  ABORT (stat, POLKAC_EINVALIDFSTATS, POLKAC_MSGEINVALIDFSTATS);
	}
    } /* for i < numlines */
 
  /* we're done: get rid of raw data-file */
  TRY ( LALDestroyParsedDataFile ( stat->statusPtr, &Fstats ), stat);
  
  /* return final candidate-list */
  CList->length = numlines;
  CList->f      = cands.f;
  CList->Alpha  = cands.Alpha;
  CList->Delta  = cands.Delta;
  CList->F      = cands.F;
  CList->fa     = cands.fa;
  CList->Ctag   = cands.Ctag;
  CList->CtagCounter   = cands.CtagCounter;



  DETATCHSTATUSPTR (stat);
  RETURN (stat);

} /* ReadOneCandidateFile() */

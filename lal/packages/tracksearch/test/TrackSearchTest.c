/*
   Example program for the use of the the "TrackSearch() routine;
   */

#include <lal/TrackSearch.h>

NRCSID (TESTSTSC,"$Id$");

INT4 lalDebugLevel=0;
int main( void )
{
  static LALStatus status;
  static TrackSearchOut out;
  static TimeFreqRep in;
  static TrackSearchParams params;
  INT4 i,j,cnt,maxVal;
  FILE *fp;
  unsigned char dummy;
  char stringd[256];
  
  /* set the parameters */
  params.sigma=2; 
  params.high = 3.0;
  params.low=1;
  params.low = params.high/3;
  /* open an input pgm file */
  fp = fopen("a.pgm","r");
  fgets(stringd,255,fp);
  fgets(stringd,255,fp);
  /* read the height and width of the image */
  fscanf(fp,"%d ",&params.height);
  fscanf(fp,"%d ",&params.width);
  fscanf(fp,"%d ",&maxVal);
  /* Allocate space for the input array */
  in.map=LALMalloc(params.height*sizeof(REAL4 *));
  for(i=0;i<params.height;i++)
    in.map[i] = LALMalloc(params.width*sizeof(REAL4));
  /* Read the image */
  for(j=0;j<params.width;j++){  
    for(i=0;i<params.height;i++){ 
      fscanf(fp,"%c",&dummy);
      /* bright parts of the image should have higer values
	 hence the inversion */
      *(in.map[i] + j) = maxVal - dummy;
    }
  }
  fclose(fp);
  /* set the allocFlag so that space can be allocated */
  params.allocFlag=1;
  /* Search for curves */
  LALSignalTrackSearch(&status, &out, &in, &params);
  REPORTSTATUS(&status);
  /* Output the details of the curves found.*/
    for(i=0;i<params.height;i++)
      for(j=0;j<params.width;j++)
	*(in.map[i] + j)=0;
  printf("number of curves %d\n",out.numberOfCurves);
  cnt=0;
  for(i=0;i<out.numberOfCurves;i++){
    if(out.curves[i].n>5){
      cnt++;
      printf(" curve number = %d, length =  %d\n",cnt,out.curves[i].n);
      for(j=0;j<out.curves[i].n;j++){
	printf("%d %d \n",out.curves[i].row[j],out.curves[i].col[j]);
	*(in.map[out.curves[i].row[j]] + out.curves[i].col[j]) = cnt;
      }
    }
  }    
  
  /* Free the space allocated for output structures */
  for(i=0;i<out.numberOfCurves;i++){
    LALFree(out.curves[i].row);
    LALFree(out.curves[i].col);
  }
  if(out.curves!=NULL)
    LALFree(out.curves);
  out.curves=NULL;
  /* set params.allocFlag=2 and call the routine again to free space*/
  params.allocFlag=2;
  LALSignalTrackSearch(&status, &out, &in, &params);
  REPORTSTATUS(&status);
  return(0);
}











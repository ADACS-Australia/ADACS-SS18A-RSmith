/* $Id$ */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>
#include "SFTReferenceLibrary.h"

int main(int argc, char** argv) {
  int i;
  float *data=NULL;
  const char* rcsid = "$Id$";
  
  /* loop over all file names on command line */
  for (i=1; i<argc; i++) {
    FILE *fp;
    int count;

    /* open the file */
    if (!(fp=fopen(argv[i], "r"))) {
      fprintf(stderr,"Unable to open %s", argv[i]);
      if (errno)
	perror(" ");
      return SFTENULLFP;
    }

    /* and read successive SFTs blocks from the file and validate CRC
       checksums */
    for (count=0; 1; count++) {
      struct headertag2 info,lastinfo;
      int err=0, swapendian, move, j;
      
      err=ReadSFTHeader(fp, &info, NULL, &swapendian, 1);

      /* at end of SFT file or merged SFT file blocks */
      if (err==SFTENONE && count)
	break;
      
      /* SFT was invalid: say why */
      if (err) {
	fprintf(stderr, "%s\n%s is not a valid SFT. %s\n", rcsid, argv[i], SFTErrorMessage(err));
	if (errno)
	  perror(NULL);
	return err;
      }

      /* check that various bits of header information are consistent */
      if (count && (err=CheckSFTHeaderConsistency(&lastinfo, &info)))
	{
	  fprintf(stderr, "%s\n%s is not a valid SFT. %s\n", rcsid, argv[i], SFTErrorMessage(err));
	  if (errno)
	    perror(NULL);
	  return err;
	}
      
      /* check that data appears valid */
      data=(float *)realloc((void *)data, info.nsamples*4*2);
      if (!data) {
	errno=SFTENULLPOINTER;
	fprintf(stderr, "%s\nran out of memory at %s. %s\n", rcsid, argv[i], SFTErrorMessage(err));
	if (errno)
	  perror(NULL);
	return err;
      }

      err=ReadSFTData(fp, data, info.firstfreqindex, info.nsamples, /*comment*/ NULL, /*headerinfo */ NULL);
      if (err) {
	fprintf(stderr, "%s\n%s is not a valid SFT. %s\n", rcsid, argv[i], SFTErrorMessage(err));
	if (errno)
	  perror(NULL);
	return err;
      }

      for (j=0; j<info.nsamples; j++) {
	if (!finite(data[2*j]) || !finite(data[2*j+1])) {
	  fprintf(stderr, "%s\n%s is not a valid SFT (data infinite at freq bin %d)\n", rcsid, argv[i], j+info.firstfreqindex);
	  return SFTNOTFINITE;
	}
      }

      /* keep copy of header for comparison the next time */
      lastinfo=info;
      
      /* Move forward to next SFT in merged file */
      if (info.version==1)
	move=sizeof(struct headertag1)+info.nsamples*2*sizeof(float);
      else
	move=sizeof(struct headertag2)+info.nsamples*2*sizeof(float)+info.comment_length;
      fseek(fp, move, SEEK_CUR);
    }
    fclose(fp);
  }
  return 0;
}

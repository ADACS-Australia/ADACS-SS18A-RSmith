/*
 * Copyright (C) 2004, 2005 R. Prix, B. Machenschalk, A.M. Sintes
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

/** \file
 * \ingroup SFTfileIO
 * \author R. Prix, B. Machenschalk, A.M. Sintes
 * 
 * \brief IO library for reading/writing "Short Fourier transform" (SFT) data files.
 *
 * $Id$ 
 *
 * \todo Sort by GPS-time, not by filename
 */

/*---------- INCLUDES ----------*/
#include <sys/types.h>
#ifndef _MSC_VER
#include <dirent.h>
#else
#include <io.h>
#endif

#include <lal/FileIO.h>
#include <lal/SFTfileIO.h>

/*---------- DEFINES ----------*/
NRCSID (SFTFILEIOC, "$Id$");

/*----- Macros ----- */
#define GPS2REAL8(gps) (1.0 * (gps).gpsSeconds + 1.e-9 * (gps).gpsNanoSeconds )

/*---------- internal types ----------*/
/* for private use: a vector of 'strings' */
typedef struct {
  UINT4 length;
  CHAR **data;
} StringVector;

/*---------- empty initializers ---------- */
/*---------- Global variables ----------*/

/*---------- internal prototypes ----------*/
static StringVector *find_files (const CHAR *fpattern);
static void DestroyStringVector (StringVector *strings);
static void SortStringVector (StringVector *strings);
static void endian_swap(CHAR * pdata, size_t dsize, size_t nelements);
static int amatch(char *str, char *p);	/* glob pattern-matcher (public domain)*/

/* number of bytes in SFT-header v1.0 */
static const size_t header_len_v1 = 32;	


/*==================== FUNCTION DEFINITIONS ====================*/

/** Function to read and return a list of SFT-headers for given 
 * filepattern and start/end times.
 *
 * The \a startTime and \a endTime entries are allowed to be NULL, 
 * in which case they are ignored.
 *
 * \note We return the headers as an SFTVector, but with empty data-fields.
 */
void
LALGetSFTheaders (LALStatus *status,
		  SFTVector **headers,		/**< [out] Vector of SFT-headers */
		  const CHAR *fpattern,		/**< path/filepattern */
		  const LIGOTimeGPS *startTime,	/**< include only SFTs after this time (can be NULL) */
		  const LIGOTimeGPS *endTime)	/**< include only SFTs before this (can be NULL)*/
{
  UINT4 i, numSFTs;
  SFTVector *out = NULL;
  StringVector *fnames;
  REAL8 t0, t1;		/* start- and end-times as reals */
  UINT4 numHeaders;

  INITSTATUS (status, "LALGetSFTheaders", SFTFILEIOC);
  ATTATCHSTATUSPTR (status); 

  /* check input */
  ASSERT ( headers, status,  SFTFILEIO_ENULL,  SFTFILEIO_MSGENULL);
  ASSERT ( *headers == NULL, status,  SFTFILEIO_ENONULL,  SFTFILEIO_MSGENONULL);
  ASSERT ( fpattern, status,  SFTFILEIO_ENULL,  SFTFILEIO_MSGENULL);

  if ( startTime )
    t0 = GPS2REAL8( *startTime );
  else
    t0 = 0;
  
  if ( endTime )
    t1 = GPS2REAL8 ( *endTime );
  else
    t1 = 0;

  /* get filelist of files matching fpattern */
  if ( (fnames = find_files (fpattern)) == NULL) {
    ABORT (status, SFTFILEIO_EGLOB, SFTFILEIO_MSGEGLOB);
  }

  /* prepare output-vector */
  if ( (out = LALCalloc (1, sizeof(*out) )) == NULL ) {
    ABORT ( status, SFTFILEIO_EMEM, SFTFILEIO_MSGEMEM );
  }

  numSFTs = fnames->length;

  /* main loop: load all SFT-headers and put them into an SFTvector */
  numHeaders = 0;
  for (i=0; i < numSFTs; i++)
    {
      SFTHeader header;
      REAL8 epoch;
      SFTtype *sft;

      TRY ( LALReadSFTheader (status->statusPtr, &header, fnames->data[i]), status );
      epoch = 1.0 * header.gpsSeconds + 1.0e-9 * header.gpsNanoSeconds;

      if ( t0 && ( epoch < t0 ) )	/* epoch earlier than start-time ==> skip */
	continue;
      if ( t1 && ( epoch > t1 ) )	/* epoch later than end-time ==> skip */
	continue;

      /* found suitable SFT ==> add to SFTVector of headers */
      numHeaders ++;
      if ( ( out->data = LALRealloc(out->data, numHeaders * sizeof( *out->data) )) == NULL ) 
	{
	  LALFree (out);
	  ABORT ( status, SFTFILEIO_EMEM, SFTFILEIO_MSGEMEM );
	}
      
      sft = &(out->data[numHeaders - 1]);

      /* fill in header-data */
      strncpy (sft->name, fnames->data[i], LALNameLength);
      sft->name[LALNameLength - 1 ] = '\0';	/* make sure it's 0-terminated */
      sft->deltaF  			= 1.0 / header.timeBase;
      sft->f0      			= header.fminBinIndex / header.timeBase;
      sft->epoch.gpsSeconds     	= header.gpsSeconds;
      sft->epoch.gpsNanoSeconds 	= header.gpsNanoSeconds;

      sft->data = NULL;	/* no SFT-data proper */

    } /* for i < numSFTs */

  out->length = numHeaders;

  DestroyStringVector (fnames);

  *headers = out;
  
  DETATCHSTATUSPTR (status);
  RETURN(status);

} /* LALGetSFTheaders() */		  



/** Basic SFT reading-function.
 *  Given a filename \a fname and frequency-limits [\a fMin, \a fMax], 
 *  returns an SFTtype \a sft containing the SFT-data.
 * 
 * \note 1) the actual returned frequency-band is 
 * <tt>[floor(Tsft * fMin), ceil(Tsft * fMax)] / Tsft</tt>,
 * i.e. we need to round to an integer frequency-bin within the SFT, but
 * the requested frequency-band is guaranteed to be contained in the output 
 * (if present in the SFT-file), but can be slightly larger.
 * 
 * 2) The special input <tt>fMin=fMax=0</tt> means to read and
 * return the <em>whole</em> frequency-band contained in the SFT-file.
 *
 * 3) Currently only SFTv1 are supported!!
 *
 */
void
LALReadSFTfile (LALStatus *status, 
		SFTtype **sft, 		/**< [out] output SFT */
		REAL8 fMin, 		/**< lower frequency-limit */
		REAL8 fMax,		/**< upper frequency-limit */
		const CHAR *fname)	/**< path+filename */
{ 
  SFTHeader  header;		/* SFT file-header version1 */
  REAL8 deltaF;
  UINT4 readlen;
  INT4 fminBinIndex, fmaxBinIndex;
  SFTtype *outputSFT = NULL;
  UINT4 i;
  REAL4 renorm;

  INITSTATUS (status, "LALReadSFTfile", SFTFILEIOC);
  ATTATCHSTATUSPTR (status); 
  
  ASSERT (sft, status, SFTFILEIO_ENULL,  SFTFILEIO_MSGENULL);
  ASSERT (*sft == NULL, status, SFTFILEIO_ENONULL, SFTFILEIO_MSGENONULL);
  ASSERT (fname,  status, SFTFILEIO_ENULL,  SFTFILEIO_MSGENULL);
  ASSERT (fMin <= fMax, status, SFTFILEIO_EVAL, SFTFILEIO_MSGEVAL);

  /* read the header */
  TRY ( LALReadSFTheader (status->statusPtr, &header, fname), status);

  /* ----- figure out which data we want to read ----- */
  deltaF = 1.0 / header.timeBase;

  /* special case: fMin==fMax==0 means "read all" */
  if ( (fMin == 0) && (fMax == 0) )
    {
      fminBinIndex = header.fminBinIndex;
      fmaxBinIndex = fminBinIndex + header.length - 1;
    }
  else
    {
      /* find the right frequency-bin and number of bins
       * The rounding here is chosen such that the required 
       * frequency-interval is _guaranteed_ to lie within the 
       * returned range  */
      fminBinIndex = (INT4) floor (fMin * header.timeBase);  /* round this down */
      fmaxBinIndex = (INT4) ceil  (fMax * header.timeBase);  /* round up */
    }

  readlen = (UINT4)(fmaxBinIndex - fminBinIndex) + 1;	/* number of bins to read */

  /* allocate the final SFT to be returned */
  TRY ( LALCreateSFTtype (status->statusPtr, &outputSFT, readlen), status);


  /* and read it, using the lower-level function: */
  LALReadSFTdata (status->statusPtr, outputSFT, fname, fminBinIndex);
  BEGINFAIL (status) {
    LALDestroySFTtype (status->statusPtr, &outputSFT);
  } ENDFAIL (status);


  /*
   * NOTE: the following renormalization is necessary for v1-SFTs
   * as the data are not multiplied by dt, therefore extracting a 
   * sub-band B' of the total band B requires mulitplication of 
   * the data by B'/B.
   * SFTv2 will store FFT-data multiplied by dt and then this will
   * not be necessary any more.
   * 
   */
  renorm = 1.0 * readlen / header.length;

  /* let's re-normalize and fill data into output-vector */
  if (renorm != 1)
    for (i=0; i < readlen; i++)
      {
	outputSFT->data->data[i].re *= renorm;
	outputSFT->data->data[i].im *= renorm;
      }    
  
  /* that's it: return */
  *sft = outputSFT;

  DETATCHSTATUSPTR (status);
  RETURN(status);

} /* LALReadSFTfile() */



/** Higher-level SFT-reading function to read a whole vector of SFT files 
 * and return an SFTvector \a sftvect. The handling of
 * [fMin, fMax] is identical to LALReadSFTfile().
 * 
 * \note 1) the file-pattern \a fpattern can use a wide range of glob-patterns.
 *
 * 2) currently the SFTs matching the pattern are required to have the same 
 * number of frequency bins, otherwise an error will be returned. 
 * (This might be relaxed in the future).
 *
 * 3) This function does <em>not</em> use <tt>glob()</tt> and should therefore
 * be safe even under condor.
 *
 */
void
LALReadSFTfiles (LALStatus *status,
		 SFTVector **sftvect,	/**< [out] output SFT vector */
		 REAL8 fMin,	       	/**< lower frequency-limit */
		 REAL8 fMax,		/**< upper frequency-limit */
		 UINT4 wingBins,	/**< number of frequency-bins to be added left and right. */
		 const CHAR *fpattern)	/**< path/filepattern */
{

  UINT4 i, numSFTs;
  SFTVector *out = NULL;
  SFTtype *oneSFT = NULL;
  SFTHeader header;
  REAL8 dFreq; 		/* frequency spacing in SFT */
  REAL8 fWing;		/* frequency band to be added as "wings" to the 'physical band' */
  StringVector *fnames;

  INITSTATUS (status, "LALReadSFTfiles", SFTFILEIOC);
  ATTATCHSTATUSPTR (status); 
  
  ASSERT (sftvect, status, SFTFILEIO_ENULL,  SFTFILEIO_MSGENULL);
  ASSERT (*sftvect == NULL, status, SFTFILEIO_ENONULL, SFTFILEIO_MSGENONULL);
  ASSERT (fpattern,  status, SFTFILEIO_ENULL,  SFTFILEIO_MSGENULL);
  ASSERT (fMin <= fMax, status, SFTFILEIO_EVAL, SFTFILEIO_MSGEVAL);

  /* make filelist 
   * NOTE: we don't use glob() as it was reported to fail under condor */
  if ( (fnames = find_files (fpattern)) == NULL) {
    ABORT (status, SFTFILEIO_EGLOB, SFTFILEIO_MSGEGLOB);
  }

  /* read header of first sft to determine Tsft, and therefore dfreq */
  LALReadSFTheader(status->statusPtr, &header, fnames->data[0]);
  BEGINFAIL(status) {
    DestroyStringVector (fnames);    
  } ENDFAIL(status);

  numSFTs = fnames->length;
  dFreq = 1.0 / header.timeBase;
  fWing = wingBins * dFreq;

  /* main loop: load all SFTs and put them into the SFTvector */
  for (i=0; i < numSFTs; i++)
    {
      LALReadSFTfile (status->statusPtr, &oneSFT, fMin-fWing, fMax+fWing, fnames->data[i]);
      BEGINFAIL (status) {
	DestroyStringVector (fnames);    
	LALDestroySFTtype (status->statusPtr, &oneSFT);
	if (out) LALDestroySFTVector (status->statusPtr, &out);
      } ENDFAIL (status);

      /* after the first time we can allocate the output-vector, 
       * as we now kpnow how many frequency-bins we need */
      if (out == NULL)
	{
	  LALCreateSFTVector(status->statusPtr, &out, numSFTs, oneSFT->data->length);
	  BEGINFAIL(status) {
	    DestroyStringVector (fnames);    
	    LALDestroySFTtype (status->statusPtr, &oneSFT);
	  } ENDFAIL(status);
	}

      /* make sure all SFTs have same length */
      if ( oneSFT->data->length != out->data->data->length)
	{
	  DestroyStringVector (fnames);    
	  LALDestroySFTtype (status->statusPtr, &oneSFT);
	  LALDestroySFTVector (status->statusPtr, &out);
	  ABORT (status, SFTFILEIO_EDIFFLENGTH, SFTFILEIO_MSGEDIFFLENGTH);
	} /* if length(thisSFT) != common length */

      /* transfer the returned SFT into the SFTVector 
       * this is a bit complicated by the fact that it's a vector
       * of SFTtypes, not pointers: therefore we need to *COPY* the stuff !
       */
      LALCopySFT (status->statusPtr, &(out->data[i]), oneSFT);
      BEGINFAIL (status) {
	DestroyStringVector (fnames);    
	LALDestroySFTtype (status->statusPtr, &oneSFT);
	LALDestroySFTVector (status->statusPtr, &out);
      } ENDFAIL (status);

      LALDestroySFTtype (status->statusPtr, &oneSFT);
      oneSFT = NULL;	/* important for next call of LALReadSFTfile()! */

    } /* for i < numSFTs */

  DestroyStringVector (fnames);

  *sftvect = out;

  DETATCHSTATUSPTR (status);
  RETURN (status);

} /* LALReadSFTfiles () */



/** Write an entire SFT to a file. 
 * 
 * \note: currently only SFT-spec v1.0 is supported
 *
 */
void
LALWriteSFTfile (LALStatus  *status,
		 const SFTtype *sft,		/**< SFT to write to disk */
		 const CHAR *outfname)		/**< filename */
{
  FILE  *fp = NULL;
  COMPLEX8  *inData;
  INT4  i;
  UINT4 datalen;
  REAL4  *rawdata;
  CHAR *rawheader, *ptr;
  SFTHeader header;

  INITSTATUS (status, "LALWriteSFTfile", SFTFILEIOC);
  ATTATCHSTATUSPTR (status);   
 
  /*   Make sure the arguments are not NULL and perform basic checks*/ 
  ASSERT (sft,   status, SFTFILEIO_ENULL, SFTFILEIO_MSGENULL);
  ASSERT (sft->data,  status, SFTFILEIO_EVAL, SFTFILEIO_MSGEVAL);
  ASSERT (sft->deltaF > 0, status, SFTFILEIO_EVAL, SFTFILEIO_MSGEVAL);
  ASSERT (outfname, status, SFTFILEIO_ENULL, SFTFILEIO_MSGENULL); 

  /* fill in the header information */
  header.version = 1.0;
  header.gpsSeconds = sft->epoch.gpsSeconds;
  header.gpsNanoSeconds = sft->epoch.gpsNanoSeconds;
  header.timeBase = 1.0 / sft->deltaF;
  header.fminBinIndex = (INT4) floor (sft->f0 / sft->deltaF + 0.5);	/* round to closest int! */
  header.length = sft->data->length; 

  /* build raw header for writing to disk */
  rawheader = LALCalloc (1, header_len_v1);
  if (rawheader == NULL) {
    ABORT (status, SFTFILEIO_EMEM, SFTFILEIO_MSGEMEM);    
  }
  ptr = rawheader;
  *(REAL8*) ptr = header.version;
  ptr += sizeof (REAL8);
  *(INT4*) ptr = header.gpsSeconds;
  ptr += sizeof (INT4);
  *(INT4*) ptr = header.gpsNanoSeconds;
  ptr += sizeof (INT4);
  *(REAL8*) ptr = header.timeBase;
  ptr += sizeof (REAL8);
  *(INT4*) ptr = header.fminBinIndex;
  ptr += sizeof (INT4);
  *(INT4*) ptr = header.length; 

  /* write data into a contiguous REAL4-array */
  datalen = 2 * header.length * sizeof(REAL4);	/* amount of bytes for SFT-data */

  rawdata = LALCalloc (1, datalen);
  if (rawdata == NULL) {
    LALFree (rawheader);
    ABORT (status, SFTFILEIO_EMEM, SFTFILEIO_MSGEMEM);    
  }

  inData = sft->data->data;
  for ( i = 0; i < header.length; i++)
    {
      rawdata[2 * i]     = inData[i].re;
      rawdata[2 * i + 1] = inData[i].im;
    } /* for i < length */


  /* open the file for writing */
  fp = fopen(outfname, "wb");
  if (fp == NULL) {
    LALFree (rawheader);
    LALFree (rawdata);
    ABORT (status, SFTFILEIO_EFILE,  SFTFILEIO_MSGEFILE);
  }

  /* write the header*/
  if( fwrite( rawheader, header_len_v1, 1, fp) != 1) {
    LALFree (rawheader);
    LALFree (rawdata);
    fclose (fp);
    ABORT (status, SFTFILEIO_EFILE, SFTFILEIO_MSGEFILE);
  }
  
  /* write the data */
  if (fwrite( rawdata, datalen, 1, fp) != 1) {
    LALFree (rawheader);
    LALFree (rawdata);
    fclose (fp);
    ABORT (status, SFTFILEIO_EFILE, SFTFILEIO_MSGEFILE);
  }

  /* done */
  fclose(fp);
  LALFree (rawheader);
  LALFree (rawdata);


  DETATCHSTATUSPTR (status);
  RETURN (status);

} /* WriteSFTtoFile() */



/*================================================================================
 * LOW-level SFT-handling functions, deprecated, and should *NOT* be used outside
 * this file!
 *================================================================================*/

/** Lower-level function to read only the SFT-header of a given file. */
void 
LALReadSFTheader (LALStatus  *status,
		  SFTHeader   *header,	/**< [out] returned header */
		  const CHAR  *fname)	/**< path+filename */
{
  FILE *fp = NULL;
  SFTHeader  header1;
  CHAR *rawheader = NULL;
  CHAR *ptr = NULL;
  CHAR inVersion[8];
  REAL8 version;
  BOOLEAN swapEndian = 0;

  INITSTATUS (status, "LALReadSFTHeader", SFTFILEIOC);
  ATTATCHSTATUSPTR (status); 
  
  /*   Make sure the arguments are not NULL: */ 
  ASSERT (header, status, SFTFILEIO_ENULL,  SFTFILEIO_MSGENULL);
  ASSERT (fname,  status, SFTFILEIO_ENULL,  SFTFILEIO_MSGENULL);
  
  /* opening the SFT binary file */
  fp = LALOpenDataFile( fname );
  if (fp == NULL) {
    ABORT (status, SFTFILEIO_EFILE,  SFTFILEIO_MSGEFILE);
  }
  
  /* read version-number */
  if  (fread (inVersion, sizeof(inVersion), 1, fp) != 1) {
    fclose (fp);
    if (lalDebugLevel) LALPrintError ("\nInvalid SFT-file: %s\n\n", fname);
    ABORT (status, SFTFILEIO_EHEADER,  SFTFILEIO_MSGEHEADER);
  }

  /* try version 1.0 */
  version = 1.0;
  /* try swapping the version if it is not equal */
  if(memcmp(inVersion,&version,sizeof(version))){
    endian_swap (inVersion, sizeof(inVersion),1);
    swapEndian = 1;
    /* fail if still not true */
    if(memcmp(inVersion,&version,sizeof(version))){
      fclose (fp);
      if (lalDebugLevel) LALPrintError ("\nOnly v1-SFTs supported at the moment!: %s\n\n", fname);
      ABORT (status, SFTFILEIO_EHEADER,  SFTFILEIO_MSGEHEADER);
    }
  }

  /* read the whole header */
  rawheader = LALCalloc (1, header_len_v1);
  if (rawheader == NULL) {
    fclose (fp);
    ABORT (status, SFTFILEIO_EMEM, SFTFILEIO_MSGEMEM);    
  }
  
  rewind (fp);	/* go back to start */
  if (fread( rawheader, header_len_v1, 1, fp) != 1) {
    fclose (fp);
    LALFree (rawheader);
    if (lalDebugLevel) LALPrintError ("\nInvalid SFT-file: %s\n\n", fname);
    ABORT (status, SFTFILEIO_EHEADER,  SFTFILEIO_MSGEHEADER);
  }

  fclose(fp);

  /* now fill-in the header-struct with the appropriate fields */
  /* NOTE: we have to do it this way, because the struct can have 
   * padding in memory, so the fields are not guaranteed to lie 'close'
   * Endian-swapping ist done here if necessary
   */
  ptr = rawheader;
  if (swapEndian) endian_swap((CHAR*)ptr,sizeof(REAL8),1);
  header1.version	= *(REAL8*) ptr;
  ptr += sizeof(REAL8);
  if (swapEndian) endian_swap((CHAR*)ptr,sizeof(INT4),1);
  header1.gpsSeconds 	= *(INT4*) ptr;
  ptr += sizeof(INT4);
  if (swapEndian) endian_swap((CHAR*)ptr,sizeof(INT4),1);
  header1.gpsNanoSeconds= *(INT4*) ptr;
  ptr += sizeof(INT4);
  if (swapEndian) endian_swap((CHAR*)ptr,sizeof(REAL8),1);
  header1.timeBase      = *(REAL8*) ptr;
  ptr += sizeof(REAL8);
  if (swapEndian) endian_swap((CHAR*)ptr,sizeof(INT4),1);
  header1.fminBinIndex  = *(INT4*) ptr;
  ptr += sizeof(INT4);
  if (swapEndian) endian_swap((CHAR*)ptr,sizeof(INT4),1);
  header1.length     	= *(INT4*) ptr;

  LALFree (rawheader);

  /* ----- do some consistency-checks on the header-fields: ----- */

  /* gps_sec and gps_nsec >= 0 */
  if ( (header1.gpsSeconds < 0) || (header1.gpsNanoSeconds <0) ) {
    if (lalDebugLevel) LALPrintError ("\nInvalid SFT-file: %s\n\n", fname);
    ABORT (status, SFTFILEIO_EHEADER,  SFTFILEIO_MSGEHEADER);
  }

  /* tbase > 0 */
  if ( header1.timeBase <= 0 ) {
    if (lalDebugLevel) LALPrintError ("\nInvalid SFT-file: %s\n\n", fname);
    ABORT (status, SFTFILEIO_EHEADER,  SFTFILEIO_MSGEHEADER);
  }

  /* fminindex >= 0 */
  if (header1.fminBinIndex < 0) {
    if (lalDebugLevel) LALPrintError ("\nInvalid SFT-file: %s\n\n", fname);
    ABORT (status, SFTFILEIO_EHEADER,  SFTFILEIO_MSGEHEADER);
  }
  
  /* nsamples >= 0 */
  if (header1.length < 0) {
    if (lalDebugLevel) LALPrintError ("\nInvalid SFT-file: %s\n\n", fname);
    ABORT (status, SFTFILEIO_EHEADER,  SFTFILEIO_MSGEHEADER);
  }

  /* ok, the SFT-header seems consistent, so let's return it */
  *header = header1;
  
  DETATCHSTATUSPTR (status);
  RETURN (status);

} /* LALReadSFTheader() */



/** This is a function for low-level SFT data-reading:
 * the SFT-data is read starting from fminBinIndex and filled
 * into the pre-allocate vector sft of length N
 *
 * NOTE: !! NO re-normalization is done here!! this remains up
 *       to the caller of this function!!
 *
 */
void
LALReadSFTdata(LALStatus *status,
	       SFTtype    *sft,    /**< [out] output-SFT: assuming memory is allocated  */
	       const CHAR *fname,  /**< path+filename */
	       INT4 fminBinIndex)  /**< minimun frequency-index to read */
{ 
  FILE        *fp = NULL;
  SFTHeader  header;
  UINT4 offset, readlen;
  REAL4 *rawdata = NULL;
  CHAR inVersion[8];
  REAL8 version;
  BOOLEAN swapEndian = 0;
  UINT4 i;

  INITSTATUS (status, "LALReadSFTdata", SFTFILEIOC);
  ATTATCHSTATUSPTR (status); 
  
  /*   Make sure the arguments are not NULL: */ 
  ASSERT (sft,   status, SFTFILEIO_ENULL, SFTFILEIO_MSGENULL);
  ASSERT (sft->data, status, SFTFILEIO_ENULL, SFTFILEIO_MSGENULL);
  ASSERT (fname, status, SFTFILEIO_ENULL, SFTFILEIO_MSGENULL);
  
  /* Read header */
  TRY ( LALReadSFTheader (status->statusPtr, &header, fname), status);

  /* check that the required frequency-interval is part of the SFT */
  readlen = sft->data->length;
  if ( (fminBinIndex < header.fminBinIndex) 
       || (fminBinIndex + (INT4)readlen > header.fminBinIndex + header.length) ) {
    ABORT (status, SFTFILEIO_EFREQBAND, SFTFILEIO_MSGEFREQBAND);
  }

  /* how many frequency-bins to skip */
  offset = fminBinIndex - header.fminBinIndex;

  /* open file for reading */
  if ( (fp = LALOpenDataFile( fname )) == NULL) {
    ABORT (status, SFTFILEIO_EFILE, SFTFILEIO_MSGEFILE);
  }

  /* read version-number */
  if  (fread (inVersion, sizeof(inVersion), 1, fp) != 1) {
    fclose (fp);
    if (lalDebugLevel) LALPrintError ("\nInvalid SFT-file: %s\n\n", fname);
    ABORT (status, SFTFILEIO_EHEADER,  SFTFILEIO_MSGEHEADER);
  }

  /* set invalid version */
  version = -1.0;

  if (version < 0) {
    /* try version 1.0 */
    version = 1.0;
    /* try swapping the version if it is not equal */
    if(memcmp(inVersion,&version,sizeof(version))){
      endian_swap (inVersion, sizeof(inVersion),1);
      swapEndian = 1;
      /* set invalid version if still not true */
      if(memcmp(inVersion,&version,sizeof(version))){
	version = -1;
	endian_swap (inVersion, sizeof(inVersion),1);
	swapEndian = 0;
      }
    }
  }

  /* fail if the version is invalid */
  if (version < 0) {
    fclose (fp);
    if (lalDebugLevel) LALPrintError ("\nInvalid SFT-file: %s\n\n", fname);
    ABORT (status, SFTFILEIO_EHEADER,  SFTFILEIO_MSGEHEADER);
  }

  /* check compatibility of version with this function */
  if (version != 1) {
    fclose (fp);
    ABORT (status, SFTFILEIO_EVERSION, SFTFILEIO_MSGEVERSION);
  }

  /* skip SFT-header in file */
  rewind (fp);
  if (fseek(fp, header_len_v1, SEEK_SET) != 0) {
    fclose (fp);
    ABORT (status, SFTFILEIO_EFILE, SFTFILEIO_MSGEFILE);
  }

  /* skip offset data points to the correct frequency-bin */
  if (fseek(fp, offset * 2 * sizeof(REAL4), SEEK_CUR) != 0) {
    fclose (fp);
    ABORT (status, SFTFILEIO_EFILE, SFTFILEIO_MSGEFILE);
  }

  /* ----- prepare memory for data-reading ----- */
  rawdata = LALCalloc (1, 2 * readlen *sizeof(REAL4) );
  if (rawdata == NULL) {
    fclose (fp);
    ABORT (status, SFTFILEIO_EMEM, SFTFILEIO_MSGEMEM);
  }

  /* we don't rely on memory-packing, so we read into a REAL4 array first */
  if (fread( rawdata, 2 * readlen * sizeof(REAL4), 1, fp) != 1) {
    fclose(fp);
    LALFree (rawdata);
    ABORT (status, SFTFILEIO_EFILE, SFTFILEIO_MSGEFILE);
  }
  fclose(fp);

  /* now fill data into output-vector */
  for (i=0; i < readlen; i++)
    {
      if (swapEndian)
	endian_swap((CHAR*)&rawdata[2*i], sizeof(REAL4), 2);
      sft->data->data[i].re = rawdata[2 * i];
      sft->data->data[i].im = rawdata[2 * i + 1];
    }    
  
  LALFree (rawdata);

  /* now fill in the header-info */
  strncpy (sft->name, fname, LALNameLength);
  sft->name[LALNameLength - 1 ] = '\0';	/* make sure it's 0-terminated */
  sft->deltaF  			= 1.0 / header.timeBase;
  sft->f0      			= fminBinIndex / header.timeBase;
  sft->epoch.gpsSeconds     	= header.gpsSeconds;
  sft->epoch.gpsNanoSeconds 	= header.gpsNanoSeconds;

  
  DETATCHSTATUSPTR (status);
  RETURN (status);

} /* LALReadSFTdata() */



/***********************************************************************
 * internal helper functions
 ***********************************************************************/

/* a little endian-swapper needed for SFT reading/writing */
static void 
endian_swap(CHAR * pdata, size_t dsize, size_t nelements)
{
  UINT4 i, j, indx;
  CHAR tempbyte;

  if (dsize <= 1) return;

  for (i=0; i<nelements; i++)
    {
      indx = dsize;
      for (j=0; j<dsize/2; j++)
	{
	  tempbyte = pdata[j];
	  indx = indx - 1;
	  pdata[j] = pdata[indx];
	  pdata[indx] = tempbyte;
	}
      
      pdata = pdata + dsize;
    }
  
  return;

} /* endian swap */

/*----------------------------------------------------------------------
 * glob() has been reported to fail under condor, so we use our own
 * function to get a filelist from a directory, using a glob-like pattern.
 *
 * looks pretty ugly with all the #ifdefs for the Microsoft C compiler
 *
 * NOTE: the list of filenames is returned SORTED ALPHABETICALLY !
 *
 *----------------------------------------------------------------------*/
static StringVector *
find_files (const CHAR *globdir)
{
#ifndef _MSC_VER
  DIR *dir;
  struct dirent *entry;
#else
  intptr_t dir;
  struct _finddata_t entry;
#endif
  CHAR *dname, *ptr1;
  CHAR *fpattern;
  size_t dirlen;
  CHAR **filelist = NULL; 
  UINT4 numFiles = 0;
  StringVector *ret = NULL;
  UINT4 j;
  UINT4 namelen;
  CHAR *thisFname;

  /* First we separate the globdir into directory-path and file-pattern */

#ifndef _MSC_VER
#define DIR_SEPARATOR '/'
#else
#define DIR_SEPARATOR '\\'
#endif

  /* any path specified or not ? */
  ptr1 = strrchr (globdir, DIR_SEPARATOR);
  if (ptr1)
    { /* yes, copy directory-path */
      dirlen = (size_t)(ptr1 - globdir) + 1;
      if ( (dname = LALCalloc (1, dirlen)) == NULL) 
	return (NULL);
      strncpy (dname, globdir, dirlen);
      dname[dirlen-1] = '\0';

      ptr1 ++; /* skip dir-separator */
      /* copy the rest as a glob-pattern for matching */
      if ( (fpattern = LALCalloc (1, strlen(ptr1) + 1)) == NULL )
	{
	  LALFree (dname);
	  return (NULL);
	}
      strcpy (fpattern, ptr1);   

    } /* if ptr1 */
  else /* no pathname given, assume "." */
    {
      if ( (dname = LALCalloc(1, 2)) == NULL) 
	return (NULL);
      strcpy (dname, ".");

      if ( (fpattern = LALCalloc(1, strlen(globdir)+1)) == NULL)
	{
	  LALFree (dname);
	  return (NULL);
	}
      strcpy (fpattern, globdir);	/* just file-pattern given */
    } /* if !ptr */
  

#ifndef _MSC_VER
  /* now go through the file-list in this directory */
  if ( (dir = opendir(dname)) == NULL) {
    LALPrintError ("Can't open data-directory `%s`\n", dname);
    LALFree (dname);
    return (NULL);
  }
#else
  if ((ptr1 = (CHAR*)LALMalloc(strlen(dname)+3)) == NULL)
    return(NULL);
  sprintf(ptr1,"%s\\*",dname);  
  dir = _findfirst(ptr1,&entry);
  LALFree(ptr1);
  if (dir == -1) {
    LALPrintError ("Can't find file for pattern `%s`\n", ptr1);
    LALFree (dname);
    return (NULL);
  }
#endif

#ifndef _MSC_VER
  while ( (entry = readdir (dir)) != NULL )
    {
      thisFname = entry->d_name;
#else
  do
    {
      thisFname = entry.name;
#endif
      /* now check if glob-pattern fpattern matches the current filename */
      if ( amatch(thisFname, fpattern) 
	   /* and check if we didnt' match some obvious garbage like "." or ".." : */
	   && strcmp( thisFname, ".") && strcmp( thisFname, "..") )
	{

	  numFiles ++;
	  if ( (filelist = LALRealloc (filelist, numFiles * sizeof(CHAR*))) == NULL) {
	    LALFree (dname);
	    LALFree (fpattern);
	    return (NULL);
	  }

	  namelen = strlen(thisFname) + strlen(dname) + 2 ;

	  if ( (filelist[ numFiles - 1 ] = LALCalloc (1, namelen)) == NULL) {
	    for (j=0; j < numFiles; j++)
	      LALFree (filelist[j]);
	    LALFree (filelist);
	    LALFree (dname);
	    LALFree (fpattern);
	    return (NULL);
	  }

	  sprintf(filelist[numFiles-1], "%s%c%s", dname, DIR_SEPARATOR, thisFname);

	} /* if filename matched pattern */

    } /* while more directory entries */
#ifdef _MSC_VER
  while ( _findnext (dir,&entry) == 0 );
#endif

#ifndef _MSC_VER
  closedir (dir);
#else
  _findclose(dir);
#endif

  LALFree (dname);
  LALFree (fpattern);

  /* ok, did we find anything? */
  if (numFiles == 0)
    return (NULL);

  if ( (ret = LALCalloc (1, sizeof (StringVector) )) == NULL) 
    {
      for (j=0; j<numFiles; j++)
	LALFree (filelist[j]);
      LALFree (filelist);
      return (NULL);
    }

  ret->length = numFiles;
  ret->data = filelist;

  /* sort this alphabetically (in-place) */
  SortStringVector (ret);

  return (ret);
} /* find_files() */


static void
DestroyStringVector (StringVector *strings)
{
  UINT4 i;

  for (i=0; i < strings->length; i++)
    LALFree (strings->data[i]);
  
  LALFree (strings->data);
  LALFree (strings);

} /* DestroyStringVector () */

/* comparison function for strings */
 typedef CHAR * charptr;	/* little trick to shut compiler-warnings up */
static int mycomp (const void *p1, const void *p2)
{
  const charptr *s1 = (const charptr *) p1;
  const charptr *s2 = (const charptr *) p2;
  return (strcmp ( *s1, *s2 ) );
}

/* sort string-vector alphabetically */
static void
SortStringVector (StringVector *strings)
{
  qsort ( (void*)(strings->data), (size_t)(strings->length), sizeof(CHAR*), mycomp );
} /* SortStringVector() */


/*======================================================================*/
/*
 * robust glob pattern matcher
 * ozan s. yigit/dec 1994
 * public domain
 *
 * glob patterns:
 *	*	matches zero or more characters
 *	?	matches any single character
 *	[set]	matches any character in the set
 *	[^set]	matches any character NOT in the set
 *		where a set is a group of characters or ranges. a range
 *		is written as two characters seperated with a hyphen: a-z denotes
 *		all characters between a to z inclusive.
 *	[-set]	set matches a literal hypen and any character in the set
 *	[]set]	matches a literal close bracket and any character in the set
 *
 *	char	matches itself except where char is '*' or '?' or '['
 *	\char	matches char, including any pattern character
 *
 * examples:
 *	a*c		ac abc abbc ...
 *	a?c		acc abc aXc ...
 *	a[a-z]c		aac abc acc ...
 *	a[-a-z]c	a-c aac abc ...
 *
 * $Log$
 * Revision 1.40  2005/11/03 12:31:07  reinhard
 * - moved a few more general-purpose SFT-handling functions into SFTutils
 * - some doxygenizing
 *
 * Revision 1.39  2005/09/09 09:48:56  reinhard
 * new function: LALGetSFTheaders() returns an SFTVect containing only the headers!
 *
 * Revision 1.38  2005/09/08 21:21:41  reinhard
 * some cleanup.
 *
 * Revision 1.37  2005/09/08 21:12:35  reinhard
 * corrected authorlist and fully doxygenized documentation.
 *
 * Revision 1.36  2005/06/17 21:49:04  jolien
 * moved glob.c code into SFTfileIO.c ; removed non-LAL functions from SFTfileIO.c
 *
 * Revision 1.2  2005/02/09 11:23:30  reinhard
 * squashed warnings
 *
 * Revision 1.1  2004/09/16 15:50:05  reinhard
 * add public-domain function amatch() that can do glob-like string matching.
 *
 * Revision 1.3  1995/09/14  23:24:23  oz
 * removed boring test/main code.
 *
 * Revision 1.2  94/12/11  10:38:15  oz
 * cset code fixed. it is now robust and interprets all
 * variations of cset [i think] correctly, including [z-a] etc.
 * 
 * Revision 1.1  94/12/08  12:45:23  oz
 * Initial revision
 */

#ifndef NEGATE
#define NEGATE	'^'			/* std cset negation char */
#endif

#define TRUE    1
#define FALSE   0

static int amatch(char *str, char *p);

static int
amatch(char *str, char *p)
{
	int negate;
	int match;
	int c;

	while (*p) {
		if (!*str && *p != '*')
			return FALSE;

		switch (c = *p++) {

		case '*':
			while (*p == '*')
				p++;

			if (!*p)
				return TRUE;

			if (*p != '?' && *p != '[' && *p != '\\')
				while (*str && *p != *str)
					str++;

			while (*str) {
				if (amatch(str, p))
					return TRUE;
				str++;
			}
			return FALSE;

		case '?':
			if (*str)
				break;
			return FALSE;
/*
 * set specification is inclusive, that is [a-z] is a, z and
 * everything in between. this means [z-a] may be interpreted
 * as a set that contains z, a and nothing in between.
 */
		case '[':
			if (*p != NEGATE)
				negate = FALSE;
			else {
				negate = TRUE;
				p++;
			}

			match = FALSE;

			while (!match && (c = *p++)) {
				if (!*p)
					return FALSE;
				if (*p == '-') {	/* c-c */
					if (!*++p)
						return FALSE;
					if (*p != ']') {
						if (*str == c || *str == *p ||
						    (*str > c && *str < *p))
							match = TRUE;
					}
					else {		/* c-] */
						if (*str >= c)
							match = TRUE;
						break;
					}
				}
				else {			/* cc or c] */
					if (c == *str)
						match = TRUE;
					if (*p != ']') {
						if (*p == *str)
							match = TRUE;
					}
					else
						break;
				}
			}

			if (negate == match)
				return FALSE;
/*
 * if there is a match, skip past the cset and continue on
 */
			while (*p && *p != ']')
				p++;
			if (!*p++)	/* oops! */
				return FALSE;
			break;

		case '\\':
			if (*p)
				c = *p++;
		default:
			if (c != *str)
				return FALSE;
			break;

		}
		str++;
	}

	return !*str;
}


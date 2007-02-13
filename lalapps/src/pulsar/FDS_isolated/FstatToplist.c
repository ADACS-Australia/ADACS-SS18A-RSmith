#include "FstatToplist.h"
#include "HeapToplist.h"
#include <lal/StringInput.h> /* for LAL_REAL8_FORMAT etc. */

#include <lal/LALStdio.h>
#include <lal/LogPrintf.h>

#if defined(USE_BOINC) || defined(EAH_BOINC)
#include "filesys.h"
#define fopen boinc_fopen
#define rename boinc_rename
#endif

#include <lal/LogPrintf.h>

RCSID("$Id$");



/* MSC specifics */
#ifdef _MSC_VER

/* errno */
extern int errno;
extern int _doserrno;

/* snprintf */
#define LALSnprintf _snprintf

/* finite */
#include <float.h>
#define finite _finite

#else /* MSC */

/* errno */
#include <errno.h>

/* this is defined in C99 and *should* be in math.h. Long term
   protect this with a HAVE_FINITE */
int finite(double);

#endif  /* MSC */



/* define min macro if not already defined */
#ifndef min
#define min(a,b) ((a)<(b)?(a):(b))
#endif



/* local prototypes */
static void reduce_fstat_toplist_precision(toplist_t *l);
static int print_fstatline_to_str(FstatOutputEntry fline, char* buf, int buflen);


/* ordering function for sorting the list */
static int fstat_toplist_qsort_function(const void *a, const void *b) {
    if      (((const FstatOutputEntry*)a)->Freq  < ((const FstatOutputEntry*)b)->Freq)
	return -1;
    else if (((const FstatOutputEntry*)a)->Freq  > ((const FstatOutputEntry*)b)->Freq)
	return 1;
    else if (((const FstatOutputEntry*)a)->Alpha < ((const FstatOutputEntry*)b)->Alpha)
	return -1;
    else if (((const FstatOutputEntry*)a)->Alpha > ((const FstatOutputEntry*)b)->Alpha)
	return 1;
    else if (((const FstatOutputEntry*)a)->Delta < ((const FstatOutputEntry*)b)->Delta)
	return -1;
    else if (((const FstatOutputEntry*)a)->Delta > ((const FstatOutputEntry*)b)->Delta)
	return 1;
    else if (((const FstatOutputEntry*)a)->f1dot < ((const FstatOutputEntry*)b)->f1dot)
	return -1;
    else if (((const FstatOutputEntry*)a)->f1dot > ((const FstatOutputEntry*)b)->f1dot)
	return 1;
    else
	return 0;
}

/* ordering function defining the toplist */
static int fstat_smaller(const void*a, const void*b) {
  if     (((const FstatOutputEntry*)a)->Fstat < ((const FstatOutputEntry*)b)->Fstat)
    return(1);
  else if(((const FstatOutputEntry*)a)->Fstat > ((const FstatOutputEntry*)b)->Fstat)
    return(-1);
  else
    return(fstat_toplist_qsort_function(a,b));
}


/* creates a toplist with length elements,
   returns -1 on error (usually out of memory), else 0 */
int create_fstat_toplist(toplist_t**tl, UINT8 length) {
  return(create_toplist(tl, length, sizeof(FstatOutputEntry), fstat_smaller));
}


/* frees the space occupied by the toplist */
void free_fstat_toplist(toplist_t**l) {
  free_toplist(l);
}


/* Inserts an element in to the toplist either if there is space left
   or the element is larger than the smallest element in the toplist.
   In the latter case, remove the smallest element from the toplist and
   look for the now smallest one.
   Returns 1 if the element was actually inserted, 0 if not. */
int insert_into_fstat_toplist(toplist_t*tl, FstatOutputEntry elem) {
  if ( !tl )
    return 0;
  else
    return(insert_into_toplist(tl, (void*)&elem));
}


/* (q)sort the toplist according to the sorting function. */
void sort_fstat_toplist(toplist_t*l) {
  qsort_toplist(l,fstat_toplist_qsort_function);
}


/* reads a (created!) toplist from an open filepointer
   returns the number of bytes read,
   -1 if the file contained a syntax error,
   -2 if given an improper toplist */
int read_fstat_toplist_from_fp(toplist_t*l, FILE*fp, UINT4*checksum, UINT4 maxbytes) {
    CHAR line[256];       /* buffer for reading a line */
    UINT4 items, lines;   /* number of items read from a line, linecounter */
    UINT4 len, chars = 0; /* length of a line, total characters read from the file */
    UINT4 i;              /* loop counter */
    CHAR lastchar;        /* last character of a line read, should be newline */
    FstatOutputEntry FstatLine;
    REAL8 epsilon=1e-5;

    /* basic check that the list argument is valid */
    if(!l)
	return -2;

    /* make sure the line buffer is terminated correctly */
    line[sizeof(line)-1]='\0';

    /* init the checksum if given */
    if(checksum)
	*checksum = 0;

    /* set maxbytes to maximum if zero */
    if (maxbytes == 0)
	maxbytes--;

    lines=1;
    while(fgets(line,sizeof(line)-1, fp)) {

	len = strlen(line);
	chars += len;

	if (len==0) {
	  LogPrintf (LOG_CRITICAL, "Line %d is empty.\n", lines);
	    return -1;
	}
	else if (line[len-1] != '\n') {
	  LogPrintf (LOG_CRITICAL, 
		     "Line %d is too long or has no NEWLINE. First %d chars are:\n'%s'\n",
		     lines,sizeof(line)-1, line);
	  return -1;
	}
      
	items = sscanf (line,
			 "%" LAL_REAL8_FORMAT
			" %" LAL_REAL8_FORMAT
			" %" LAL_REAL8_FORMAT
			" %" LAL_REAL8_FORMAT
			" %" LAL_REAL8_FORMAT "%c",
			&FstatLine.Freq,
			&FstatLine.Alpha,
			&FstatLine.Delta,
			&FstatLine.f1dot,
			&FstatLine.Fstat,
			&lastchar);

	/* check the values scanned */
	if (
	    items != 6 ||

	    !finite(FstatLine.Freq)	||
	    !finite(FstatLine.f1dot)	||
	    !finite(FstatLine.Alpha)	||
	    !finite(FstatLine.Delta)	||
	    !finite(FstatLine.Fstat)	||

	    FstatLine.Freq  < 0.0                    ||
	    FstatLine.Alpha <         0.0 - epsilon  ||
	    FstatLine.Alpha >   LAL_TWOPI + epsilon  ||
	    FstatLine.Delta < -0.5*LAL_PI - epsilon  ||
	    FstatLine.Delta >  0.5*LAL_PI + epsilon  ||
            FstatLine.Fstat < 0.0                    ||

	    lastchar != '\n'
	    ) {
	    LogPrintf (LOG_CRITICAL, 
		       "Line %d has invalid values.\n"
		       "First %d chars are:\n"
		       "%s\n"
		       "All fields should be finite\n"
		       "1st and 2nd field should be positive.\n" 
		       "3rd field should lie between 0 and %1.15f.\n" 
		       "4th field should lie between %1.15f and %1.15f.\n",
		       lines, sizeof(line)-1, line,
		       (double)LAL_TWOPI, (double)-LAL_PI/2.0, (double)LAL_PI/2.0);
	    return -1;
        }

	if (checksum)
	    for(i=0;i<len;i++)
		*checksum += line[i];
	
	insert_into_toplist(l, &FstatLine);
	lines++;

	/* NOTE: it *CAN* happen (and on Linux it DOES) that the fully buffered Fstat stream
	 * gets written to the File at program termination.
	 * This does not seem to happen on Mac though, most likely due to different
	 * exit()-calls used (_exit() vs exit() etc.....)
	 *
	 * The bottom-line is: the File-contents CAN legally extend beyond maxbytes,
	 * which is why we'll ensure here that we don't actually read more than 
	 * maxbytes.
	 */
	if ( chars == maxbytes )
	  {
	    LogPrintf (LOG_DEBUG, "Read exactly %d == maxbytes from Fstat-file, that's enough.\n", 
		       chars);
	    break;
	  }
	/* however, if we've read more than maxbytes, something is gone wrong */
	if ( chars > maxbytes )
	  {
	    LogPrintf (LOG_CRITICAL, "Read %d bytes > maxbytes %d from Fstat-file ... corrupted.\n",
		       chars, maxbytes );
	    return -1;
	  }

    } /* while (fgets() ) */

    return 0;

} /* read_fstat_toplist_from_fp() */


/* Prints a Tooplist line to a string buffer.
   Separate function to force consistency of output and reduced precision for sorting */
static int print_fstatline_to_str(FstatOutputEntry fline, char* buf, int buflen) {
      return(LALSnprintf(buf, buflen,
		  /* output precision: choose by following (generous!) significant-digit constraints:
		   * Freq:1e-13 
		   * Alpha,Delta:1e-7 
		   * f1dot:1e-5
		   * F:1e-6 
		   */
		  "%.13g %.7g %.7g %.5g %.6g\n",
		  fline.Freq,
		  fline.Alpha,
		  fline.Delta,
		  fline.f1dot,
		  fline.Fstat));
}


/* writes an FstatOutputEntry line to an open filepointer.
   Returns the number of chars written, -1 if in error
   Updates checksum if given */
int write_fstat_toplist_item_to_fp(FstatOutputEntry fline, FILE*fp, UINT4*checksum) {
    char linebuf[256];
    UINT4 i;

    UINT4 length = print_fstatline_to_str(fline, linebuf, sizeof(linebuf));
    
    if(length>sizeof(linebuf)) {
       return -1;
    }

    if (checksum)
	for(i=0;i<length;i++)
	    *checksum += linebuf[i];

    return(fprintf(fp,"%s",linebuf));
}


/* Reduces the precision of all elements in the toplist which influence the sorting order.
   To be called before sorting and finally writing out the list */
static void reduce_fstatline_precision(void*line) {
  char linebuf[256];
  print_fstatline_to_str((*(FstatOutputEntry*)line), linebuf, sizeof(linebuf));
  sscanf(linebuf,
	 "%" LAL_REAL8_FORMAT
	 " %" LAL_REAL8_FORMAT
	 " %" LAL_REAL8_FORMAT
	 " %" LAL_REAL8_FORMAT
	 "%*s\n",
	 &((*(FstatOutputEntry*)line).Freq),
	 &((*(FstatOutputEntry*)line).Alpha),
	 &((*(FstatOutputEntry*)line).Delta),
	 &((*(FstatOutputEntry*)line).f1dot));
}

static void reduce_fstat_toplist_precision(toplist_t *l) {
  go_through_toplist(l,reduce_fstatline_precision);
}


/* Writes the toplist to an (already open) filepointer
   Returns the number of written charactes
   Returns something <0 on error */
int write_fstat_toplist_to_fp(toplist_t*tl, FILE*fp, UINT4*checksum) {
   UINT8 c=0,i;
   INT8 r;
   if(checksum)
       *checksum = 0;
   for(i=0;i<tl->elems;i++)
     if ((r = write_fstat_toplist_item_to_fp(*((FstatOutputEntry*)(tl->heap[i])), fp, checksum)) < 0) {
       LogPrintf (LOG_CRITICAL, "Failed to write toplistitem to output fp: %d: %s\n",
		  errno,strerror(errno));
#ifdef _MSC_VER
       LogPrintf (LOG_CRITICAL, "Windows system call returned: %d\n", _doserrno);
#endif
      return(r);
     } else
       c += r;
   return(c);
}


/* writes the given toplitst to a temporary file, then renames the
   temporary file to filename. The name of the temporary file is
   derived from the filename by appending ".tmp". Returns the number
   of chars written or -1 if the temp file could not be opened. */
int atomic_write_fstat_toplist_to_file(toplist_t *l, char *filename, UINT4*checksum) {
    char tempname[MAXFILENAMELENGTH];
    INT4 length;
    FILE * fpnew;

    strncpy(tempname,filename,sizeof(tempname)-4);
    strcat(tempname,".tmp");
    fpnew=fopen(tempname, "wb");
    if(!fpnew) {
      LogPrintf (LOG_CRITICAL, "Failed to open temp Fstat file \"%s\" for writing: %d: %s\n",
		 tempname,errno,strerror(errno));
#ifdef _MSC_VER
      LogPrintf (LOG_CRITICAL, "Windows system call returned: %d\n", _doserrno);
#endif
      return -1;
    }
    length = write_fstat_toplist_to_fp(l,fpnew,checksum);
    fclose(fpnew);
    if (length < 0) {
      LogPrintf (LOG_CRITICAL, "Failed to write temp Fstat file \"%s\": %d: %s\n",
		 tempname,errno,strerror(errno));
#ifdef _MSC_VER
      LogPrintf (LOG_CRITICAL, "Windows system call returned: %d\n", _doserrno);
#endif
      return(length);
    }
    if(rename(tempname, filename)) {
      LogPrintf (LOG_CRITICAL, "Failed to rename Fstat file to \"%s\": %d: %s\n",
		 filename,errno,strerror(errno));
#ifdef _MSC_VER
      LogPrintf (LOG_CRITICAL, "Windows system call returned: %d\n", _doserrno);
#endif
      return -1;
    } else
      return length;
}


/* meant for the final writing of the toplist
   - reduces toplist precision
   - sorts the toplist
   - the calls atomic_write_fstat_toplist_to_file() */
int final_write_fstat_toplist_to_file(toplist_t *l, char *filename, UINT4*checksum) {
  reduce_fstat_toplist_precision(l);
  sort_fstat_toplist(l);
  return(atomic_write_fstat_toplist_to_file(l,filename,checksum));
}




/* sets up a FStatCheckpointFile from parameters */
int fstat_cpt_file_create (FStatCheckpointFile **cptf,
			   CHAR  *filename,
			   UINT4 bufsize,
			   UINT4 maxsize,
			   toplist_t*list) {

  /* input sanity checks */
  if ( (cptf == NULL) ||
       (*cptf != NULL) ||
       (list == NULL) ||
       (filename == NULL) ||
       (strlen (filename) == 0) ) {
    LogPrintf (LOG_CRITICAL, "ERROR: error in input parameters (fstat_cpt_file_create)\n");
    return(-1);
  }

  /* allocation */
  *cptf = LALMalloc(sizeof(FStatCheckpointFile));
  if (!(*cptf)) {
    LogPrintf (LOG_CRITICAL, "ERROR: out of memeory (fstat_cpt_file_create)\n");
    return(-1);
  }

  (*cptf)->filename = LALMalloc(strlen(filename)+1);
  if (!((*cptf)->filename)) {
    LogPrintf (LOG_CRITICAL, "ERROR: out of memeory (fstat_cpt_file_create)\n");
    LALFree(*cptf);
    *cptf = NULL;
    return(-1);
  }

  if (bufsize > 0) {
    (*cptf)->buffer = LALMalloc(bufsize);
    if (!((*cptf)->buffer)) {
      LogPrintf (LOG_CRITICAL, "ERROR: out of memeory (fstat_cpt_file_create)\n");
      LALFree(*cptf);
      LALFree((*cptf)->filename);
      *cptf = NULL;
      return(-1);
    }
  }

  /* initializing */
  strncpy((*cptf)->filename,filename,strlen(filename)+1);

  (*cptf)->bytes = 0;
  (*cptf)->bufsize = bufsize;
  (*cptf)->maxsize = maxsize;
  (*cptf)->checksum = 0;
  (*cptf)->fp = NULL;
  (*cptf)->list = list;
  return(0);
}



/* destroys a FStatCheckpointFile structure */
int fstat_cpt_file_destroy (FStatCheckpointFile **cptf) {
  if (!cptf) {
    LogPrintf (LOG_CRITICAL, "ERROR: FStatCheckpointFile is NULL\n");
    return(-1);
  }
  if((*cptf)->filename)
    LALFree((*cptf)->filename);
  if((*cptf)->buffer)
    LALFree((*cptf)->buffer);
  LALFree(*cptf);
  *cptf = NULL;
  return(0);
}



/* opens the file named in the structure (for writing) and attaches an output buffer if specified */
int fstat_cpt_file_open (FStatCheckpointFile *cptf) {
  if (!cptf) {
    LogPrintf (LOG_CRITICAL, "ERROR: FStatCheckpointFile is NULL\n");
    return(-1);
  }
  cptf->fp = fopen(cptf->filename, "rb+");
  if (!(cptf->fp)) {
    LogPrintf (LOG_NORMAL, "ERROR: Couldn't open existing checkpointing toplist file %s\n",cptf->filename);
    cptf->fp = fopen(cptf->filename, "wb+");
  }
  if (!(cptf->fp)) {
    LogPrintf (LOG_CRITICAL, "ERROR: Couldn't open new checkpointing toplist file %s\n",cptf->filename);
    return(-1);
  }
  /* set a buffer large enough that no output is written to disk
   * unless we fflush(). Policy is fully buffered. */
  if (cptf->bufsize > 0)
    setvbuf(cptf->fp, cptf->buffer, _IOFBF, cptf->bufsize);
  return(0);
}



/* flushes the checkpoint file (only useful if buffered) */
int fstat_cpt_file_flush (FStatCheckpointFile *cptf) {
  if (!cptf) {
    LogPrintf (LOG_CRITICAL, "ERROR: FStatCheckpointFile is NULL\n");
    return(-1);
  }
  if (!(cptf->fp)) {
    LogPrintf (LOG_CRITICAL, "ERROR: invalid checkpointing toplist file pointer\n");
    return(-1);
  }
  return(fflush(cptf->fp));
}



/* returns information for checkpointing */
extern int fstat_cpt_file_info (FStatCheckpointFile *cptf,
				CHAR**filename, UINT4*bytes, UINT4*checksum) {
  if (!cptf) {
    LogPrintf (LOG_CRITICAL, "ERROR: FStatCheckpointFile is NULL\n");
    return(-1);
  }
  if (filename)
    *filename = cptf->filename;
  if (bytes)
    *bytes = cptf->bytes;
  if (checksum)
    *checksum = cptf->checksum;
  return(0);
}


/* closes and compacts the file */
int fstat_cpt_file_close(FStatCheckpointFile*cptf) {
  if (!cptf) {
    LogPrintf (LOG_CRITICAL, "ERROR: FStatCheckpointFile is NULL\n");
    return(-1);
  }
  fclose(cptf->fp);
  return(final_write_fstat_toplist_to_file(cptf->list,
					   cptf->filename,
					   &(cptf->checksum)));
}


/* adds an item to the toplist and keeps the file consistent, i.e.
   adds the entry to the file if it was really inserted
   and compacts the file if necessary
   returns 1 if the item was actually inserted, 0 if not,
   -1 in case of an error
 */
int fstat_cpt_file_add (FStatCheckpointFile*cptf, FstatOutputEntry line) {
  int ret, bytes;
  ret = insert_into_toplist(cptf->list, &line);
  if (ret) {
    bytes = write_fstat_toplist_item_to_fp(line, cptf->fp, &(cptf->checksum));
    if (bytes < 0) {
      LogPrintf(LOG_CRITICAL, "Failed to write toplist item to file: %d: %s\n",
		errno,strerror(errno));
      return(-1);
    }
    cptf->bytes += bytes;
    if (cptf->bytes >= cptf->maxsize) {
      LogPrintf(LOG_NORMAL, "Compacting toplist file\n");
      bytes = atomic_write_fstat_toplist_to_file(cptf->list, cptf->filename, &(cptf->checksum));
      if (bytes < 0) {
	LogPrintf(LOG_CRITICAL, "Failed to write toplist to file: %d: %s\n",
		  errno,strerror(errno));
	return(-1);
      }
      cptf->bytes = bytes;
    }
  }
  return(ret);
}


/* reads a written checkpointed toplist back into a toplist
   returns 0 if successful,
   -1 if the file contained a syntax error,
   -2 if given an improper toplist */
int fstat_cpt_file_read (FStatCheckpointFile*cptf, UINT4 checksum_should, UINT4 maxbytes) {
  INT4  bytes;
  UINT4 checksum_read;
  if (!cptf) {
    LogPrintf (LOG_CRITICAL, "ERROR: FStatCheckpointFile is NULL\n");
    return(-1);
  }

  bytes = read_fstat_toplist_from_fp(cptf->list, cptf->fp, &checksum_read, maxbytes);

  LogPrintf (LOG_DEBUG, "DEBUG: read_fstat_toplist_from_fp() returned %d\n", bytes);

  cptf->bytes = 0;
  cptf->checksum = 0;

  if (bytes == -2) {
    LogPrintf (LOG_CRITICAL, "ERROR: invalid toplist\n");
    return(bytes);
  } else if (bytes == -1) {
    LogPrintf (LOG_CRITICAL, "ERROR: format error in toplist\n");
    rewind(cptf->fp);
    clear_toplist(cptf->list);
    return(bytes);
  } else if (checksum_read != checksum_should) {
    LogPrintf (LOG_CRITICAL, "ERROR: checksum error in toplist %d / %d\n",
	       checksum_should, checksum_read);
    rewind(cptf->fp);
    clear_toplist(cptf->list);
    return(bytes);
  }

  cptf->bytes = bytes;
  cptf->checksum = checksum_read;

  return(0);
}

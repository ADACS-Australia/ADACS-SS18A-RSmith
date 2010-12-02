/*
*  Copyright (C) 2007 Bernd Machenschalk, Reinhard Prix, Holger Pletsch
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


#include <stdio.h>
#include <unistd.h>
#include "GCTtoplist.h"
#include "HeapToplist.h"
#include <lal/StringInput.h> /* for LAL_REAL8_FORMAT etc. */

#include <lal/LALConstants.h>
#include <lal/LALStdio.h>
#include <lal/LogPrintf.h>
#include <lalapps.h>

#if defined(USE_BOINC) || defined(EAH_BOINC)
#ifdef _WIN32
/* On MS Windows boinc_rename() is not as atomic as rename()
   on POSIX systems. We therefore use our own implementation
   eah_rename (in win_lib.h) */
#define rename eah_rename
#else  // _WIN32
#include "boinc/filesys.h"
#define rename boinc_rename
#endif // _WIN32
#endif // _BOINC

#include <lal/LogPrintf.h>

RCSID("$Id$");


/* Windows specifics */
#ifdef _WIN32

/* errno */
extern int errno;
extern int _doserrno;

/* snprintf */
#define snprintf _snprintf

/* fsync */
#define fsync _commit
#define fileno _fileno

/* finite */
#include <float.h>
#define finite _finite

#else /* WIN32 */

/* errno */
#include <errno.h>

/* this is defined in C99 and *should* be in math.h. Long term
   protect this with a HAVE_FINITE */
int finite(double);

#endif /* WIN32 */

#ifdef DEBUG_SORTING
static FILE*debugfp = NULL;
#endif

/* define min macro if not already defined */
#ifndef min
#define min(a,b) ((a)<(b)?(a):(b))
#endif

/* maximum number of succesive failures before switching off syncing */
#define SYNC_FAIL_LIMIT 5

/* local prototypes */
static void reduce_gctFStat_toplist_precision(toplist_t *l);
static int _atomic_write_gctFStat_toplist_to_file(toplist_t *l, const char *filename, UINT4*checksum, int write_done);
static int print_gctFStatline_to_str(GCTtopOutputEntry fline, char* buf, int buflen);
static int write_gctFStat_toplist_item_to_fp(GCTtopOutputEntry fline, FILE*fp, UINT4*checksum);

/* ordering function for sorting the list */
static int gctFStat_result_order(const void *a, const void *b) {
#ifdef DEBUG_SORTING
  if(debugfp)
    fprintf(debugfp,"%20lf  %20lf\n%20lf  %20lf\n%20lf  %20lf\n%20lf  %20lf\n\n",
	    ((const GCTtopOutputEntry*)a)->Freq,  ((const GCTtopOutputEntry*)b)->Freq,
	    ((const GCTtopOutputEntry*)a)->Alpha, ((const GCTtopOutputEntry*)b)->Alpha,
	    ((const GCTtopOutputEntry*)a)->Delta, ((const GCTtopOutputEntry*)b)->Delta,
	    ((const GCTtopOutputEntry*)a)->F1dot, ((const GCTtopOutputEntry*)b)->F1dot);
#endif
  if      (((const GCTtopOutputEntry*)a)->Freq  < ((const GCTtopOutputEntry*)b)->Freq)
    return -1;
  else if (((const GCTtopOutputEntry*)a)->Freq  > ((const GCTtopOutputEntry*)b)->Freq)
    return 1;
  else if (((const GCTtopOutputEntry*)a)->Alpha < ((const GCTtopOutputEntry*)b)->Alpha)
    return -1;
  else if (((const GCTtopOutputEntry*)a)->Alpha > ((const GCTtopOutputEntry*)b)->Alpha)
    return 1;
  else if (((const GCTtopOutputEntry*)a)->Delta < ((const GCTtopOutputEntry*)b)->Delta)
    return -1;
  else if (((const GCTtopOutputEntry*)a)->Delta > ((const GCTtopOutputEntry*)b)->Delta)
    return 1;
  else if (((const GCTtopOutputEntry*)a)->F1dot < ((const GCTtopOutputEntry*)b)->F1dot)
    return -1;
  else if (((const GCTtopOutputEntry*)a)->F1dot > ((const GCTtopOutputEntry*)b)->F1dot)
    return 1;
  else
    return 0;
}

/* ordering function defining the toplist: SORT BY sumTwoF */
static int gctFStat_smaller(const void*a, const void*b) {
#ifdef DEBUG_SORTING
  if(debugfp)
    fprintf(debugfp,"%20lf  %20lf\n%20u  %20u\n\n",
	    ((const GCTtopOutputEntry*)a)->sumTwoF,  ((const GCTtopOutputEntry*)b)->sumTwoF,
	    ((const GCTtopOutputEntry*)a)->nc, ((const GCTtopOutputEntry*)b)->nc);
#endif
  if      (((const GCTtopOutputEntry*)a)->sumTwoF < ((const GCTtopOutputEntry*)b)->sumTwoF)
    return 1;
  else if (((const GCTtopOutputEntry*)a)->sumTwoF > ((const GCTtopOutputEntry*)b)->sumTwoF)
    return -1;
  else if (((const GCTtopOutputEntry*)a)->nc < ((const GCTtopOutputEntry*)b)->nc)
    return 1;
  else if (((const GCTtopOutputEntry*)a)->nc > ((const GCTtopOutputEntry*)b)->nc)
    return -1;
  else
    return(gctFStat_result_order(a,b));
}


/* ordering function defining the toplist: SORT BY Numbercount */
static int gctNC_smaller(const void*a, const void*b) {
#ifdef DEBUG_SORTING
  if(debugfp)
    fprintf(debugfp,"%20lf  %20lf\n%20u  %20u\n\n",
	    ((const GCTtopOutputEntry*)a)->sumTwoF,  ((const GCTtopOutputEntry*)b)->sumTwoF,
	    ((const GCTtopOutputEntry*)a)->nc, ((const GCTtopOutputEntry*)b)->nc);
#endif
  if      (((const GCTtopOutputEntry*)a)->nc < ((const GCTtopOutputEntry*)b)->nc)
    return 1;
  else if (((const GCTtopOutputEntry*)a)->nc > ((const GCTtopOutputEntry*)b)->nc)
    return -1;
  else if (((const GCTtopOutputEntry*)a)->sumTwoF < ((const GCTtopOutputEntry*)b)->sumTwoF)
    return 1;
  else if (((const GCTtopOutputEntry*)a)->sumTwoF > ((const GCTtopOutputEntry*)b)->sumTwoF)
    return -1;
  else
    return(gctFStat_result_order(a,b));
}


/* functions for qsort based on the above ordering functions */
static int gctFStat_restore_heap_qsort(const void*a, const void*b) {
  void const* const* pa = (void const* const*)a;
  void const* const* pb = (void const* const*)b;
  return(gctFStat_smaller(*pb,*pa));
}

static int gctFStat_strongest_qsort(const void*a, const void*b) {
  void const* const* pa = (void const* const*)a;
  void const* const* pb = (void const* const*)b;
  return(gctFStat_smaller(*pa,*pb));
}

static int gctFStat_final_qsort(const void*a, const void*b) {
  void const* const* pa = (void const* const*)a;
  void const* const* pb = (void const* const*)b;
  return(gctFStat_result_order(*pa,*pb));
}


/* creates a toplist with length elements,
   returns -1 on error (usually out of memory), else 0 */
int create_gctFStat_toplist(toplist_t**tl, UINT8 length, UINT4 whatToSortBy) {
#ifdef DEBUG_SORTING
  if(!debugfp)
    debugfp=fopen("debug_sort","w");
#endif

  if (whatToSortBy==1) {
    return( create_toplist(tl, length, sizeof(GCTtopOutputEntry), gctNC_smaller) );
  }
  else {
    return( create_toplist(tl, length, sizeof(GCTtopOutputEntry), gctFStat_smaller) );
  }
  
}

/* frees the space occupied by the toplist */
void free_gctFStat_toplist(toplist_t**l) {
  free_toplist(l);
}


/* Inserts an element in to the toplist either if there is space left
   or the element is larger than the smallest element in the toplist.
   In the latter case, remove the smallest element from the toplist and
   look for the now smallest one.
   Returns 1 if the element was actually inserted, 0 if not. */
int insert_into_gctFStat_toplist(toplist_t*tl, GCTtopOutputEntry elem) {
  if ( !tl )
    return 0;
  else
    return(insert_into_toplist(tl, (void*)&elem));
}

/* (q)sort the toplist according to the sorting function. */
void sort_gctFStat_toplist(toplist_t*l) {
  qsort(l->heap,l->elems,sizeof(char*),gctFStat_final_qsort);
}

/* (q)sort the toplist in order of strongest candidates */
void sort_gctFStat_toplist_strongest(toplist_t*l) {
  qsort(l->heap,l->elems,sizeof(char*),gctFStat_strongest_qsort);
}


/* Prints a Toplist line to a string buffer.
   Separate function to assure consistency of output and reduced precision for sorting */
static int print_gctFStatline_to_str(GCTtopOutputEntry fline, char* buf, int buflen) {
  return(snprintf(buf, buflen,
#ifdef EAH_BOINC /* for S5GC1HF Apps use exactly the precision used in the workunit generator
		    (12g for Freq and F1dot) and skygrid file (7f for Alpha & Delta)
		    as discussed with Holger & Reinhard 5.11.2010 */
                     "%.14f %.7f %.7f %.12g %d %.6f\n",
#else
                     "%.14g %.13g %.13g %.13g %d %.6f\n",
#endif
                     fline.Freq,
                     fline.Alpha,
                     fline.Delta,
                     fline.F1dot,
                     fline.nc,
                     fline.sumTwoF));
}



/* writes an GCTtopOutputEntry line to an open filepointer.
   Returns the number of chars written, -1 if in error
   Updates checksum if given */
static int write_gctFStat_toplist_item_to_fp(GCTtopOutputEntry fline, FILE*fp, UINT4*checksum) {
  char linebuf[256];
  UINT4 i;

  UINT4 length = print_gctFStatline_to_str(fline, linebuf, sizeof(linebuf)-1);

  if(length>sizeof(linebuf)-1) {
    return -1;
  }

  if (checksum)
    for(i=0;i<length;i++)
      *checksum += linebuf[i];

  linebuf[sizeof(linebuf)-1] = '\0';

  return(fprintf(fp,"%s",linebuf));
}



/* Reduces the precision of all elements in the toplist which influence the sorting order.
   To be called before sorting and finally writing out the list */
static void reduce_gctFStatline_precision(void*line) {
  char linebuf[256];
  print_gctFStatline_to_str((*(GCTtopOutputEntry*)line), linebuf, sizeof(linebuf));
  sscanf(linebuf,
         "%" LAL_REAL8_FORMAT
         " %" LAL_REAL8_FORMAT
         " %" LAL_REAL8_FORMAT
         " %" LAL_REAL8_FORMAT
         "%*s\n",
         &((*(GCTtopOutputEntry*)line).Freq),
         &((*(GCTtopOutputEntry*)line).Alpha),
         &((*(GCTtopOutputEntry*)line).Delta),
         &((*(GCTtopOutputEntry*)line).F1dot));
}

static void reduce_gctFStat_toplist_precision(toplist_t *l) {
  go_through_toplist(l,reduce_gctFStatline_precision);
}


/* Writes the toplist to an (already open) filepointer
   Returns the number of written charactes
   Returns something <0 on error */
int write_gctFStat_toplist_to_fp(toplist_t*tl, FILE*fp, UINT4*checksum) {
  UINT8 c=0,i;
  INT8 r;
  if(checksum)
    *checksum = 0;
  for(i=0;i<tl->elems;i++)
    if ((r = write_gctFStat_toplist_item_to_fp(*((GCTtopOutputEntry*)(tl->heap[i])), fp, checksum)) < 0) {
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


/* function that does the actual work of atomic_write_gctFStat_toplist_to_file(),
   appending a %DONE marker if specified (not when called from atomic_write_gctFStat_toplist_to_file().
   NOTE that the checksum will be a little wrong when %DOME is appended, as this line is not counted */
static int _atomic_write_gctFStat_toplist_to_file(toplist_t *l, const char *filename, UINT4*checksum, int write_done) {
  char* tempname;
  INT4 length=0;
  FILE * fpnew;
  UINT4 s;
  int ret;

#define TEMP_EXT ".tmp"
  s = strlen(filename)+strlen(TEMP_EXT)+1;
  tempname = (char*)malloc(s);
  if(!tempname) {
    LogPrintf (LOG_CRITICAL, "Could not allocate new filename\n");
    return(-1);
  }
  strncpy(tempname,filename,s);
  strncat(tempname,TEMP_EXT,s);

  fpnew=LALFopen(tempname, "wb");
  if(!fpnew) {
    LogPrintf (LOG_CRITICAL, "Failed to open temp gctFStat file \"%s\" for writing: %d: %s\n",
	       tempname,errno,strerror(errno));
#ifdef _MSC_VER
    LogPrintf (LOG_CRITICAL, "Windows system call returned: %d\n", _doserrno);
#endif
    free(tempname);
    return -1;
  }

  /* when done, write code version and command line as comment in the result file */
  if (write_done) {
    int a;
    CHAR *VCSInfoString;

    /* write the version string */
    if ( (VCSInfoString = XLALGetVersionString(0)) == NULL ) {
      LogPrintf (LOG_CRITICAL, "XLALGetVersionString(0) failed.\n");
      length = -1;
    } else {
      ret = fprintf(fpnew,"%s", VCSInfoString);
      XLALFree(VCSInfoString);
      if (ret < 0)
	length = ret;
      else
	length += ret;
    }

    /* write the command-line */
    if (length >= 0) {
      for(a=0;a<global_argc;a++) {
	ret = fprintf(fpnew,"%%%% argv[%d]: '%s'\n", a, global_argv[a]);
	if (ret < 0) {
	  length = ret;
	  break;
	} else
	  length += ret;
      }
    }
  }

  /* write the actual toplist */
  if (length >= 0) {
    ret = write_gctFStat_toplist_to_fp(l,fpnew,checksum);
      if (ret < 0)
	length = ret;
      else
	length += ret;
  }

  /* write the done marker if told to */
  if ((write_done) && (length >= 0)) {
    ret = fprintf(fpnew,"%%DONE\n");
    if (ret < 0)
      length = ret;
    else
      length += ret;
  }

  fclose(fpnew);

  if (length < 0) {
    LogPrintf (LOG_CRITICAL, "Failed to write temp gctFStat file \"%s\": %d: %s\n",
	       tempname,errno,strerror(errno));
#ifdef _MSC_VER
    LogPrintf (LOG_CRITICAL, "Windows system call returned: %d\n", _doserrno);
#endif
    free(tempname);
    return(length);
  }

  if(rename(tempname, filename)) {
    LogPrintf (LOG_CRITICAL, "Failed to rename gctFStat file to \"%s\": %d: %s\n",
	       filename,errno,strerror(errno));
#ifdef _MSC_VER
    LogPrintf (LOG_CRITICAL, "Windows system call returned: %d\n", _doserrno);
#endif
    free(tempname);
    return -1;
  }

  free(tempname);
  return length;
}


/* New easier checkpointing - simply dump the whole toplist (plus a counter and
   a checksum) into a binary file.
   The heap structure array is hard to dump because it's based on pointers, so it
   is restored after reding the data back in by sorting the list once.
*/

/** log an I/O error, i.e. source code line no., ferror, errno and strerror, and doserrno on Windows, too */
#ifndef __func__
#ifdef __FUNCTION__
#define __func__  __FUNCTION__
#else
#define __func__  ""
#endif
#endif

#ifdef _WIN32
#define LOGIOERROR(mess,filename) \
    LogPrintf(LOG_CRITICAL, "ERROR: %s %s: %s (%s:%d): doserr:%d, ferr:%d, errno:%d: %s\n",\
	      mess,filename,__func__,__FILE__,__LINE__,_doserrno,((fp)?(ferror(fp)):0),errno,strerror(errno))
#else
#define LOGIOERROR(mess,filename) \
    LogPrintf(LOG_CRITICAL, "ERROR: %s %s: %s (%s:%d): ferr:%d, errno:%d: %s\n",\
	      mess,filename,__func__,__FILE__,__LINE__,((fp)?(ferror(fp)):0),errno,strerror(errno))
#endif

/* dumps toplist to a temporary file, then renames the file to filename */
int write_hfs_checkpoint(const char*filename, toplist_t*tl, UINT4 counter, BOOLEAN do_sync) {
#define TMP_EXT ".tmp"
  char*tmpfilename;
  FILE*fp;
  UINT4 len;
  UINT4 checksum;
  static UINT4 sync_fail_counter = 0;

  /* construct temporary filename */
  len = strlen(filename)+strlen(TMP_EXT)+1;
  tmpfilename=LALCalloc(len,sizeof(char));
  if(!tmpfilename){
    LogPrintf(LOG_CRITICAL,"Couldn't allocate tmpfilename\n");
    return(-2);
  }
  strncpy(tmpfilename,filename,len);
  strncat(tmpfilename,TMP_EXT,len);

  /* calculate checksum */
  checksum = 0;
  for(len = 0; len < sizeof(tl->elems); len++)
    checksum += *(((char*)&(tl->elems)) + len);
  for(len = 0; len < (tl->elems * tl->size); len++)
    checksum += *(((char*)tl->data) + len);
  for(len = 0; len < sizeof(counter); len++)
    checksum += *(((char*)&counter) + len);

  /* open tempfile */
  fp=LALFopen(tmpfilename,"wb");
  if(!fp) {
    LOGIOERROR("Couldn't open",tmpfilename);
    LALFree(tmpfilename);
    return(-1);
  }

  /* write number of elements */
  len = fwrite(&(tl->elems), sizeof(tl->elems), 1, fp);
  if(len != 1) {
    LOGIOERROR("Couldn't write elems to", tmpfilename);
    LogPrintf(LOG_CRITICAL,"fwrite() returned %d, length was %d\n",len,1);
    if(fclose(fp))
      LOGIOERROR("In addition: couldn't close", tmpfilename);
      LALFree(tmpfilename);
    return(-1);
  }

  /* write data */
  len = fwrite(tl->data, tl->size, tl->elems, fp);
  if(len != tl->elems) {
    LOGIOERROR("Couldn't write data to", tmpfilename);
    LogPrintf(LOG_CRITICAL,"fwrite() returned %d, length was %d\n", len, tl->elems);
    if(fclose(fp))
      LOGIOERROR("In addition: couldn't close", tmpfilename);
      LALFree(tmpfilename);
    return(-1);
  }

  /* write counter */
  len = fwrite(&counter, sizeof(counter), 1, fp);
  if(len != 1) {
    LOGIOERROR("Couldn't write counter to", tmpfilename);
    LogPrintf(LOG_CRITICAL,"fwrite() returned %d, length was %d\n",len,1);
    if(fclose(fp))
      LOGIOERROR("In addition: couldn't close", tmpfilename);
      LALFree(tmpfilename);
    return(-1);
  }

  /* write checksum */
  len = fwrite(&checksum, sizeof(checksum), 1, fp);
  if(len != 1) {
    LOGIOERROR("Couldn't write checksum to", tmpfilename);
    LogPrintf(LOG_CRITICAL,"fwrite() returned %d, length was %d\n",len,1);
    if(fclose(fp))
      LOGIOERROR("In addition: couldn't close", tmpfilename);
      LALFree(tmpfilename);
    return(-1);
  }

  if ( do_sync && (sync_fail_counter < SYNC_FAIL_LIMIT) ) {
    /* make sure the data ends up on disk */
    if(fsync(fileno(fp))) {
      LOGIOERROR("Couldn't sync", tmpfilename);
      sync_fail_counter++;
      if (sync_fail_counter >= SYNC_FAIL_LIMIT)
        LogPrintf(LOG_NORMAL,"WARNING: syncing disabled\n");
      } else {
        sync_fail_counter = 0;
    }
  }

  /* close tempfile */
  if(fclose(fp)) {
    LOGIOERROR("Couldn't close", tmpfilename);
    LALFree(tmpfilename);
    return(-1);
  }

  /* rename to filename */
  if(rename(tmpfilename,filename)) {
    LOGIOERROR("Couldn't rename\n", tmpfilename);
    LALFree(tmpfilename);
    return(-1);
  }

  /* all went well */
  LALFree(tmpfilename);
  return(0);
}


int read_hfs_checkpoint(const char*filename, toplist_t*tl, UINT4*counter) {
  FILE*fp;
  UINT4 len;
  UINT4 checksum;

  /* counter should be 0 if we couldn't read a checkpoint */
  *counter = 0;

  /* try to open file */
  fp = LALFopen(filename, "rb");
  if(!fp) {
    if(errno == ENOENT) {
      LogPrintf(LOG_NORMAL,"INFO: No checkpoint %s found - starting from scratch\n", filename);
      clear_toplist(tl);
      return(1);
    } else {
      LOGIOERROR("Checkpoint found but couldn't open", filename);
      clear_toplist(tl);
      return(-1);
    }
  }

  /* read number of elements */
  len = fread(&(tl->elems), sizeof(tl->elems), 1, fp);
  if(len != 1) {
    LOGIOERROR("Couldn't read elems from", filename);
    LogPrintf(LOG_CRITICAL,"fread() returned %d, length was %d\n", len, 1);
    if(fclose(fp))
      LOGIOERROR("In addition: couldn't close", filename);
    return(-1);
  }
  /* sanity check */
  if (tl->elems > tl->length) {
    LogPrintf(LOG_CRITICAL,
	      "Number of elements read larger than length of toplist: %d, > %d\n",
	      tl->elems, tl->length);
    if(fclose(fp))
      LOGIOERROR("In addition: couldn't close", filename);
    return(-2);
  }

  /* read data */
  len = fread(tl->data, tl->size, tl->elems, fp);
  if(len != tl->elems) {
    LOGIOERROR("Couldn't read data from", filename);
    LogPrintf(LOG_CRITICAL,"fread() returned %d, length was %d\n", len, tl->elems);
    if(fclose(fp))
      LOGIOERROR("In addition: couldn't close", filename);
    clear_toplist(tl);
    return(-1);
  }

  /* read counter */
  len = fread(counter, sizeof(*counter), 1, fp);
  if(len != 1) {
    LOGIOERROR("Couldn't read counter from", filename);
    LogPrintf(LOG_CRITICAL,"fread() returned %d, length was %d\n", len, 1);
    if(fclose(fp))
      LOGIOERROR("In addition: couldn't close", filename);
    clear_toplist(tl);
    return(-1);
  }

  /* read checksum */
  len = fread(&checksum, sizeof(checksum), 1, fp);
  if(len != 1) {
    LOGIOERROR("Couldn't read checksum to", filename);
    LogPrintf(LOG_CRITICAL,"fread() returned %d, length was %d\n", len, 1);
    if(fclose(fp))
      LOGIOERROR("In addition: couldn't close", filename);
    clear_toplist(tl);
    return(-1);
  }

  /* close file */
  if(fclose(fp)) {
    LOGIOERROR("Couldn't close", filename);
    clear_toplist(tl);
    return(-1);
  }

  /* verify checksum */
  for(len = 0; len < sizeof(tl->elems); len++)
    checksum -= *(((char*)&(tl->elems)) + len);
  for(len = 0; len < (tl->elems * tl->size); len++)
    checksum -= *(((char*)tl->data) + len);
  for(len = 0; len < sizeof(*counter); len++)
    checksum -= *(((char*)counter) + len);
  if(checksum) {
    LogPrintf(LOG_CRITICAL,"Checksum error: %d\n", checksum);
    clear_toplist(tl);
    return(-2);
  }

  /* restore Heap structure by sorting */
  for(len = 0; len < tl->elems; len++)
    tl->heap[len] = tl->data + len * tl->size;

#ifdef DEBUG_SORTING
  _atomic_write_gctFStat_toplist_to_file(tl, "toplist_read_from_checkpoint", NULL, 0);
#endif

  qsort(tl->heap,tl->elems,sizeof(char*),gctFStat_restore_heap_qsort);

#ifdef DEBUG_SORTING
  _atomic_write_gctFStat_toplist_to_file(tl, "toplist_sorted_from_checkpoint", NULL, 0);
#endif

  /* all went well */
  LogPrintf(LOG_DEBUG,"Successfully read checkpoint:%d\n", *counter);

  return(0);
}

#ifdef DEBUG_SORTING
static void dump_heap_order(const toplist_t*tl, const char*name) {
  unsigned int i;
  FILE*fp;
  if((fp=fopen(name,"w"))) {
    for(i = 0; i < tl->elems; i++) {
      fprintf(fp,"%u\n",(unsigned int)((tl->heap[i] - tl->data) / sizeof(GCTtopOutputEntry)));
    }
    fclose(fp);
  }
}

static void sort_gctFStat_toplist_debug(toplist_t*l) {
  if(!debugfp)
    debugfp=fopen("debug_sort","w");
  sort_gctFStat_toplist(l); 
  /*sort_gctFStat_toplist_strongest(l);*/
  if(debugfp) {
    fclose(debugfp);
    debugfp=NULL;
  }
}
#endif

int write_hfs_oputput(const char*filename, toplist_t*tl) {
  /* reduce the precision of the calculated values before doing the sort to
     the precision we will write the result with. This should ensure a sorting
     order that looks right to the validator, too */
  reduce_gctFStat_toplist_precision(tl);
#ifdef DEBUG_SORTING
  dump_heap_order(tl,"heap_before.dump");
  sort_gctFStat_toplist_debug(tl);
  dump_heap_order(tl,"heap_after.dump");
#else
  sort_gctFStat_toplist(tl);
#endif
  return(_atomic_write_gctFStat_toplist_to_file(tl, filename, NULL, 1));
}

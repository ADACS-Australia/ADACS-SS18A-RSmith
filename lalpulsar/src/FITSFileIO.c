//
// Copyright (C) 2016 Karl Wette
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with with program; see the file COPYING. If not, write to the
// Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston,
// MA 02111-1307 USA
//

#include <config.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <float.h>

#if defined(HAVE_LIBCFITSIO)
#include <fitsio.h>
#else
#error CFITSIO library is not available
#endif

// If fffree() is missing, use free() instead
#if !defined(HAVE_FFFREE)
#undef fits_free_memory
#define fits_free_memory(ptr, status) free(ptr)

// If ffree() is present but not declared, declare it
#elif defined(HAVE_DECL_FFFREE) && !HAVE_DECL_FFFREE
int fffree(void *, int *);
#undef fits_free_memory
#define fits_free_memory fffree

#endif // ffree()

#include <lal/FITSFileIO.h>
#include <lal/LALString.h>
#include <lal/StringVector.h>
#include <lal/Date.h>
#include <lal/GSLHelpers.h>

#ifdef __GNUC__
#define UNUSED __attribute__ ((unused))
#else
#define UNUSED
#endif

// Call a CFITSIO function, or print error messages on failure
#define CALL_FITS(function, ...) \
  do { \
    if (function(__VA_ARGS__, &status) != 0) { \
      CHAR CALL_FITS_buf[FLEN_STATUS + FLEN_ERRMSG]; \
      fits_get_errstatus(status, CALL_FITS_buf); \
      XLAL_PRINT_ERROR("%s() failed: %s", #function, CALL_FITS_buf); \
      while (fits_read_errmsg(CALL_FITS_buf) > 0) { \
        XLAL_PRINT_ERROR("%s() error: %s", #function, CALL_FITS_buf); \
      } \
      XLAL_ERROR_FAIL(XLAL_EIO); \
    } \
  } while(0)

// Internal representation of a FITS file opened for reading or writing
struct tagFITSFile {
  fitsfile *ff;				// Pointer to a CFITSIO FITS file representation
  int write;				// True if the file is open for writing (otherwise reading)
  int hdutype;				// Type of current HDU
  CHAR hduname[FLEN_VALUE];		// Name of current HDU
  CHAR hducomment[FLEN_COMMENT];	// Comment for name of current HDU
  struct {				// Parameters of current array
    int naxis;					// Number of dimensions of array
    long naxes[FFIO_MAX];			// Size of dimensions of array
    int bitpix;					// Bits per pixel of array
    int datatype;				// Datatype of array
    size_t size;				// Number of bytes in element of array
  } array;
  struct {				// Parameters of current table
    int tfields;				// Number of columns in table
    intptr_t offset[FFIO_MAX];			// Offset of field in table row record
    CHAR ttype[FFIO_MAX][FLEN_VALUE];		// Names of columns in table
    CHAR tform[FFIO_MAX][FLEN_VALUE];		// Format of columns in table
    int datatype[FFIO_MAX];			// Datatype of columns in table
    LONGLONG nelements[FFIO_MAX];		// Number of elements in columns in table
    LONGLONG nrows;				// Number of rows in table
    LONGLONG irow;				// Index of current row in table
  } table;
};

void XLALFITSFileClose(FITSFile *file)
{
  int UNUSED status = 0;
  if (file != NULL) {
    if (file->ff != NULL) {
      fits_close_file(file->ff, &status);
    }
    XLALFree(file);
  }
}

FITSFile *XLALFITSFileOpenWrite(const CHAR *file_name, const LALVCSInfo *const vcs_list[])
{
  int UNUSED status = 0;
  FITSFile *file = NULL;

  // Check input
  XLAL_CHECK_FAIL(file_name != NULL, XLAL_EFAULT);
  XLAL_CHECK_FAIL(vcs_list != NULL, XLAL_EFAULT);

  // Create FITSFile struct
  file = XLALCalloc(1, sizeof(*file));
  XLAL_CHECK_FAIL(file != NULL, XLAL_ENOMEM);

  // Set FITSFile fields
  file->write = 1;

  // Remove any previous file, otherwise fits_create_diskfile() will fail
  if (remove(file_name) != 0 && errno != ENOENT) {
    XLAL_ERROR_FAIL(XLAL_ESYS, "System function remove('%s') failed: %s (%i)", file_name, strerror(errno), errno);
  }
  errno = 0;

  // Open FITS file for writing
  CALL_FITS(fits_create_diskfile, &file->ff, file_name);

  // By convention, create an empty image for the first HDU,
  // so that the correct FITS header 'SIMPLE = T' is written
  CALL_FITS(fits_create_img, file->ff, SHORT_IMG, 0, NULL);
  file->hdutype = INT_MAX;

  // Write the current system date to the FITS file
  CALL_FITS(fits_write_date, file->ff);

  // Write the VCS information list to the FITS file
  for (size_t i = 0; vcs_list[i] != NULL; ++i) {
    CHAR buf[FLEN_COMMENT];
    snprintf(buf, sizeof(buf), "%s version: %s", vcs_list[i]->name, vcs_list[i]->version);
    CALL_FITS(fits_write_history, file->ff, buf);
    snprintf(buf, sizeof(buf), "%s commit : %s", vcs_list[i]->name, vcs_list[i]->vcsId);
    CALL_FITS(fits_write_history, file->ff, buf);
    snprintf(buf, sizeof(buf), "%s status : %s", vcs_list[i]->name, vcs_list[i]->vcsStatus);
    CALL_FITS(fits_write_history, file->ff, buf);
  }

  return file;

XLAL_FAIL:

  // Delete FITS file and free memory on error
  if (file != NULL) {
    if (file->ff != NULL) {
      fits_delete_file(file->ff, &status);
    }
    XLALFree(file);
  }

  return NULL;

}

FITSFile *XLALFITSFileOpenRead(const CHAR *file_name)
{
  int UNUSED status = 0;
  FITSFile *file = NULL;

  // Check input
  XLAL_CHECK_FAIL(file_name != NULL, XLAL_EFAULT);

  // Create FITSFile struct
  file = XLALCalloc(1, sizeof(*file));
  XLAL_CHECK_FAIL(file != NULL, XLAL_ENOMEM);

  // Set FITSFile fields
  file->write = 0;

  // Open FITS file for reading
  CALL_FITS(fits_open_diskfile, &file->ff, file_name, READONLY);

  // Return FITS file on success
  return file;

  // Close FITS file and free memory on error
XLAL_FAIL:
  if (file != NULL) {
    if (file->ff != NULL) {
      fits_close_file(file->ff, &status);
    }
    XLALFree(file);
  }
  return NULL;

}

int XLALFITSHeaderWriteComment(FITSFile *file, const CHAR *comment)
{
  int UNUSED status = 0;

  // Check input
  XLAL_CHECK_FAIL(file != NULL, XLAL_EFAULT);
  XLAL_CHECK_FAIL(file->write, XLAL_EINVAL, "FITS file is not open for writing");
  XLAL_CHECK_FAIL(comment != NULL, XLAL_EFAULT);

  // Write comment to current header
  CALL_FITS(fits_write_comment, file->ff, comment);

  return XLAL_SUCCESS;

XLAL_FAIL:

  // Delete FITS file on error
  if (file != NULL && file->ff != NULL) {
    fits_delete_file(file->ff, &status);
    file->ff = NULL;
  }

  return XLAL_FAILURE;

}

int XLALFITSHeaderWriteBOOLEAN(FITSFile *file, const CHAR *keyword, const BOOLEAN value, const CHAR *comment)
{
  int UNUSED status = 0;

  // Check input
  XLAL_CHECK_FAIL(file != NULL, XLAL_EFAULT);
  XLAL_CHECK_FAIL(file->write, XLAL_EINVAL, "FITS file is not open for writing");
  XLAL_CHECK_FAIL(keyword != NULL, XLAL_EFAULT);
  XLAL_CHECK_FAIL(comment != NULL, XLAL_EFAULT);

  // Write boolean value to current header
  CALL_FITS(fits_write_key_log, file->ff, keyword, value ? 1 : 0, comment);

  return XLAL_SUCCESS;

XLAL_FAIL:

  // Delete FITS file on error
  if (file != NULL && file->ff != NULL) {
    fits_delete_file(file->ff, &status);
    file->ff = NULL;
  }

  return XLAL_FAILURE;

}

int XLALFITSHeaderReadBOOLEAN(FITSFile *file, const CHAR *keyword, BOOLEAN *value)
{
  int UNUSED status = 0;

  // Check input
  XLAL_CHECK_FAIL(file != NULL, XLAL_EFAULT);
  XLAL_CHECK_FAIL(!file->write, XLAL_EINVAL, "FITS file is not open for reading");
  XLAL_CHECK_FAIL(keyword != NULL, XLAL_EFAULT);
  XLAL_CHECK_FAIL(value != NULL, XLAL_EFAULT);

  // Read boolean value from current header
  int val = 0;
  CHAR comment[FLEN_COMMENT];
  CALL_FITS(fits_read_key_log, file->ff, keyword, &val, comment);
  *value = val ? 1 : 0;

  return XLAL_SUCCESS;

XLAL_FAIL:
  return XLAL_FAILURE;

}

int XLALFITSHeaderWriteINT4(FITSFile *file, const CHAR *keyword, const INT4 value, const CHAR *comment)
{
  int UNUSED status = 0;

  // Check input
  XLAL_CHECK_FAIL(file != NULL, XLAL_EFAULT);
  XLAL_CHECK_FAIL(file->write, XLAL_EINVAL, "FITS file is not open for writing");
  XLAL_CHECK_FAIL(keyword != NULL, XLAL_EFAULT);
  XLAL_CHECK_FAIL(comment != NULL, XLAL_EFAULT);

  // Write 32-bit integer value to current header
  CALL_FITS(fits_write_key_lng, file->ff, keyword, value, comment);

  return XLAL_SUCCESS;

XLAL_FAIL:

  // Delete FITS file on error
  if (file != NULL && file->ff != NULL) {
    fits_delete_file(file->ff, &status);
    file->ff = NULL;
  }

  return XLAL_FAILURE;

}

int XLALFITSHeaderReadINT4(FITSFile *file, const CHAR *keyword, INT4 *value)
{
  int UNUSED status = 0;

  // Check input
  XLAL_CHECK_FAIL(file != NULL, XLAL_EFAULT);
  XLAL_CHECK_FAIL(!file->write, XLAL_EINVAL, "FITS file is not open for reading");
  XLAL_CHECK_FAIL(keyword != NULL, XLAL_EFAULT);
  XLAL_CHECK_FAIL(value != NULL, XLAL_EFAULT);

  // Read 32-bit integer value from current header
  LONGLONG val = 0;
  CHAR comment[FLEN_COMMENT];
  CALL_FITS(fits_read_key_lnglng, file->ff, keyword, &val, comment);
  XLAL_CHECK_FAIL(INT32_MIN <= val && val <= INT32_MAX, XLAL_ERANGE);
  *value = val;

  return XLAL_SUCCESS;

XLAL_FAIL:
  return XLAL_FAILURE;

}

int XLALFITSHeaderWriteINT8(FITSFile *file, const CHAR *keyword, const INT8 value, const CHAR *comment)
{
  int UNUSED status = 0;

  // Check input
  XLAL_CHECK_FAIL(file != NULL, XLAL_EFAULT);
  XLAL_CHECK_FAIL(file->write, XLAL_EINVAL, "FITS file is not open for writing");
  XLAL_CHECK_FAIL(keyword != NULL, XLAL_EFAULT);
  XLAL_CHECK_FAIL(comment != NULL, XLAL_EFAULT);

  // Write 64-bit integer value to current header
  CALL_FITS(fits_write_key_lng, file->ff, keyword, value, comment);

  return XLAL_SUCCESS;

XLAL_FAIL:

  // Delete FITS file on error
  if (file != NULL && file->ff != NULL) {
    fits_delete_file(file->ff, &status);
    file->ff = NULL;
  }

  return XLAL_FAILURE;

}

int XLALFITSHeaderReadINT8(FITSFile *file, const CHAR *keyword, INT8 *value)
{
  int UNUSED status = 0;

  // Check input
  XLAL_CHECK_FAIL(file != NULL, XLAL_EFAULT);
  XLAL_CHECK_FAIL(!file->write, XLAL_EINVAL, "FITS file is not open for reading");
  XLAL_CHECK_FAIL(keyword != NULL, XLAL_EFAULT);
  XLAL_CHECK_FAIL(value != NULL, XLAL_EFAULT);

  // Read 64-bit integer value from current header
  LONGLONG val = 0;
  CHAR comment[FLEN_COMMENT];
  CALL_FITS(fits_read_key_lnglng, file->ff, keyword, &val, comment);
  XLAL_CHECK_FAIL(INT64_MIN <= val && val <= INT64_MAX, XLAL_ERANGE);
  *value = val;

  return XLAL_SUCCESS;

XLAL_FAIL:
  return XLAL_FAILURE;

}

int XLALFITSHeaderWriteREAL4(FITSFile *file, const CHAR *keyword, const REAL4 value, const CHAR *comment)
{
  int UNUSED status = 0;

  // Check input
  XLAL_CHECK_FAIL(file != NULL, XLAL_EFAULT);
  XLAL_CHECK_FAIL(file->write, XLAL_EINVAL, "FITS file is not open for writing");
  XLAL_CHECK_FAIL(keyword != NULL, XLAL_EFAULT);
  XLAL_CHECK_FAIL(comment != NULL, XLAL_EFAULT);

  // Write 32-bit floating-point value to current header
  CALL_FITS(fits_write_key_flt, file->ff, keyword, value, FLT_DIG, comment);

  return XLAL_SUCCESS;

XLAL_FAIL:

  // Delete FITS file on error
  if (file != NULL && file->ff != NULL) {
    fits_delete_file(file->ff, &status);
    file->ff = NULL;
  }

  return XLAL_FAILURE;

}

int XLALFITSHeaderReadREAL4(FITSFile *file, const CHAR *keyword, REAL4 *value)
{
  int UNUSED status = 0;

  // Check input
  XLAL_CHECK_FAIL(file != NULL, XLAL_EFAULT);
  XLAL_CHECK_FAIL(!file->write, XLAL_EINVAL, "FITS file is not open for reading");
  XLAL_CHECK_FAIL(keyword != NULL, XLAL_EFAULT);
  XLAL_CHECK_FAIL(value != NULL, XLAL_EFAULT);

  // Read 32-bit floating-point value from current header
  CHAR comment[FLEN_COMMENT];
  CALL_FITS(fits_read_key_flt, file->ff, keyword, value, comment);

  return XLAL_SUCCESS;

XLAL_FAIL:
  return XLAL_FAILURE;

}

int XLALFITSHeaderWriteREAL8(FITSFile *file, const CHAR *keyword, const REAL8 value, const CHAR *comment)
{
  int UNUSED status = 0;

  // Check input
  XLAL_CHECK_FAIL(file != NULL, XLAL_EFAULT);
  XLAL_CHECK_FAIL(file->write, XLAL_EINVAL, "FITS file is not open for writing");
  XLAL_CHECK_FAIL(keyword != NULL, XLAL_EFAULT);
  XLAL_CHECK_FAIL(comment != NULL, XLAL_EFAULT);

  // Write 64-bit floating-point value to current header
  CALL_FITS(fits_write_key_dbl, file->ff, keyword, value, DBL_DIG, comment);

  return XLAL_SUCCESS;

XLAL_FAIL:

  // Delete FITS file on error
  if (file != NULL && file->ff != NULL) {
    fits_delete_file(file->ff, &status);
    file->ff = NULL;
  }

  return XLAL_FAILURE;

}

int XLALFITSHeaderReadREAL8(FITSFile *file, const CHAR *keyword, REAL8 *value)
{
  int UNUSED status = 0;

  // Check input
  XLAL_CHECK_FAIL(file != NULL, XLAL_EFAULT);
  XLAL_CHECK_FAIL(!file->write, XLAL_EINVAL, "FITS file is not open for reading");
  XLAL_CHECK_FAIL(keyword != NULL, XLAL_EFAULT);
  XLAL_CHECK_FAIL(value != NULL, XLAL_EFAULT);

  // Read 64-bit floating-point value from current header
  CHAR comment[FLEN_COMMENT];
  CALL_FITS(fits_read_key_dbl, file->ff, keyword, value, comment);

  return XLAL_SUCCESS;

XLAL_FAIL:
  return XLAL_FAILURE;

}

int XLALFITSHeaderWriteCOMPLEX8(FITSFile *file, const CHAR *keyword, const COMPLEX8 value, const CHAR *comment)
{
  int UNUSED status = 0;

  // Check input
  XLAL_CHECK_FAIL(file != NULL, XLAL_EFAULT);
  XLAL_CHECK_FAIL(file->write, XLAL_EINVAL, "FITS file is not open for writing");
  XLAL_CHECK_FAIL(keyword != NULL, XLAL_EFAULT);
  XLAL_CHECK_FAIL(comment != NULL, XLAL_EFAULT);

  // Write 64-bit complex floating-point value to current header
  REAL4 val[2] = {crealf(value), cimagf(value)};
  CALL_FITS(fits_write_key_cmp, file->ff, keyword, val, FLT_DIG, comment);

  return XLAL_SUCCESS;

XLAL_FAIL:

  // Delete FITS file on error
  if (file != NULL && file->ff != NULL) {
    fits_delete_file(file->ff, &status);
    file->ff = NULL;
  }

  return XLAL_FAILURE;

}

int XLALFITSHeaderReadCOMPLEX8(FITSFile *file, const CHAR *keyword, COMPLEX8 *value)
{
  int UNUSED status = 0;

  // Check input
  XLAL_CHECK_FAIL(file != NULL, XLAL_EFAULT);
  XLAL_CHECK_FAIL(!file->write, XLAL_EINVAL, "FITS file is not open for reading");
  XLAL_CHECK_FAIL(keyword != NULL, XLAL_EFAULT);
  XLAL_CHECK_FAIL(value != NULL, XLAL_EFAULT);

  // Read 64-bit floating-point value from current header
  CHAR comment[FLEN_COMMENT];
  REAL4 val[2] = {0, 0};
  CALL_FITS(fits_read_key_cmp, file->ff, keyword, val, comment);
  *value = crectf(val[0], val[1]);

  return XLAL_SUCCESS;

XLAL_FAIL:
  return XLAL_FAILURE;

}

int XLALFITSHeaderWriteCOMPLEX16(FITSFile *file, const CHAR *keyword, const COMPLEX16 value, const CHAR *comment)
{
  int UNUSED status = 0;

  // Check input
  XLAL_CHECK_FAIL(file != NULL, XLAL_EFAULT);
  XLAL_CHECK_FAIL(file->write, XLAL_EINVAL, "FITS file is not open for writing");
  XLAL_CHECK_FAIL(keyword != NULL, XLAL_EFAULT);
  XLAL_CHECK_FAIL(comment != NULL, XLAL_EFAULT);

  // Write 128-bit complex floating-point value to current header
  REAL8 val[2] = {creal(value), cimag(value)};
  CALL_FITS(fits_write_key_dblcmp, file->ff, keyword, val, DBL_DIG, comment);

  return XLAL_SUCCESS;

XLAL_FAIL:

  // Delete FITS file on error
  if (file != NULL && file->ff != NULL) {
    fits_delete_file(file->ff, &status);
    file->ff = NULL;
  }

  return XLAL_FAILURE;

}

int XLALFITSHeaderReadCOMPLEX16(FITSFile *file, const CHAR *keyword, COMPLEX16 *value)
{
  int UNUSED status = 0;

  // Check input
  XLAL_CHECK_FAIL(file != NULL, XLAL_EFAULT);
  XLAL_CHECK_FAIL(!file->write, XLAL_EINVAL, "FITS file is not open for reading");
  XLAL_CHECK_FAIL(keyword != NULL, XLAL_EFAULT);
  XLAL_CHECK_FAIL(value != NULL, XLAL_EFAULT);

  // Read 128-bit floating-point value from current header
  CHAR comment[FLEN_COMMENT];
  REAL8 val[2] = {0, 0};
  CALL_FITS(fits_read_key_dblcmp, file->ff, keyword, val, comment);
  *value = crect(val[0], val[1]);

  return XLAL_SUCCESS;

XLAL_FAIL:
  return XLAL_FAILURE;

}

int XLALFITSHeaderWriteString(FITSFile *file, const CHAR *keyword, const CHAR *value, const CHAR *comment)
{
  int UNUSED status = 0;

  // Check input
  XLAL_CHECK_FAIL(file != NULL, XLAL_EFAULT);
  XLAL_CHECK_FAIL(file->write, XLAL_EINVAL, "FITS file is not open for writing");
  XLAL_CHECK_FAIL(keyword != NULL, XLAL_EFAULT);
  XLAL_CHECK_FAIL(value != NULL, XLAL_EFAULT);
  XLAL_CHECK_FAIL(comment != NULL, XLAL_EFAULT);

  // Write string value to current header
  union {
    const CHAR *cc;
    CHAR *c;
  } bad_cast = { .cc = value };
  CALL_FITS(fits_write_key_longstr, file->ff, keyword, bad_cast.c, comment);

  return XLAL_SUCCESS;

XLAL_FAIL:

  // Delete FITS file on error
  if (file != NULL && file->ff != NULL) {
    fits_delete_file(file->ff, &status);
    file->ff = NULL;
  }

  return XLAL_FAILURE;

}

int XLALFITSHeaderReadString(FITSFile *file, const CHAR *keyword, CHAR **value)
{
  int UNUSED status = 0;
  CHAR *val = NULL;

  // Check input
  XLAL_CHECK_FAIL(file != NULL, XLAL_EFAULT);
  XLAL_CHECK_FAIL(!file->write, XLAL_EINVAL, "FITS file is not open for reading");
  XLAL_CHECK_FAIL(keyword != NULL, XLAL_EFAULT);
  XLAL_CHECK_FAIL(value != NULL, XLAL_EFAULT);

  // Read string value from current header
  CHAR comment[FLEN_COMMENT];
  CALL_FITS(fits_read_key_longstr, file->ff, keyword, &val, comment);
  XLAL_CHECK_FAIL((*value = XLALStringDuplicate(val)) != NULL, XLAL_EFUNC);

  if (val != NULL) {
    fits_free_memory(val, &status);
  }
  return XLAL_SUCCESS;

XLAL_FAIL:
  if (val != NULL) {
    fits_free_memory(val, &status);
  }
  return XLAL_FAILURE;

}

int XLALFITSHeaderWriteStringVector(FITSFile *file, const CHAR *keyword, const LALStringVector *values, const CHAR *comment)
{
  int UNUSED status = 0;

  // Check input
  XLAL_CHECK_FAIL(file != NULL, XLAL_EFAULT);
  XLAL_CHECK_FAIL(file->write, XLAL_EINVAL, "FITS file is not open for writing");
  XLAL_CHECK_FAIL(keyword != NULL, XLAL_EFAULT);
  XLAL_CHECK_FAIL(values != NULL, XLAL_EFAULT);
  XLAL_CHECK_FAIL(values->length > 0, XLAL_EFAULT);
  XLAL_CHECK_FAIL(comment != NULL, XLAL_EFAULT);

  // Write string values to current header
  {
    union {
      const CHAR *cc;
      CHAR *c;
    } bad_casts[values->length];
    for (size_t i = 0; i < values->length; ++i) {
      bad_casts[i].cc = comment;
    }
    CALL_FITS(fits_write_keys_str, file->ff, keyword, 1, values->length, values->data, (CHAR **) &bad_casts[0]);
  }

  return XLAL_SUCCESS;

XLAL_FAIL:

  // Delete FITS file on error
  if (file != NULL && file->ff != NULL) {
    fits_delete_file(file->ff, &status);
    file->ff = NULL;
  }

  return XLAL_FAILURE;

}

int XLALFITSHeaderReadStringVector(FITSFile *file, const CHAR *keyword, LALStringVector **values)
{
  int UNUSED status = 0;

  // Check input
  XLAL_CHECK_FAIL(file != NULL, XLAL_EFAULT);
  XLAL_CHECK_FAIL(!file->write, XLAL_EINVAL, "FITS file is not open for reading");
  XLAL_CHECK_FAIL(keyword != NULL, XLAL_EFAULT);
  XLAL_CHECK_FAIL(values != NULL, XLAL_EFAULT);
  XLAL_CHECK_FAIL(*values == NULL, XLAL_EFAULT);

  // Read string values from current header
  CHAR vals[FFIO_MAX][FLEN_VALUE], *vals_ptr[FFIO_MAX];
  for (int i = 0; i < FFIO_MAX; ++i) {
    vals_ptr[i] = vals[i];
  }
  int nfound = 0;
  CALL_FITS(fits_read_keys_str, file->ff, keyword, 1, FFIO_MAX, vals_ptr, &nfound);
  XLAL_CHECK_FAIL(nfound <= FFIO_MAX, XLAL_EIO, "Too many items to read into string vector '%s'", keyword);
  for (int i = 0; i < nfound; ++i) {
    *values = XLALAppendString2Vector(*values, vals[i]);
    XLAL_CHECK_FAIL(*values != NULL, XLAL_EFUNC);
  }

  return XLAL_SUCCESS;

XLAL_FAIL:
  return XLAL_FAILURE;

}


int XLALFITSHeaderWriteGPSTime(FITSFile *file, const CHAR *keyword, const LIGOTimeGPS *value, const CHAR *comment)
{
  int UNUSED status = 0;

  // Check input
  XLAL_CHECK_FAIL(file != NULL, XLAL_EFAULT);
  XLAL_CHECK_FAIL(file->write, XLAL_EINVAL, "FITS file is not open for writing");
  XLAL_CHECK_FAIL(keyword != NULL, XLAL_EFAULT);
  XLAL_CHECK_FAIL(value != NULL, XLAL_EFAULT);
  XLAL_CHECK_FAIL(comment != NULL, XLAL_EFAULT);

  // Ensure GPS time is canonical
  LIGOTimeGPS gps;
  XLAL_CHECK_FAIL(XLALGPSSet(&gps, value->gpsSeconds, value->gpsNanoSeconds) != NULL, XLAL_EFUNC);

  // Write time in UTC format to current header
  struct tm XLAL_INIT_DECL(utc);
  XLAL_CHECK_FAIL(XLALGPSToUTC(&utc, gps.gpsSeconds) != NULL, XLAL_EFUNC);
  utc.tm_year += 1900;
  utc.tm_mon += 1;
  CHAR utc_str[FLEN_VALUE];
  CALL_FITS(fits_time2str, utc.tm_year, utc.tm_mon, utc.tm_mday, utc.tm_hour, utc.tm_min, utc.tm_sec + (gps.gpsNanoSeconds / XLAL_BILLION_REAL8), 9, utc_str);
  CALL_FITS(fits_write_key_str, file->ff, keyword, utc_str, comment);

  // Write comment containing time in GPS seconds/nanoseconds format
  CHAR buf[FLEN_COMMENT];
  snprintf(buf, sizeof(buf), "%s = GPS %" LAL_GPS_FORMAT, keyword, LAL_GPS_PRINT(gps));
  CALL_FITS(fits_write_comment, file->ff, buf);

  return XLAL_SUCCESS;

XLAL_FAIL:

  // Delete FITS file on error
  if (file != NULL && file->ff != NULL) {
    fits_delete_file(file->ff, &status);
    file->ff = NULL;
  }

  return XLAL_FAILURE;

}

int XLALFITSHeaderReadGPSTime(FITSFile *file, const CHAR *keyword, LIGOTimeGPS *value)
{
  int UNUSED status = 0;
  CHAR *utc_str = NULL;

  // Check input
  XLAL_CHECK_FAIL(file != NULL, XLAL_EFAULT);
  XLAL_CHECK_FAIL(!file->write, XLAL_EINVAL, "FITS file is not open for reading");
  XLAL_CHECK_FAIL(keyword != NULL, XLAL_EFAULT);
  XLAL_CHECK_FAIL(value != NULL, XLAL_EFAULT);

  // Read time in UTC format from current header
  XLAL_CHECK_FAIL(XLALFITSHeaderReadString(file, keyword, &utc_str) == XLAL_SUCCESS, XLAL_EFUNC);
  struct tm XLAL_INIT_DECL(utc);
  double sec = 0, int_sec = 0, frac_sec = 0;
  CALL_FITS(fits_str2time, utc_str, &utc.tm_year, &utc.tm_mon, &utc.tm_mday, &utc.tm_hour, &utc.tm_min, &sec);
  frac_sec = modf(sec, &int_sec);
  utc.tm_year -= 1900;
  utc.tm_mon -= 1;
  utc.tm_sec = lrint(int_sec);
  INT4 gps_sec = XLALUTCToGPS(&utc);
  XLAL_CHECK_FAIL(xlalErrno == 0, XLAL_EFUNC);
  INT4 gps_nsec = lrint(frac_sec * XLAL_BILLION_REAL8);
  XLAL_CHECK_FAIL(XLALGPSSet(value, gps_sec, gps_nsec) != NULL, XLAL_EFUNC);

  XLALFree(utc_str);
  return XLAL_SUCCESS;

XLAL_FAIL:
  XLALFree(utc_str);
  return XLAL_FAILURE;

}

int XLALFITSArrayOpenWrite(FITSFile *file, const CHAR *name, const size_t ndim, const size_t dims[], const CHAR *comment)
{
  int UNUSED status = 0;

  // Check input
  XLAL_CHECK_FAIL(file != NULL, XLAL_EFAULT);
  XLAL_CHECK_FAIL(file->write, XLAL_EINVAL, "FITS file is not open for writing");
  XLAL_CHECK_FAIL(name != NULL, XLAL_EFAULT);
  XLAL_CHECK_FAIL(ndim <= FFIO_MAX, XLAL_ESIZE);
  XLAL_CHECK_FAIL(dims != NULL, XLAL_EFAULT);

  // Set current HDU
  file->hdutype = IMAGE_HDU;
  strncpy(file->hduname, name, sizeof(file->hduname) - 1);
  strncpy(file->hducomment, comment, sizeof(file->hducomment) - 1);

  // Set current HDU data
  XLAL_INIT_MEM(file->array);
  file->array.bitpix = INT_MAX;
  file->array.datatype = INT_MAX;

  // Save image dimensions
  file->array.naxis = ndim;
  for (int i = 0; i < file->array.naxis; ++i) {
    XLAL_CHECK_FAIL(dims[i] <= LONG_MAX, XLAL_ESIZE);
    file->array.naxes[i] = dims[i];
  }

  return XLAL_SUCCESS;

XLAL_FAIL:

  // Delete FITS file on error
  if (file != NULL && file->ff != NULL) {
    fits_delete_file(file->ff, &status);
    file->ff = NULL;
  }

  return XLAL_FAILURE;

}

int XLALFITSArrayOpenRead(FITSFile *file, const CHAR *name, size_t *ndim, size_t dims[])
{
  int UNUSED status = 0;

  // Check input
  XLAL_CHECK_FAIL(file != NULL, XLAL_EFAULT);
  XLAL_CHECK_FAIL(!file->write, XLAL_EINVAL, "FITS file is not open for reading");
  XLAL_CHECK_FAIL(name != NULL, XLAL_EFAULT);
  XLAL_CHECK_FAIL(ndim != NULL, XLAL_EFAULT);
  XLAL_CHECK_FAIL(dims != NULL, XLAL_EFAULT);

  // Set current HDU
  file->hdutype = IMAGE_HDU;
  strncpy(file->hduname, name, sizeof(file->hduname) - 1);

  // Set current HDU data
  XLAL_INIT_MEM(file->array);
  file->array.bitpix = INT_MAX;
  file->array.datatype = INT_MAX;

  // Go to HDU with given name
  CALL_FITS(fits_movnam_hdu, file->ff, file->hdutype, file->hduname, 0);

  // Get image dimensions
  CALL_FITS(fits_get_img_dim, file->ff, &file->array.naxis);
  *ndim = file->array.naxis;
  XLAL_CHECK_FAIL(*ndim <= FFIO_MAX, XLAL_ESIZE);
  CALL_FITS(fits_get_img_size, file->ff, file->array.naxis, file->array.naxes);
  for (int i = 0; i < file->array.naxis; ++i) {
    XLAL_CHECK_FAIL(0 < file->array.naxes[i] && ((size_t) file->array.naxes[i]) <= SIZE_MAX, XLAL_ESIZE);
    dims[i] = file->array.naxes[i];
  }

  return XLAL_SUCCESS;

XLAL_FAIL:
  return XLAL_FAILURE;

}

int XLALFITSArrayOpenWrite1(FITSFile *file, const CHAR *name, const size_t dim0, const CHAR *comment)
{
  const size_t dims[1] = { dim0 };
  XLAL_CHECK(XLALFITSArrayOpenWrite(file, name, 1, dims, comment) == XLAL_SUCCESS, XLAL_EFUNC);
  return XLAL_SUCCESS;
}

int XLALFITSArrayOpenRead1(FITSFile *file, const CHAR *name, size_t *dim0)
{
  size_t ndim = 0, dims[FFIO_MAX];
  XLAL_CHECK(XLALFITSArrayOpenRead(file, name, &ndim, dims) == XLAL_SUCCESS, XLAL_EFUNC);
  XLAL_CHECK(ndim == 1, XLAL_EIO);
  if (dim0 != NULL) {
    *dim0 = dims[0];
  }
  return XLAL_SUCCESS;
}

int XLALFITSArrayOpenWrite2(FITSFile *file, const CHAR *name, const size_t dim0, const size_t dim1, const CHAR *comment)
{
  const size_t dims[2] = { dim0, dim1 };
  XLAL_CHECK(XLALFITSArrayOpenWrite(file, name, 2, dims, comment) == XLAL_SUCCESS, XLAL_EFUNC);
  return XLAL_SUCCESS;
}

int XLALFITSArrayOpenRead2(FITSFile *file, const CHAR *name, size_t *dim0, size_t *dim1)
{
  size_t ndim = 0, dims[FFIO_MAX];
  XLAL_CHECK(XLALFITSArrayOpenRead(file, name, &ndim, dims) == XLAL_SUCCESS, XLAL_EFUNC);
  XLAL_CHECK(ndim == 2, XLAL_EIO);
  if (dim0 != NULL) {
    *dim0 = dims[0];
  }
  if (dim1 != NULL) {
    *dim1 = dims[1];
  }
  return XLAL_SUCCESS;
}

static int XLALFITSArrayWrite(FITSFile *file, const size_t idx[], const int bitpix, const int datatype, const void *elem)
{
  int UNUSED status = 0;

  // Check input
  XLAL_CHECK_FAIL(file != NULL, XLAL_EFAULT);
  XLAL_CHECK_FAIL(file->write, XLAL_EINVAL, "FITS file is not open for writing");
  XLAL_CHECK_FAIL(idx != 0, XLAL_EINVAL);
  XLAL_CHECK_FAIL(elem != NULL, XLAL_EFAULT);

  // Check that we are at an array
  XLAL_CHECK_FAIL(file->hdutype == IMAGE_HDU, XLAL_EIO, "Current FITS file HDU is not an array");

  // Check for valid array indexes
  for (int i = 0; i < file->array.naxis; ++i) {
    XLAL_CHECK_FAIL(((long) idx[i]) < file->array.naxes[i], XLAL_EDOM, "Array index #%i out of range (%zu >= %li)", i, idx[i], file->array.naxes[i]);
  }

  // Check for bitpix/datatype consistency
  if (file->array.bitpix == INT_MAX || file->array.datatype == INT_MAX) {
    file->array.bitpix = bitpix;
    file->array.datatype = datatype;

    // Create a new FITS image to store array
    CALL_FITS(fits_create_img, file->ff, bitpix, file->array.naxis, file->array.naxes);
    CALL_FITS(fits_write_key_str, file->ff, "HDUNAME", file->hduname, file->hducomment);

  } else {
    XLAL_CHECK_FAIL(file->array.bitpix == bitpix && file->array.datatype == datatype, XLAL_EINVAL, "Inconsistent use of %s...() functions", __func__);
  }

  // Write array element
  long fpixel[FFIO_MAX];
  for (int i = 0; i < file->array.naxis; ++i) {
    fpixel[i] = 1 + idx[i];
  }
  union {
    const void *cv;
    CHAR *c;
  } bad_cast = { .cv = elem };
  CALL_FITS(fits_write_pix, file->ff, file->array.datatype, fpixel, 1, bad_cast.c);

  return XLAL_SUCCESS;

XLAL_FAIL:

  // Delete FITS file on error
  if (file != NULL && file->ff != NULL) {
    fits_delete_file(file->ff, &status);
    file->ff = NULL;
  }

  return XLAL_FAILURE;

}

static int XLALFITSArrayRead(FITSFile *file, const size_t idx[], const int bitpix, const int datatype, void *elem, void *nulelem)
{
  int UNUSED status = 0;

  // Check input
  XLAL_CHECK_FAIL(file != NULL, XLAL_EFAULT);
  XLAL_CHECK_FAIL(!file->write, XLAL_EINVAL, "FITS file is not open for reading");
  XLAL_CHECK_FAIL(idx != 0, XLAL_EINVAL);
  XLAL_CHECK_FAIL(elem != NULL, XLAL_EFAULT);
  XLAL_CHECK_FAIL(nulelem != NULL, XLAL_EFAULT);

  // Check that we are at an array
  XLAL_CHECK_FAIL(file->hdutype == IMAGE_HDU, XLAL_EIO, "Current FITS file HDU is not an array");

  // Check for valid array indexes
  for (int i = 0; i < file->array.naxis; ++i) {
    XLAL_CHECK_FAIL(((long) idx[i]) < file->array.naxes[i], XLAL_EDOM, "Array index #%i out of range (%zu >= %li)", i, idx[i], file->array.naxes[i]);
  }

  // Check for bitpix/datatype consistency
  if (file->array.bitpix == INT_MAX || file->array.datatype == INT_MAX) {
    file->array.bitpix = bitpix;
    file->array.datatype = datatype;
  } else {
    XLAL_CHECK_FAIL(file->array.bitpix == bitpix && file->array.datatype == datatype, XLAL_EINVAL, "Inconsistent use of %s...() functions", __func__);
  }

  // Read array elememt
  long fpixel[FFIO_MAX];
  for (int i = 0; i < file->array.naxis; ++i) {
    fpixel[i] = 1 + idx[i];
  }
  int anynul = 0;
  CALL_FITS(fits_read_pix, file->ff, datatype, fpixel, 1, nulelem, elem, &anynul);

  return XLAL_SUCCESS;

XLAL_FAIL:
  return XLAL_FAILURE;

}

int XLALFITSArrayWriteUINT2(FITSFile *file, const size_t idx[], const UINT2 elem)
{
  XLAL_CHECK(XLALFITSArrayWrite(file, idx, USHORT_IMG, TUSHORT, &elem) == XLAL_SUCCESS, XLAL_EFUNC);
  return XLAL_SUCCESS;
}

int XLALFITSArrayReadUINT2(FITSFile *file, const size_t idx[], UINT2 *elem)
{
  UINT2 nulelem = 0;
  XLAL_CHECK(XLALFITSArrayRead(file, idx, USHORT_IMG, TUSHORT, elem, &nulelem) == XLAL_SUCCESS, XLAL_EFUNC);
  return XLAL_SUCCESS;
}

int XLALFITSArrayWriteUINT4(FITSFile *file, const size_t idx[], const UINT4 elem)
{
  XLAL_CHECK(XLALFITSArrayWrite(file, idx, ULONG_IMG, TULONG, &elem) == XLAL_SUCCESS, XLAL_EFUNC);
  return XLAL_SUCCESS;
}

int XLALFITSArrayReadUINT4(FITSFile *file, const size_t idx[], UINT4 *elem)
{
  UINT4 nulelem = 0;
  XLAL_CHECK(XLALFITSArrayRead(file, idx, ULONG_IMG, TULONG, elem, &nulelem) == XLAL_SUCCESS, XLAL_EFUNC);
  return XLAL_SUCCESS;
}

int XLALFITSArrayWriteINT2(FITSFile *file, const size_t idx[], const INT2 elem)
{
  XLAL_CHECK(XLALFITSArrayWrite(file, idx, SHORT_IMG, TSHORT, &elem) == XLAL_SUCCESS, XLAL_EFUNC);
  return XLAL_SUCCESS;
}

int XLALFITSArrayReadINT2(FITSFile *file, const size_t idx[], INT2 *elem)
{
  INT2 nulelem = 0;
  XLAL_CHECK(XLALFITSArrayRead(file, idx, SHORT_IMG, TSHORT, elem, &nulelem) == XLAL_SUCCESS, XLAL_EFUNC);
  return XLAL_SUCCESS;
}

int XLALFITSArrayWriteINT4(FITSFile *file, const size_t idx[], const INT4 elem)
{
  XLAL_CHECK(XLALFITSArrayWrite(file, idx, LONG_IMG, TLONG, &elem) == XLAL_SUCCESS, XLAL_EFUNC);
  return XLAL_SUCCESS;
}

int XLALFITSArrayReadINT4(FITSFile *file, const size_t idx[], INT4 *elem)
{
  INT4 nulelem = 0;
  XLAL_CHECK(XLALFITSArrayRead(file, idx, LONG_IMG, TLONG, elem, &nulelem) == XLAL_SUCCESS, XLAL_EFUNC);
  return XLAL_SUCCESS;
}

int XLALFITSArrayWriteREAL4(FITSFile *file, const size_t idx[], const REAL4 elem)
{
  XLAL_CHECK(XLALFITSArrayWrite(file, idx, FLOAT_IMG, TFLOAT, &elem) == XLAL_SUCCESS, XLAL_EFUNC);
  return XLAL_SUCCESS;
}

int XLALFITSArrayReadREAL4(FITSFile *file, const size_t idx[], REAL4 *elem)
{
  REAL4 nulelem = 0;
  XLAL_CHECK(XLALFITSArrayRead(file, idx, FLOAT_IMG, TFLOAT, elem, &nulelem) == XLAL_SUCCESS, XLAL_EFUNC);
  return XLAL_SUCCESS;
}

int XLALFITSArrayWriteREAL8(FITSFile *file, const size_t idx[], const REAL8 elem)
{
  XLAL_CHECK(XLALFITSArrayWrite(file, idx, DOUBLE_IMG, TDOUBLE, &elem) == XLAL_SUCCESS, XLAL_EFUNC);
  return XLAL_SUCCESS;
}

int XLALFITSArrayReadREAL8(FITSFile *file, const size_t idx[], REAL8 *elem)
{
  REAL8 nulelem = 0;
  XLAL_CHECK(XLALFITSArrayRead(file, idx, DOUBLE_IMG, TDOUBLE, elem, &nulelem) == XLAL_SUCCESS, XLAL_EFUNC);
  return XLAL_SUCCESS;
}

int XLALFITSArrayWriteGSLMatrix(FITSFile *file, const size_t idx[], const gsl_matrix *elems)
{
  int UNUSED status = 0;

  // Check input
  XLAL_CHECK_FAIL(file != NULL, XLAL_EFAULT);
  XLAL_CHECK_FAIL(file->write, XLAL_EINVAL, "FITS file is not open for writing");
  XLAL_CHECK_FAIL(elems != NULL, XLAL_EFAULT);

  // Check that we are at an array of sufficient size
  XLAL_CHECK_FAIL(file->hdutype == IMAGE_HDU, XLAL_EIO, "Current FITS file HDU is not an array");
  XLAL_CHECK_FAIL(file->array.naxis >= 2, XLAL_EINVAL, "Array must have at least 2 dimensions");
  const size_t m = file->array.naxis - 2;
  XLAL_CHECK_FAIL(elems->size1 == (size_t) file->array.naxes[m], XLAL_EINVAL, "Number of 'elems' rows (%zu) does not match array dimension #%zu (%li)", elems->size1, m, file->array.naxes[m]);
  const size_t n = m + 1;
  XLAL_CHECK_FAIL(elems->size2 == (size_t) file->array.naxes[n], XLAL_EINVAL, "Number of 'elems' rows (%zu) does not match array dimension #%zu (%li)", elems->size2, n, file->array.naxes[n]);

  // Copy index vector, if given
  size_t XLAL_INIT_ARRAY_DECL(i, FFIO_MAX);
  if (idx != NULL) {
    memcpy(i, idx, file->array.naxis * sizeof(i[0]));
  }

  // Write GSL matrix elements to last 2 dimensions
  for (i[m] = 0; i[m] < (size_t) file->array.naxes[m]; ++i[m]) {
    for (i[n] = 0; i[n] < (size_t) file->array.naxes[n]; ++i[n]) {
      const REAL8 elem = gsl_matrix_get(elems, i[m], i[n]);
      XLAL_CHECK_FAIL(XLALFITSArrayWriteREAL8(file, i, elem) == XLAL_SUCCESS, XLAL_EFUNC);
    }
  }

  return XLAL_SUCCESS;

XLAL_FAIL:

  // Delete FITS file on error
  if (file != NULL && file->ff != NULL) {
    fits_delete_file(file->ff, &status);
    file->ff = NULL;
  }

  return XLAL_FAILURE;

}

int XLALFITSArrayReadGSLMatrix(FITSFile *file, const size_t idx[], gsl_matrix **elems)
{
  int UNUSED status = 0;

  // Check input
  XLAL_CHECK_FAIL(file != NULL, XLAL_EFAULT);
  XLAL_CHECK_FAIL(!file->write, XLAL_EINVAL, "FITS file is not open for reading");
  XLAL_CHECK_FAIL(elems != NULL, XLAL_EFAULT);
  XLAL_CHECK_FAIL(*elems == NULL, XLAL_EINVAL);

  // Check that we are at an array of sufficient size
  XLAL_CHECK_FAIL(file->hdutype == IMAGE_HDU, XLAL_EIO, "Current FITS file HDU is not an array");
  XLAL_CHECK_FAIL(file->array.naxis >= 2, XLAL_EINVAL, "Array must have at least 2 dimensions");

  // Create GSL matrix
  const size_t m = file->array.naxis - 2;
  const size_t n = m + 1;
  GAMAT(*elems, file->array.naxes[m], file->array.naxes[n]);

  // Copy index vector, if given
  size_t XLAL_INIT_ARRAY_DECL(i, FFIO_MAX);
  if (idx != NULL) {
    memcpy(i, idx, file->array.naxis * sizeof(i[0]));
  }

  // Read GSL matrix elements to last 2 dimensions
  for (i[m] = 0; i[m] < (size_t) file->array.naxes[m]; ++i[m]) {
    for (i[n] = 0; i[n] < (size_t) file->array.naxes[n]; ++i[n]) {
      REAL8 elem = 0;
      XLAL_CHECK_FAIL(XLALFITSArrayReadREAL8(file, i, &elem) == XLAL_SUCCESS, XLAL_EFUNC);
      gsl_matrix_set(*elems, i[m], i[n], elem);
    }
  }

  return XLAL_SUCCESS;

XLAL_FAIL:
  return XLAL_FAILURE;

}

int XLALFITSTableOpenWrite(FITSFile *file, const CHAR *name, const CHAR *comment)
{
  int UNUSED status = 0;

  // Check input
  XLAL_CHECK_FAIL(file != NULL, XLAL_EFAULT);
  XLAL_CHECK_FAIL(file->write, XLAL_EINVAL, "FITS file is not open for writing");
  XLAL_CHECK_FAIL(name != NULL, XLAL_EFAULT);

  // Set current HDU
  file->hdutype = BINARY_TBL;
  strncpy(file->hduname, name, sizeof(file->hduname) - 1);
  strncpy(file->hducomment, comment, sizeof(file->hducomment) - 1);

  // Set current HDU data
  XLAL_INIT_MEM(file->table);

  return XLAL_SUCCESS;

XLAL_FAIL:

  // Delete FITS file on error
  if (file != NULL && file->ff != NULL) {
    fits_delete_file(file->ff, &status);
    file->ff = NULL;
  }

  return XLAL_FAILURE;

}

int XLALFITSTableOpenRead(FITSFile *file, const CHAR *name, UINT8 *nrows)
{
  int UNUSED status = 0;

  // Check input
  XLAL_CHECK_FAIL(file != NULL, XLAL_EFAULT);
  XLAL_CHECK_FAIL(!file->write, XLAL_EINVAL, "FITS file is not open for reading");
  XLAL_CHECK_FAIL(name != NULL, XLAL_EFAULT);

  // Set current HDU
  file->hdutype = BINARY_TBL;
  strncpy(file->hduname, name, sizeof(file->hduname) - 1);

  // Set current HDU data
  XLAL_INIT_MEM(file->table);

  // Go to HDU with given name
  CALL_FITS(fits_movnam_hdu, file->ff, file->hdutype, file->hduname, 0);

  // Get number of table rows
  CALL_FITS(fits_get_num_rowsll, file->ff, &file->table.nrows);
  if (nrows != NULL) {
    *nrows = file->table.nrows;
  }

  return XLAL_SUCCESS;

XLAL_FAIL:
  return XLAL_FAILURE;

}

static int XLALFITSTableColumnAdd(FITSFile *file, const CHAR *col_name, const CHAR *col_suffix, const void *record, const size_t record_size, const void *field, const size_t field_size, const size_t elem_size, const int typechar, const int datatype)
{
  int UNUSED status = 0;

  // Save previous number of table columns
  const int save_tfields = (file != NULL) ? file->table.tfields : 0;

  // Check input
  XLAL_CHECK_FAIL(file != NULL, XLAL_EFAULT);
  XLAL_CHECK_FAIL(col_name != NULL, XLAL_EFAULT);
  XLAL_CHECK_FAIL(col_suffix != NULL, XLAL_EFAULT);
  XLAL_CHECK_FAIL(strlen(col_name) + strlen(col_suffix) < FLEN_VALUE, XLAL_EINVAL, "Column name '%s%s' is too long", col_name, col_suffix);
  XLAL_CHECK_FAIL(record != NULL, XLAL_EFAULT);
  XLAL_CHECK_FAIL(record_size > 0, XLAL_EINVAL);
  XLAL_CHECK_FAIL(field != NULL, XLAL_EFAULT);
  XLAL_CHECK_FAIL(field_size > 0, XLAL_EINVAL);
  XLAL_CHECK_FAIL(((intptr_t) field) >= ((intptr_t) record), XLAL_EINVAL, "Invalid field pointer");
  XLAL_CHECK_FAIL(((intptr_t) field) + field_size <= ((intptr_t) record) + record_size, XLAL_EINVAL, "Invalid field pointer");
  XLAL_CHECK_FAIL(elem_size > 0, XLAL_EINVAL);

  // Check that we are at a table
  XLAL_CHECK_FAIL(file->hdutype == BINARY_TBL, XLAL_EIO, "Current FITS file HDU is not a table");

  // Move to next column
  XLAL_CHECK_FAIL(file->table.tfields <= FFIO_MAX, XLAL_ESIZE);
  const int i = file->table.tfields++;

  // Store field offset
  file->table.offset[i] = ((intptr_t) field) - ((intptr_t) record);

  // Copy column name
  snprintf(file->table.ttype[i], sizeof(file->table.ttype[i]), "%s%s", col_name, col_suffix);

  // Create column format
  snprintf(file->table.tform[i], sizeof(file->table.tform[i]), "%zu%c", field_size / elem_size, typechar);

  // Store column datatype and number of elements
  file->table.datatype[i] = datatype;
  file->table.nelements[i] = (datatype == TSTRING) ? 1 : field_size / elem_size;

  if (!file->write) {

    // Search for and verify existing column name
    CHAR ttype_chk[FLEN_VALUE];
    int i_chk = 0;
    CALL_FITS(fits_get_colname, file->ff, CASEINSEN, file->table.ttype[i], ttype_chk, &i_chk);
    XLAL_CHECK_FAIL(i_chk > 0, XLAL_EIO, "Column '%s' does not exist in FITS file", file->table.ttype[i]);
    XLAL_CHECK_FAIL(i_chk == 1 + i, XLAL_EIO, "Column '%s' is in a diferent position in FITS file", file->table.ttype[i]);
    XLAL_CHECK_FAIL(strcmp(ttype_chk, file->table.ttype[i]) == 0, XLAL_EIO, "Inconsistent column #%i name '%s'; should be '%s'", i, ttype_chk, file->table.ttype[i]);

    // Verify existing column format
    int datatype_chk = 0;
    long repeat_chk = 0, width_chk = 0;
    CALL_FITS(fits_get_eqcoltype, file->ff, 1 + i, &datatype_chk, &repeat_chk, &width_chk);
    XLAL_CHECK_FAIL(datatype_chk == datatype, XLAL_EIO, "Inconsistent column #%i datatype %i; should be %i", i, datatype_chk, datatype);
    const size_t elem_size_chk = (datatype_chk == TSTRING) ? 1 : width_chk;
    const size_t field_size_chk = repeat_chk * elem_size_chk;
    XLAL_CHECK_FAIL(field_size_chk == field_size, XLAL_EIO, "Inconsistent column #%i field size %zu; should be %zu", i, field_size_chk, field_size);
    XLAL_CHECK_FAIL(elem_size_chk == elem_size, XLAL_EIO, "Inconsistent column #%i element size %zu; should be %zu", i, elem_size_chk, elem_size);

  }

  return XLAL_SUCCESS;

XLAL_FAIL:

  // Restore previous number of table columns
  if (file != NULL) {
    file->table.tfields = save_tfields;
  }

  return XLAL_FAILURE;

}

int XLALFITSTableColumnAddBOOLEAN(FITSFile *file, const CHAR *col_name, const void *record, const size_t record_size, const BOOLEAN *field, const size_t field_size)
{
  XLAL_CHECK(XLALFITSTableColumnAdd(file, col_name, "", record, record_size, field, field_size, sizeof(BOOLEAN), 'L', TLOGICAL) == XLAL_SUCCESS, XLAL_EFUNC);
  return XLAL_SUCCESS;
}

int XLALFITSTableColumnAddINT2(FITSFile *file, const CHAR *col_name, const void *record, const size_t record_size, const INT2 *field, const size_t field_size)
{
  XLAL_CHECK(XLALFITSTableColumnAdd(file, col_name, "", record, record_size, field, field_size, sizeof(INT2), 'I', TSHORT) == XLAL_SUCCESS, XLAL_EFUNC);
  return XLAL_SUCCESS;
}

int XLALFITSTableColumnAddINT4(FITSFile *file, const CHAR *col_name, const void *record, const size_t record_size, const INT4 *field, const size_t field_size)
{
  XLAL_CHECK(XLALFITSTableColumnAdd(file, col_name, "", record, record_size, field, field_size, sizeof(INT4), 'J', TINT32BIT) == XLAL_SUCCESS, XLAL_EFUNC);
  return XLAL_SUCCESS;
}

int XLALFITSTableColumnAddREAL4(FITSFile *file, const CHAR *col_name, const void *record, const size_t record_size, const REAL4 *field, const size_t field_size)
{
  XLAL_CHECK(XLALFITSTableColumnAdd(file, col_name, "", record, record_size, field, field_size, sizeof(REAL4), 'E', TFLOAT) == XLAL_SUCCESS, XLAL_EFUNC);
  return XLAL_SUCCESS;
}

int XLALFITSTableColumnAddREAL8(FITSFile *file, const CHAR *col_name, const void *record, const size_t record_size, const REAL8 *field, const size_t field_size)
{
  XLAL_CHECK(XLALFITSTableColumnAdd(file, col_name, "", record, record_size, field, field_size, sizeof(REAL8), 'D', TDOUBLE) == XLAL_SUCCESS, XLAL_EFUNC);
  return XLAL_SUCCESS;
}

int XLALFITSTableColumnAddCOMPLEX8(FITSFile *file, const CHAR *col_name, const void *record, const size_t record_size, const COMPLEX8 *field, const size_t field_size)
{
  XLAL_CHECK(XLALFITSTableColumnAdd(file, col_name, "", record, record_size, field, field_size, sizeof(COMPLEX8), 'C', TCOMPLEX) == XLAL_SUCCESS, XLAL_EFUNC);
  return XLAL_SUCCESS;
}

int XLALFITSTableColumnAddCOMPLEX16(FITSFile *file, const CHAR *col_name, const void *record, const size_t record_size, const COMPLEX16 *field, const size_t field_size)
{
  XLAL_CHECK(XLALFITSTableColumnAdd(file, col_name, "", record, record_size, field, field_size, sizeof(COMPLEX16), 'M', TDBLCOMPLEX) == XLAL_SUCCESS, XLAL_EFUNC);
  return XLAL_SUCCESS;
}

int XLALFITSTableColumnAddCHAR(FITSFile *file, const CHAR *col_name, const void *record, const size_t record_size, const void *field, const size_t field_size)
{
  XLAL_CHECK(XLALFITSTableColumnAdd(file, col_name, "", record, record_size, field, field_size, sizeof(CHAR), 'A', TSTRING) == XLAL_SUCCESS, XLAL_EFUNC);
  return XLAL_SUCCESS;
}

int XLALFITSTableColumnAddGPSTime(FITSFile *file, const CHAR *col_name, const void *record, const size_t record_size, const LIGOTimeGPS *field, const size_t field_size)
{
  XLAL_CHECK(field_size == sizeof(LIGOTimeGPS), XLAL_EINVAL, "Array of GPS times is not supported");
  XLAL_CHECK(XLALFITSTableColumnAdd(file, col_name, "_s", record, record_size, &(field->gpsSeconds), sizeof(field->gpsSeconds), sizeof(field->gpsSeconds), 'J', TINT32BIT) == XLAL_SUCCESS, XLAL_EFUNC);
  XLAL_CHECK(XLALFITSTableColumnAdd(file, col_name, "_ns", record, record_size, &(field->gpsNanoSeconds), sizeof(field->gpsNanoSeconds), sizeof(field->gpsNanoSeconds), 'J', TINT32BIT) == XLAL_SUCCESS, XLAL_EFUNC);
  return XLAL_SUCCESS;
}

int XLALFITSTableWriteRow(FITSFile *file, const void *record)
{
  int UNUSED status = 0;

  // Check input
  XLAL_CHECK_FAIL(file != NULL, XLAL_EFAULT);
  XLAL_CHECK_FAIL(file->write, XLAL_EINVAL, "FITS file is not open for writing");
  XLAL_CHECK_FAIL(record != NULL, XLAL_EFAULT);

  // Check that we are at a table
  XLAL_CHECK_FAIL(file->hdutype == BINARY_TBL, XLAL_EIO, "Current FITS file HDU is not a table");

  // Create new table if required
  if (file->table.irow == 0) {
    CHAR *ttype_ptr[FFIO_MAX], *tform_ptr[FFIO_MAX];
    for (int i = 0; i < file->table.tfields; ++i) {
      ttype_ptr[i] = file->table.ttype[i];
      tform_ptr[i] = file->table.tform[i];
    }
    CALL_FITS(fits_create_tbl, file->ff, file->hdutype, 0, file->table.tfields, ttype_ptr, tform_ptr, NULL, NULL);
    CALL_FITS(fits_write_key_str, file->ff, "HDUNAME", file->hduname, file->hducomment);
  }

  // Advance to next row
  ++file->table.irow;

  // Write next table row
  for (int i = 0; i < file->table.tfields; ++i) {
    union {
      const void *cv;
      CHAR *c;
    } bad_cast = { .cv = record };
    void *value = bad_cast.c + file->table.offset[i];
    void *pvalue = (file->table.datatype[i] == TSTRING) ? &value : value;
    CALL_FITS(fits_write_col, file->ff, file->table.datatype[i], 1 + i, file->table.irow, 1, file->table.nelements[i], pvalue);
  }

  return XLAL_SUCCESS;

XLAL_FAIL:

  // Delete FITS file on error
  if (file != NULL && file->ff != NULL) {
    fits_delete_file(file->ff, &status);
    file->ff = NULL;
  }

  return XLAL_FAILURE;

}

int XLALFITSTableReadRow(FITSFile *file, void *record, UINT8 *rem_nrows)
{
  int UNUSED status = 0;

  // Check input
  XLAL_CHECK_FAIL(file != NULL, XLAL_EFAULT);
  XLAL_CHECK_FAIL(!file->write, XLAL_EINVAL, "FITS file is not open for reading");
  XLAL_CHECK_FAIL(record != NULL, XLAL_EFAULT);

  // Check that we are at a table
  XLAL_CHECK_FAIL(file->hdutype == BINARY_TBL, XLAL_EIO, "Current FITS file HDU is not a table");

  // Return if there are no more rows
  if (file->table.irow == file->table.nrows) {
    return XLAL_SUCCESS;
  }

  // Advance to next row, and return number of remaining rows
  ++file->table.irow;
  if (rem_nrows != NULL) {
    *rem_nrows = file->table.nrows - file->table.irow;
  }

  // Read next table row
  for (int i = 0; i < file->table.tfields; ++i) {
    void *value = ((CHAR *) record) + file->table.offset[i];
    void *pvalue = (file->table.datatype[i] == TSTRING) ? &value : value;
    CALL_FITS(fits_read_col, file->ff, file->table.datatype[i], 1 + i, file->table.irow, 1, file->table.nelements[i], NULL, pvalue, NULL);
  }

  return XLAL_SUCCESS;

XLAL_FAIL:
  return XLAL_FAILURE;

}

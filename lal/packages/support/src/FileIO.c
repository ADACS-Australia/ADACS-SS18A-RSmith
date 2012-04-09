/*
*  Copyright (C) 2007 Bernd Machenschalk, Jolien Creighton, Reinhard Prix
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
   \addtogroup FileIO_h

   \heading{Obsolete LAL Prototypes}
   \code
FILE *LALFopen( const char *path, const char *mode );
int LALFclose( FILE *stream );
   \endcode

\heading{Description}

The routines <tt>LALFopen()</tt> and <tt>LALFclose()</tt> are macro defined to be
the same as the standard C routines <tt>LALFopen()</tt> and <tt>fclose()</tt>.  These
should only be used in test programs.

The routine <tt>LALOpenDataFile()</tt> is used to open a data file for reading.
This routine is also to be used in test programs only.  Unless the data file
is specified with an absolute path (beginning with a <tt>/</tt>), or a specific
path (beginning with a <tt>./</tt> or a <tt>../</tt>), the directory
that the file is in is obtained from the environment variable
\c LAL_DATA_PATH, which must be set at run-time.  (If the environment
variable is not set, the default path is <tt>.</tt> --- i.e., the current
directory.)

\c LAL_DATA_PATH should typically be set to
<tt>/usr/local/share/lal</tt>, or wherever LAL data is installed in your system
(which may be different if you used a <tt>--prefix</tt> argument when
configuring LAL), but when the test suite is run with <tt>make check</tt>, the
variable \c LAL_DATA_PATH is set to the current source directory.  If the
filename (including the directory path) is too long (more than 256
characters), <tt>LALOpenDataFile()</tt> returns \c NULL and sets
\c errno to \c ENAMETOOLONG.

\c LAL_DATA_PATH can be any colon-delimeted list of directories, which
are searched in order (just like the \c PATH environment variable).
An extra colon inserts the default data directory
(\f$\langle\f$prefix\f$\rangle\f$<tt>/share/lal</tt>) into the search path at that
point.  E.g., a leading/trailing colon will look for the default data
directory at the start/end of the list of directories.

It is strongly recommended that <tt>LALOpenDataFile()</tt> be used when writing test code.

*/

#include "config.h"

#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#ifdef HAVE_ZLIB_H
/* can't actually include zlib.h since older versions have broken prototypes */
/* #include <zlib.h> */
#define Z_FULL_FLUSH    3
typedef void * gzFile;
extern gzFile gzopen (const char *path, const char *mode);
extern gzFile gzdopen (int fd, const char *mode);
extern int gzsetparams (gzFile file, int level, int strategy);
extern int gzread (gzFile file, void * buf, unsigned len);
extern int gzwrite (gzFile file, const void * buf, unsigned len);
extern int gzprintf (gzFile file, const char *format, ...);
extern int gzputs (gzFile file, const char *s);
extern char * gzgets (gzFile file, char *buf, int len);
extern int gzputc (gzFile file, int c);
extern int gzgetc (gzFile file);
extern int gzungetc (int c, gzFile file);
extern int gzflush (gzFile file, int flush);
extern long gzseek (gzFile file, long offset, int whence);
extern int gzrewind (gzFile file);
extern long gztell (gzFile file);
extern int gzeof (gzFile file);
extern int gzdirect (gzFile file);
extern int gzclose (gzFile file);
extern const char * gzerror (gzFile file, int *errnum);
extern void gzclearerr (gzFile file);
#define ZLIB_ENABLED
#endif

#include <lal/LALStdlib.h>
#include <lal/LALStdio.h>
#include <lal/FileIO.h>

LALFILE * lalstdin( void )
{
	static LALFILE _lalstdin;
	_lalstdin.fp = (void*)stdin;
	return &_lalstdin;
}
LALFILE * lalstdout( void )
{
	static LALFILE _lalstdout;
	_lalstdout.fp = (void*)stdout;
	return &_lalstdout;
}
LALFILE * lalstderr( void )
{
	static LALFILE _lalstderr;
	_lalstderr.fp = (void*)stderr;
	return &_lalstderr;
}

#define STR( x ) #x
#define XSTR( x ) STR( x )
#define INFOMSG( msg, file ) ( ( lalDebugLevel & LALINFO ) ? \
    LALPrintError( "Info: function LALOpenDataFile, file " __FILE__ ", line " \
      XSTR( __LINE__ ) ", $Id$\n\t%s %s\n", msg, file ) : 0 )
#define ERRORMSG( file ) ( ( lalDebugLevel & LALERROR ) ? \
    LALPrintError( "Info: function LALOpenDataFile, file " __FILE__ ", line " \
      XSTR( __LINE__ ) ", $Id$\n\tCould not open data file %s\n", file ) : 0 )

#ifndef LAL_PREFIX
#define LAL_PREFIX "/usr/local"
#endif



FILE *
LALOpenDataFile( const char *fname )
{
  FILE *fp;
  const char *path;
  char *datapath;	/* locally allocated copy of env-var LAL_DATA_PATH */
  const char *p0; 	/* pointer to current sub-path of datapath*/
  char *p1;		/* pointer to next sub-path */
  char  fdata[265];
  int   n;

  if ( ! fname )
    return NULL;

  if ( *fname == '/' ) /* absolute path is given */
  {
    fp = LALFopen( fname, "r" );
    if ( ! fp )
      ERRORMSG( fname );
    else
      INFOMSG( "Opening data file", fname );
    return fp;
  }

  n = strlen( fname );
  if ( *fname == '.' && n > 0 && ( fname[1] == '/' || ( n > 1 && fname[1] == '.'
          && fname[2] == '/' ) ) ) /* specific path is given */
  {
    fp = LALFopen( fname, "r" );
    if ( ! fp )
      ERRORMSG( fname );
    else
      INFOMSG( "Opening data file", fname );
    return fp;
  }

  path = getenv( "LAL_DATA_PATH" );

  if ( ! path || ! strlen( path ) ) /* path is NULL or empty */
  {
    fp = LALFopen( fname, "r" );
    if ( ! fp )
      ERRORMSG( fname );
    else
      INFOMSG( "Opening data file", fname );
    return fp;
  }

  /* scan through all directories in colon-delmited list of directories */
  if ( (datapath = LALCalloc (strlen(path)+1, 1)) == NULL)	/* we need local copy */
    {
      ERRORMSG( fname );
      return NULL;
    }
  strcpy (datapath, path);
  p0 = datapath;
  do {
    p1 = strchr( p0, ':' ); /* look for additional directories */
    if ( p1 ) /* there are more things in the list */
      *p1++ = 0; /* NUL-terminate current directory */
    if ( ! strlen( p0 ) ) /* this directory is empty */
      p0 = LAL_PREFIX "/share/lal"; /* default data directory */

    n = snprintf( fdata, sizeof(fdata), "%s/%s", p0 ? p0 : ".", fname );
    if ( n > (int) sizeof( fdata ) ) /* data file name too long */
    {
      errno = ENAMETOOLONG;
      LALFree (datapath);
      return NULL;
    }

    INFOMSG( "Looking for file", fdata );
    fp = LALFopen( fdata, "r" );
    if ( fp ) /* we've found it! */
    {
      INFOMSG( "Opening data file", fdata );
      LALFree (datapath);
      return fp;
    }

    p0 = p1;
  }
  while ( p0 );

  LALFree (datapath);
  ERRORMSG( fname );
  return NULL;
}


int XLALFileIsCompressed( const char *path )
{
	FILE *fp;
	unsigned char magic[2] = { 0, 0 };
	size_t c;
	if ( ! ( fp = LALFopen( path, "rb" ) ) )
		XLAL_ERROR( XLAL_EIO );
	c = fread( magic, sizeof(*magic), sizeof(magic)/sizeof(*magic), fp );
  if (c == 0)
    XLAL_ERROR( XLAL_EIO );
	fclose( fp );
	return magic[0] == 0x1f && magic[1] == 0x8b;
}

LALFILE * XLALFileOpenRead( const char *path )
{
	int compression;
	LALFILE *file;
	if ( 0 > (compression = XLALFileIsCompressed(path) ) )
		XLAL_ERROR_NULL( XLAL_EIO );
#	ifndef ZLIB_ENABLED
	if ( compression ) {
		XLALPrintError( "XLAL Error - %s: Cannot read compressed file\n", __func__ );
		XLAL_ERROR_NULL( XLAL_EIO );
	}
#	endif
	if ( ! ( file = XLALMalloc( sizeof(*file ) ) ) )
		XLAL_ERROR_NULL( XLAL_ENOMEM );
	file->compression = compression;
#	ifdef ZLIB_ENABLED
	file->fp = compression ? gzopen( path, "rb" ) : LALFopen( path, "rb" );
#	else
	file->fp = LALFopen( path, "rb" );
#	endif
	if ( ! file->fp ) {
		XLALFree( file );
		XLAL_ERROR_NULL( XLAL_EIO );
	}
	return file;
}

LALFILE * XLALFileOpenAppend( const char *path, int compression )
{
	LALFILE *file;
	if ( ! ( file = XLALMalloc( sizeof(*file ) ) ) )
		XLAL_ERROR_NULL( XLAL_ENOMEM );
#	ifdef ZLIB_ENABLED
	file->fp = compression ? gzopen( path, "a+" ) : LALFopen( path, "a+" );
#	else
	if ( compression ) {
		XLALPrintWarning( "XLAL Warning - %s: Compression not supported\n", __func__ );
		compression = 0;
	}
	file->fp = LALFopen( path, "a+" );
#	endif
	file->compression = compression;
	if ( ! file->fp ) {
		XLALFree( file );
		XLAL_ERROR_NULL( XLAL_EIO );
	}
	return file;
}

LALFILE * XLALFileOpenWrite( const char *path, int compression )
{
	LALFILE *file;
	if ( ! ( file = XLALMalloc( sizeof(*file ) ) ) )
		XLAL_ERROR_NULL( XLAL_ENOMEM );
#	ifdef ZLIB_ENABLED
	file->fp = compression ? gzopen( path, "wb" ) : LALFopen( path, "wb" );
#	else
	if ( compression ) {
		XLALPrintWarning( "XLAL Warning - %s: Compression not supported\n", __func__ );
		compression = 0;
	}
	file->fp = LALFopen( path, "wb" );
#	endif
	file->compression = compression;
	if ( ! file->fp ) {
		XLALFree( file );
		XLAL_ERROR_NULL( XLAL_EIO );
	}
	return file;
}

LALFILE * XLALFileOpen( const char *path, const char *mode )
{
	int compression;
	char *ext;
	switch ( *mode ) {
		case 'r':
			return XLALFileOpenRead( path );
		case 'w':
			/* check if filename ends in .gz */
			ext = strrchr( path, '.' );
			compression = ext ? ! strcmp( ext, ".gz" ) : 0;
			return XLALFileOpenWrite( path, compression );
		default:
			break; /* fall-out */
	}
	/* error if code gets here */
	XLAL_ERROR_NULL( XLAL_EINVAL );
}

int XLALFileClose( LALFILE * file )
{
	/* this routine acts as a no-op if the file is NULL */
	/* this behavior is different from BSD fclose */
	if ( file ) {
		int c;
		if ( ! file->fp )
			XLAL_ERROR( XLAL_EINVAL );
#		ifdef ZLIB_ENABLED
		c = file->compression ? gzclose(file->fp) : fclose(file->fp);
#		else
		c = fclose(file->fp);
#		endif
		if ( c == EOF )
			XLAL_ERROR( XLAL_EIO );
		XLALFree( file );
	}
	return 0;
}

size_t XLALFileRead( void *ptr, size_t size, size_t nobj, LALFILE *file )
{
	size_t c;
	if ( ! file )
		XLAL_ERROR( XLAL_EFAULT );
#	ifdef ZLIB_ENABLED
	c = file->compression ? (size_t)gzread( file->fp, ptr, size * nobj ) : fread( ptr, size, nobj, file->fp );
#	else
	c = fread( ptr, size, nobj, file->fp );
#	endif
	if ( c == (size_t)(-1) || (file->compression == 0 && ferror((FILE*)(file->fp))) )
		XLAL_ERROR( XLAL_EIO );
	return c;
}

size_t XLALFileWrite( const void *ptr, size_t size, size_t nobj, LALFILE *file )
{
	size_t c;
	if ( ! file )
		XLAL_ERROR( XLAL_EFAULT );
#	ifdef ZLIB_ENABLED
	c = file->compression ? (size_t)gzwrite( file->fp, ptr, size * nobj ) : fwrite( ptr, size, nobj, file->fp );
#	else
	c = fwrite( ptr, size, nobj, (FILE*)(file->fp) );
#	endif
	if ( c == 0 || (file->compression == 0 && ferror((FILE*)(file->fp))) )
		XLAL_ERROR( XLAL_EIO );
	return c;
}

int XLALFileGetc( LALFILE *file )
{
	int c;
	if ( ! file )
		XLAL_ERROR( XLAL_EFAULT );
#	ifdef ZLIB_ENABLED
	c = file->compression ? gzgetc(file->fp) : fgetc(file->fp);
#	else
	c = fgetc(file->fp);
#	endif
	return c;
}

int XLALFilePutc( int c, LALFILE *file )
{
	int result;
	if ( ! file )
		XLAL_ERROR( XLAL_EFAULT );
#	ifdef ZLIB_ENABLED
	result = file->compression ? gzputc(file->fp, c) : fputc(c, file->fp);
#	else
	result = fputc(c, (FILE*)(file->fp));
#	endif
	if ( result == -1 )
		XLAL_ERROR( XLAL_EIO );
	return result;
}

char * XLALFileGets( char * s, int size, LALFILE *file )
{
	char *c;
	if ( ! file )
		XLAL_ERROR_NULL( XLAL_EFAULT );
#	ifdef ZLIB_ENABLED
	c = file->compression ? gzgets( file->fp, s, size ) : fgets( s, size, file->fp );
#	else
	c = fgets( s, size, file->fp );
#	endif
	return c;
}

int XLALFilePuts( const char * s, LALFILE *file )
{
	if ( ! file )
		XLAL_ERROR( XLAL_EFAULT );
	if ( 0 > (int)XLALFileWrite( s, sizeof(*s), strlen(s), file ) )
		XLAL_ERROR( XLAL_EFUNC );
	return 0;
}

int XLALFileVPrintf( LALFILE *file, const char *fmt, va_list ap )
{
	char buf[LAL_PRINTF_BUFSIZE];
	int len;
	int c;
	if ( ! file )
		XLAL_ERROR( XLAL_EFAULT );
	len = vsnprintf( buf, LAL_PRINTF_BUFSIZE, fmt, ap );
	if ( len < 0 )
		XLAL_ERROR( XLAL_EFAILED );
	if ( len >= (int)sizeof(buf) ) { /* buffer is too small */
		char *s;
		s = XLALMalloc( len + 1 );
		if ( !s )
			XLAL_ERROR( XLAL_ENOMEM );
		len = vsnprintf( s, len + 1, fmt, ap );
		c = XLALFilePuts( s, file );
		XLALFree( s );
	} else {
		c = XLALFilePuts( buf, file );
	}
	if ( c < 0 )
		XLAL_ERROR( XLAL_EFUNC );
	return len;
}

int XLALFilePrintf( LALFILE *file, const char *fmt, ... )
{
	int c;
	va_list ap;
	va_start( ap, fmt );
	c = XLALFileVPrintf( file, fmt, ap );
	va_end( ap );
	if ( c < 0 )
		XLAL_ERROR( XLAL_EFUNC );
	return c;
}


int XLALFileFlush( LALFILE *file )
{
	int c;
	if ( ! file )
		XLAL_ERROR( XLAL_EFAULT );
#	ifdef ZLIB_ENABLED
	c = file->compression ? gzflush(file->fp, Z_FULL_FLUSH) : fflush(file->fp);
#	else
	c = fflush(file->fp);
#	endif
	if ( c == -1 )
		XLAL_ERROR( XLAL_EIO );
	return c;
}

int XLALFileSeek( LALFILE *file, long offset, int whence )
{
	int c;
	if ( ! file )
		XLAL_ERROR( XLAL_EFAULT );
#	ifdef ZLIB_ENABLED
	if ( file->compression && whence == SEEK_END ) {
		XLALPrintError( "XLAL Error - %s: SEEK_END not supported with compressed files\n", __func__ );
		XLAL_ERROR( XLAL_EINVAL );
	}
	c = file->compression ? gzseek(file->fp, offset, whence) : fseek(file->fp, offset, whence);
#	else
	c = fseek(file->fp, offset, whence);
#	endif
	if ( c == -1 )
		XLAL_ERROR( XLAL_EIO );
	return 0;
}

long XLALFileTell( LALFILE *file )
{
	long c;
	if ( ! file )
		XLAL_ERROR( XLAL_EFAULT );
#	ifdef ZLIB_ENABLED
	c = file->compression ? (long)gztell(file->fp) : ftell(file->fp);
#	else
	c = ftell(file->fp);
#	endif
	if ( c == -1 )
		XLAL_ERROR( XLAL_EIO );
	return 0;
}

void XLALFileRewind( LALFILE *file )
{
	if ( ! file )
		XLAL_ERROR_VOID( XLAL_EFAULT );
#	ifdef ZLIB_ENABLED
	file->compression ? (void)gzrewind(file->fp) : rewind(file->fp);
#	else
	rewind(file->fp);
#	endif
	return;
}

int XLALFileEOF( LALFILE *file )
{
	int c;
	if ( ! file )
		XLAL_ERROR( XLAL_EFAULT );
#	ifdef ZLIB_ENABLED
	c = file->compression ? gzeof(file->fp) : feof((FILE*)(file->fp));
#	else
	c = feof((FILE*)(file->fp));
#	endif
	return c;
}

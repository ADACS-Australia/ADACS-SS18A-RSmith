/*
*  Copyright (C) 2007 Teviet Creighton
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
 * \ingroup MatrixUtils_h
 * \author Creighton, T. D.
 *
 * \brief Computes the inverse and determinant of a matrix.
 *
 * ### Usage ###
 *
 * \code
 * DetInverseTest [-n size | -i infile] [-o outfile] [-v] [-t] [-s] [-d debuglevel]
 * \endcode
 *
 * ### Description ###
 *
 * This program computes the inverse and determinant of a square real
 * matrix using the routines in \ref DetInverse_c.
 * The following option flags are accepted:
 * <ul>
 * <li>[<tt>-n</tt>] Generates a random symmetric
 * \c size\f$\times\f$\c size metric.  If this option is not given,
 * <tt>-n 3</tt> is assumed.  This option (or its default) is
 * \e overridden by the <tt>-i</tt> option, below.</li>
 * <li>[<tt>-i</tt>] Reads a matrix from an input file \c infile
 * using the function <tt>LALSReadVector()</tt>.  If the input file is
 * specified as \c stdin, the data is read from standard input (not a
 * file named \c stdin).</li>
 * <li>[<tt>-o</tt>] Writes the determinant and inverse matrix to an
 * output file \c outfile.  If the output file is specified as
 * \c stdout or \c stderr, the data is written to standard output
 * or standard error (not to files named \c stdout or \c stderr).</li>
 * <li>[<tt>-v</tt>] Specifies that the inverse matrix is to be computed
 * as well the determinant.</li>
 * <li>[<tt>-t</tt>] Specifies that the computation is to be timed;
 * timing information is written to \c stderr.</li>
 * <li>[<tt>-s</tt>] Specifies that the calculations are to be done to
 * single-precision (\c REAL4) rather than double-precision
 * (\c REAL8).</li>
 * <li>[<tt>-d</tt>] Sets the debug level to \c debuglevel.  If not
 * specified, level 0 is assumed.</li>
 * </ul>
 *
 * \par Input format:
 * If an input file or stream is specified, it
 * should consist of \f$N\f$ consecutive lines of \f$N\f$ whitespace-separated
 * numbers, that will be parsed using <tt>LALDReadVector()</tt>, or
 * <tt>LALSReadVector()</tt> if the <tt>-s</tt> option was given.  The data
 * block may be preceded by blank or comment lines (lines containing no
 * parseable numbers), but once a parseable number is found, the rest
 * should follow in a contiguous block.  If the lines contain different
 * numbers of data columns, or if there are fewer lines than columns,
 * then an error is returned; if there are \e more lines than
 * columns, then the extra lines are ignored.
 *
 * \par Output format:
 * If an output file or stream is specified,
 * the input matrix is first written as \f$N\f$ consecutive lines of \f$N\f$
 * whitespace-separated numbers.  This will be followed with a blank
 * line, then a single number representing the determinant.  If the
 * <tt>-v</tt> option is specified, then another blank line will be
 * appended to the output, followed by the inverse matrix written as \f$N\f$
 * lines of \f$N\f$ whitespace-separated numbers.
 *
 */

/** \name Error Codes */
/*@{*/
#define DETINVERSETESTC_ENORM 0		/**< Normal exit */
#define DETINVERSETESTC_ESUB  1		/**< Subroutine failed */
#define DETINVERSETESTC_EARG  2		/**< Error parsing arguments */
#define DETINVERSETESTC_EMEM  3		/**< Out of memory */
#define DETINVERSETESTC_EFILE 4		/**< Could not open file */
#define DETINVERSETESTC_EFMT  5		/**< Bad input file format */
/*@}*/




/** \cond DONT_DOXYGEN */
#define DETINVERSETESTC_MSGENORM "Normal exit"
#define DETINVERSETESTC_MSGESUB  "Subroutine failed"
#define DETINVERSETESTC_MSGEARG  "Error parsing arguments"
#define DETINVERSETESTC_MSGEMEM  "Out of memory"
#define DETINVERSETESTC_MSGEFILE "Could not open file"
#define DETINVERSETESTC_MSGEFMT  "Bad input file format"


#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <lal/LALStdio.h>
#include <lal/LALStdlib.h>
#include <lal/AVFactories.h>
#include <lal/StreamInput.h>
#include <lal/Random.h>
#include <lal/MatrixUtils.h>

/* Default parameter settings. */
#define SIZE 3

/* Usage format string. */
#define USAGE "Usage: %s [-n size | -i infile] [-o outfile]\n"       \
"\t[-v] [-t] [-s] [-d debuglevel]\n"

/* Macros for printing errors and testing subroutines. */
#define ERROR( code, msg, statement )                                \
do                                                                   \
if ( lalDebugLevel & LALERROR )                                      \
{                                                                    \
  LALPrintError( "Error[0] %d: program %s, file %s, line %d, %s\n"   \
		 "        %s %s\n", (code), *argv, __FILE__,         \
		 __LINE__, "$Id$", statement ?              \
                 statement : "", (msg) );                            \
}                                                                    \
while (0)

#define INFO( statement )                                            \
do                                                                   \
if ( lalDebugLevel & LALINFO )                                       \
{                                                                    \
  LALPrintError( "Info[0]: program %s, file %s, line %d, %s\n"       \
		 "        %s\n", *argv, __FILE__, __LINE__,          \
		 "$Id$", (statement) );                     \
}                                                                    \
while (0)

#define WARNING( statement )                                         \
do                                                                   \
if ( lalDebugLevel & LALWARNING )                                    \
{                                                                    \
  LALPrintError( "Warning[0]: program %s, file %s, line %d, %s\n"    \
		 "        %s\n", *argv, __FILE__, __LINE__,          \
		 "$Id$", (statement) );                     \
}                                                                    \
while (0)

#define SUB( func, statusptr )                                       \
do                                                                   \
if ( (func), (statusptr)->statusCode )                               \
{                                                                    \
  ERROR( DETINVERSETESTC_ESUB, DETINVERSETESTC_MSGESUB,              \
         "Function call \"" #func "\" failed:" );                    \
  return DETINVERSETESTC_ESUB;                                       \
}                                                                    \
while (0)

/* A global pointer for debugging. */
#ifndef NDEBUG
char *lalWatch;
#endif


int
main( int argc, char **argv )
{
  static LALStatus stat;       /* status structure */
  int arg;                     /* command-line argument counter */
  UINT4 n = SIZE, i, j;        /* array size and indecies */
  CHAR *infile = NULL;         /* input filename */
  CHAR *outfile = NULL;        /* output filename */
  BOOLEAN timing = 0;          /* whether -t option was given */
  BOOLEAN single = 0;          /* whether -s option was given */
  BOOLEAN invert = 0;          /* whether -v option was given */
  REAL4 sDet = 0.0;            /* determinant if -s option is given */
  REAL8 dDet = 0.0;            /* determinant if -s option isn't given */
  REAL4Array *sMatrix = NULL;  /* matrix if -s option is given */
  REAL8Array *dMatrix = NULL;  /* matrix if -s option is not given */
  REAL4Array *sInverse = NULL; /* inverse if -s option is given */
  REAL8Array *dInverse = NULL; /* inverse if -s option isn't given */
  UINT4Vector dimLength;       /* dimensions used to create matrix */
  UINT4 dims[2];               /* dimLength.data array */
  clock_t start = 0, stop = 0; /* start and stop times for timing */
  FILE *fp = NULL;             /* input/output file pointer */


  /*******************************************************************
   * PARSE ARGUMENTS (arg stores the current position)               *
   *******************************************************************/

  arg = 1;
  while ( arg < argc ) {

    /* Parse matrix size option. */
    if ( !strcmp( argv[arg], "-n" ) ) {
      if ( argc > arg + 1 ) {
	arg++;
	n = atoi( argv[arg++] );
      } else {
	ERROR( DETINVERSETESTC_EARG, DETINVERSETESTC_MSGEARG, 0 );
        LALPrintError( USAGE, *argv );
        return DETINVERSETESTC_EARG;
      }
    }

    /* Parse input option. */
    else if ( !strcmp( argv[arg], "-i" ) ) {
      if ( argc > arg + 1 ) {
	arg++;
	infile = argv[arg++];
      } else {
	ERROR( DETINVERSETESTC_EARG, DETINVERSETESTC_MSGEARG, 0 );
        LALPrintError( USAGE, *argv );
        return DETINVERSETESTC_EARG;
      }
    }

    /* Parse output option. */
    else if ( !strcmp( argv[arg], "-o" ) ) {
      if ( argc > arg + 1 ) {
	arg++;
	outfile = argv[arg++];
      } else {
	ERROR( DETINVERSETESTC_EARG, DETINVERSETESTC_MSGEARG, 0 );
        LALPrintError( USAGE, *argv );
        return DETINVERSETESTC_EARG;
      }
    }

    /* Parse inversion, single-precision, and timing options. */
    else if ( !strcmp( argv[arg], "-v" ) ) {
      arg++;
      invert = 1;
    }
    else if ( !strcmp( argv[arg], "-s" ) ) {
      arg++;
      single = 1;
    }
    else if ( !strcmp( argv[arg], "-t" ) ) {
      arg++;
      timing = 1;
    }

    /* Parse debug level option. */
    else if ( !strcmp( argv[arg], "-d" ) ) {
      if ( argc > arg + 1 ) {
	arg++;
      }else{
	ERROR( DETINVERSETESTC_EARG, DETINVERSETESTC_MSGEARG, 0 );
        LALPrintError( USAGE, *argv );
        return DETINVERSETESTC_EARG;
      }
    }

    /* Check for unrecognized options. */
    else {
      ERROR( DETINVERSETESTC_EARG, DETINVERSETESTC_MSGEARG, 0 );
      LALPrintError( USAGE, *argv );
      return DETINVERSETESTC_EARG;
    }
  } /* End of argument parsing loop. */


  /*******************************************************************
   * GET INPUT MATRIX                                                *
   *******************************************************************/

  /* Set up array creation vector. */
  dimLength.length = 2;
  dimLength.data = dims;

  /* Read input matrix file. */
  if ( infile ) {

    /* Open input file. */
    if ( !strcmp( infile, "stdin" ) )
      fp = stdin;
    else if ( !( fp = fopen( infile, "r" ) ) ) {
      ERROR( DETINVERSETESTC_EFILE, "- " DETINVERSETESTC_MSGEFILE, infile );
      return DETINVERSETESTC_EFILE;
    }

    /* Single-precision mode: */
    if ( single ) {
      REAL4Vector *vec = NULL; /* parsed input line */

      /* Read first non-comment line and create array. */
      do {
	SUB( LALSReadVector( &stat, &vec, fp, 0 ), &stat );
      } while ( ( n = vec->length ) == 0 );
      dimLength.data[0] = dimLength.data[1] = n;
      SUB( LALSCreateArray( &stat, &sMatrix, &dimLength ), &stat );
      for ( j = 0; j < n; j++ )
	sMatrix->data[j] = vec->data[j];
      SUB( LALSDestroyVector( &stat, &vec ), &stat );

      /* Read remaining lines. */
      for ( i = 1; i < n; i++ ) {
	SUB( LALSReadVector( &stat, &vec, fp, 1 ), &stat );
	if ( vec->length != n ) {
	  ERROR( DETINVERSETESTC_EFMT, DETINVERSETESTC_MSGEFMT, 0 );
	  return DETINVERSETESTC_EFMT;
	}
	for ( j = 0; j < n; j++ )
	  sMatrix->data[i*n+j] = vec->data[j];
	SUB( LALSDestroyVector( &stat, &vec ), &stat );
      }
    }

    /* Double-precision mode: */
    else {
      REAL8Vector *vec = NULL; /* parsed input line */

      /* Read first non-comment line and create array. */
      do {
	SUB( LALDReadVector( &stat, &vec, fp, 0 ), &stat );
      } while ( ( n = vec->length ) == 0 );
      dimLength.data[0] = dimLength.data[1] = n;
      SUB( LALDCreateArray( &stat, &dMatrix, &dimLength ), &stat );
      for ( j = 0; j < n; j++ )
	dMatrix->data[j] = vec->data[j];
      SUB( LALDDestroyVector( &stat, &vec ), &stat );

      /* Read remaining lines. */
      for ( i = 1; i < n; i++ ) {
	SUB( LALDReadVector( &stat, &vec, fp, 1 ), &stat );
	if ( vec->length != n ) {
	  ERROR( DETINVERSETESTC_EFMT, DETINVERSETESTC_MSGEFMT, 0 );
	  return DETINVERSETESTC_EFMT;
	}
	for ( j = 0; j < n; j++ )
	  dMatrix->data[i*n+j] = vec->data[j];
	SUB( LALDDestroyVector( &stat, &vec ), &stat );
      }
    }
  }

  /* Generate random matrix. */
  else {
    RandomParams *params = NULL;
    SUB( LALCreateRandomParams( &stat, &params, 0 ), &stat );
    dimLength.data[0] = dimLength.data[1] = n;

    /* Single-precision mode: */
    if ( single ) {
      SUB( LALSCreateArray( &stat, &sMatrix, &dimLength ), &stat );
      for ( i = 0; i < n; i++ ) {
	REAL4 x;
	for ( j = 0; j < n; j++ ) {
	  SUB( LALUniformDeviate( &stat, &x, params ), &stat );
	  sMatrix->data[i*n+j] = 2.0*x - 1.0;
	}
      }
    }

    /* Double-precision mode: */
    else {
      SUB( LALDCreateArray( &stat, &dMatrix, &dimLength ), &stat );
      for ( i = 0; i < n; i++ ) {
	REAL4 x;
	for ( j = 0; j < n; j++ ) {
	  SUB( LALUniformDeviate( &stat, &x, params ), &stat );
	  dMatrix->data[i*n+j] = 2.0*(REAL8)( x ) - 1.0;
	}
      }
    }

    SUB( LALDestroyRandomParams( &stat, &params ), &stat );
  }

  /* Write input matrix to output file. */
  if ( outfile ) {

    /* Open output file. */
    if ( !strcmp( outfile, "stdout" ) )
      fp = stdout;
    else if ( !strcmp( outfile, "stderr" ) )
      fp = stderr;
    else if ( !( fp = fopen( outfile, "r" ) ) ) {
      ERROR( DETINVERSETESTC_EFILE, "- " DETINVERSETESTC_MSGEFILE, outfile );
      return DETINVERSETESTC_EFILE;
    }

    /* Single-precision mode: */
    if ( single ) {
      for ( i = 0; i < n; i++ ) {
	fprintf( fp, "%16.9e", sMatrix->data[i*n] );
	for ( j = 1; j < n; j++ )
	  fprintf( fp, " %16.9e", sMatrix->data[i*n+j] );
	fprintf( fp, "\n" );
      }
    }

    /* Double-precision mode: */
    else {
      for ( i = 0; i < n; i++ ) {
	fprintf( fp, "%25.17e", dMatrix->data[i*n] );
	for ( j = 1; j < n; j++ )
	  fprintf( fp, " %25.17e", dMatrix->data[i*n+j] );
	fprintf( fp, "\n" );
      }
    }
  }


  /*******************************************************************
   * COMPUTE INVERSE                                                 *
   *******************************************************************/

  if ( timing )
    start = clock();

  /* Single-precision mode: */
  if ( single ) {
    if ( invert ) {
      SUB( LALSCreateArray( &stat, &sInverse, &dimLength ), &stat );
      SUB( LALSMatrixInverse( &stat, &sDet, sMatrix, sInverse ),
	   &stat );
    }
  }

  /* Double-precision mode: */
  else {
    if ( invert ) {
      SUB( LALDCreateArray( &stat, &dInverse, &dimLength ), &stat );
      SUB( LALDMatrixInverse( &stat, &dDet, dMatrix, dInverse ),
	   &stat );
    } else {
      SUB( LALDMatrixDeterminant( &stat, &dDet, dMatrix ), &stat );
    }
  }

  if ( timing ) {
    stop = clock();
    fprintf( stderr, "Elapsed time: %.2f s\n",
	     (double)( stop - start )/CLOCKS_PER_SEC );
  }

  /* Write output. */
  if ( outfile ) {

    /* Write determinant. */
    fprintf( fp, "\n" );
    if ( single )
      fprintf( fp, "%16.9e\n", sDet );
    else
      fprintf( fp, "%25.17e\n", dDet );

    /* Write inverse. */
    if ( invert ) {
      fprintf( fp, "\n" );
      if ( single ) {
	for ( i = 0; i < n; i++ ) {
	  fprintf( fp, "%16.9e", sInverse->data[i*n] );
	  for ( j = 1; j < n; j++ )
	    fprintf( fp, " %16.9e", sInverse->data[i*n+j] );
	  fprintf( fp, "\n" );
	}
      } else {
	for ( i = 0; i < n; i++ ) {
	  fprintf( fp, "%25.17e", dInverse->data[i*n] );
	  for ( j = 1; j < n; j++ )
	    fprintf( fp, " %25.17e", dInverse->data[i*n+j] );
	  fprintf( fp, "\n" );
	}
      }
    }

    /* Finished output. */
    if ( fp != stdout && fp != stderr )
      fclose( fp );
  }

  /* Clean up and exit. */
  if ( single ) {
    SUB( LALSDestroyArray( &stat, &sMatrix ), &stat );
    if ( invert ) {
      SUB( LALSDestroyArray( &stat, &sInverse ), &stat );
    }
  } else {
    SUB( LALDDestroyArray( &stat, &dMatrix ), &stat );
    if ( invert ) {
      SUB( LALDDestroyArray( &stat, &dInverse ), &stat );
    }
  }
  LALCheckMemoryLeaks();
  INFO( DETINVERSETESTC_MSGENORM );
  return DETINVERSETESTC_ENORM;
}
/** \endcond */

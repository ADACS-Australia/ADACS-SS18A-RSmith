/**
\file
\ingroup AVFactories_h

\brief A program to test create/destroy vector routines.

\heading{Usage}
\code
VectorFactoriesTest [options]
Options:
  -h         print help
  -q         quiet: run silently
  -v         verbose: print extra information
  -d level   set lalDebugLevel to level
\endcode

\heading{Description}

\heading{Exit codes}
<table><tr><th>Code</th><th>Explanation</th></tr>
<tr><td>0</td><td>Success, normal exit.</td></tr>
<tr><td>1</td><td>Subroutine failed.</td></tr>
</table>

\heading{Algorithm}

\heading{Uses}
\code
lalDebugLevel
\<datatype\>CreateVector()
\<datatype\>ResizeVector()
\<datatype\>DestroyVector()
\endcode

\heading{Notes}

*/
/** \cond DONT_DOXYGEN */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <lal/LALConfig.h>
#include <lal/PrintVector.h>

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

#ifdef HAVE_GETOPT_H
#include <getopt.h>
#endif

#include <lal/LALStdlib.h>
#include <lal/AVFactories.h>

#define CODES_(x) #x
#define CODES(x) CODES_(x)

NRCSID( MAIN, "$Id$" );

extern char *optarg;
extern int   optind;

extern int lalDebugLevel;
int verbose    = 0;

static void
Usage (const char *program, int exitflag);

static void
ParseOptions (int argc, char *argv[]);

static void
TestStatus (LALStatus *status, const char *expectedCodes, int exitCode);

define(`TYPECODE',`Z')
include(`VectorFactoriesTestFunction.m4')

define(`TYPECODE',`C')
include(`VectorFactoriesTestFunction.m4')

define(`TYPECODE',`D')
include(`VectorFactoriesTestFunction.m4')

define(`TYPECODE',`S')
include(`VectorFactoriesTestFunction.m4')

define(`TYPECODE',`I2')
include(`VectorFactoriesTestFunction.m4')

define(`TYPECODE',`I4')
include(`VectorFactoriesTestFunction.m4')

define(`TYPECODE',`I8')
include(`VectorFactoriesTestFunction.m4')

define(`TYPECODE',`U2')
include(`VectorFactoriesTestFunction.m4')

define(`TYPECODE',`U4')
include(`VectorFactoriesTestFunction.m4')

define(`TYPECODE',`U8')
include(`VectorFactoriesTestFunction.m4')

define(`TYPECODE',`CHAR')
include(`VectorFactoriesTestFunction.m4')

define(`TYPECODE',`')
include(`VectorFactoriesTestFunction.m4')

int main( int argc, char *argv[] )
{
  lalDebugLevel = 0;

  ParseOptions( argc, argv );

  VectorFactoriesTest();
  ZVectorFactoriesTest();
  CVectorFactoriesTest();
  DVectorFactoriesTest();
  SVectorFactoriesTest();
  I2VectorFactoriesTest();
  I4VectorFactoriesTest();
  I8VectorFactoriesTest();
  U2VectorFactoriesTest();
  U4VectorFactoriesTest();
  U8VectorFactoriesTest();
  CHARVectorFactoriesTest();

  return 0;
}


/*
 * TestStatus ()
 *
 * Routine to check that the status code status->statusCode agrees with one of
 * the codes specified in the space-delimited string ignored; if not,
 * exit to the system with code exitcode.
 *
 */
static void
TestStatus (LALStatus *status, const char *ignored, int exitcode)
{
  char  str[64];
  char *tok;

  if (verbose)
  {
    REPORTSTATUS (status);
  }

  if (strncpy (str, ignored, sizeof (str)))
  {
    if ((tok = strtok (str, " ")))
    {
      do
      {
        if (status->statusCode == atoi (tok))
        {
          return;
        }
      }
      while ((tok = strtok (NULL, " ")));
    }
    else
    {
      if (status->statusCode == atoi (tok))
      {
        return;
      }
    }
  }

  fprintf (stderr, "\nExiting to system with code %d\n", exitcode);
  exit (exitcode);
}

/*
 * Usage ()
 *
 * Prints a usage message for program program and exits with code exitcode.
 *
 */
static void
Usage (const char *program, int exitcode)
{
  fprintf (stderr, "Usage: %s [options]\n", program);
  fprintf (stderr, "Options:\n");
  fprintf (stderr, "  -h         print this message\n");
  fprintf (stderr, "  -q         quiet: run silently\n");
  fprintf (stderr, "  -v         verbose: print extra information\n");
  fprintf (stderr, "  -d level   set lalDebugLevel to level\n");
  exit (exitcode);
}


/*
 * ParseOptions ()
 *
 * Parses the argc - 1 option strings in argv[].
 *
 */
static void
ParseOptions (int argc, char *argv[])
{
  while (1)
  {
    int c = -1;

    c = getopt (argc, argv, "hqvd:");
    if (c == -1)
    {
      break;
    }

    switch (c)
    {
      case 'd': /* set debug level */
        lalDebugLevel = atoi (optarg);
        break;

      case 'v': /* verbose */
        ++verbose;
        break;

      case 'q': /* quiet: run silently (ignore error messages) */
        freopen ("/dev/null", "w", stderr);
        freopen ("/dev/null", "w", stdout);
        break;

      case 'h':
        Usage (argv[0], 0);
        break;

      default:
        Usage (argv[0], 1);
    }

  }

  if (optind < argc)
  {
    Usage (argv[0], 1);
  }

  return;
}

/** \endcond DONT_DOXYGEN */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#include <lal/LALStdlib.h>
#include <lal/Date.h>
#include <lal/AVFactories.h>

INT4 lalDebugLevel = 0;

NRCSID (LALTESTUTCTOGPSC, "$Id$");

int main(int argc, char *argv[])
{
  static LALStatus     status;
  LIGOTimeGPS          gpsTime;
  LIGOTimeGPS          refGPS;
  LALDate              utcDate;
  LALLeapSecAccuracy   accuracy = LALLEAPSEC_LOOSE;
  CHARVector          *timestamp = NULL;
  time_t               sec;

  if (argc > 1)
    lalDebugLevel = atoi(argv[1]);

  LALCHARCreateVector(&status, &timestamp, (UINT4)64);
  if (status.statusCode && lalDebugLevel > 0)
    {
      fprintf(stderr,
              "TestUTCtoGPS: error in LALCHARCreateVector, line %i, %s\n",
              __LINE__, LALTESTUTCTOGPSC);
      REPORTSTATUS(&status);
      return status.statusCode;
    }

  if (lalDebugLevel > 0)
    REPORTSTATUS(&status);
  
  /* Set the date */
  utcDate.unixDate.tm_year = 80;
  utcDate.unixDate.tm_mon  =  0;
  utcDate.unixDate.tm_mday =  6;
  utcDate.unixDate.tm_hour =  0;
  utcDate.unixDate.tm_min  =  0;
  utcDate.unixDate.tm_sec  =  0;
  utcDate.residualNanoSeconds = 0;

  LALUTCtoGPS(&status, &gpsTime, &utcDate, &accuracy);
  if (status.statusCode && lalDebugLevel > 0)
    {
      fprintf(stderr,
              "TestUTCtoGPS: error in LALUTCtoGPS, line %i, %s\n",
              __LINE__, LALTESTUTCTOGPSC);
      REPORTSTATUS(&status);
      return status.statusCode;
    }
  if (lalDebugLevel > 0)
    REPORTSTATUS(&status);
  
  refGPS.gpsSeconds = 0;
  refGPS.gpsNanoSeconds = 0;
  LALDateString(&status, timestamp, &utcDate);
  if (status.statusCode && lalDebugLevel > 0)
    {
      fprintf(stderr,
              "TestUTCtoGPS: error in LALDateString, line %i, %s\n",
              __LINE__, LALTESTUTCTOGPSC);
      REPORTSTATUS(&status);
      return status.statusCode;
    }
  if (lalDebugLevel > 0)
    REPORTSTATUS(&status);

  if (lalDebugLevel > 0)
    {
      fprintf(stderr, "For: %s\n", timestamp->data);
      fprintf(stderr, "  expect GPS = {%10d, %9d}\n", refGPS.gpsSeconds,
              refGPS.gpsNanoSeconds);
      fprintf(stderr, "  got    GPS = {%10d, %9d}\n", gpsTime.gpsSeconds,
              gpsTime.gpsNanoSeconds);
    }

  if (gpsTime.gpsSeconds != refGPS.gpsSeconds ||
      gpsTime.gpsNanoSeconds != refGPS.gpsNanoSeconds)
    {
      LALCHARDestroyVector(&status, &timestamp);
      if (status.statusCode && lalDebugLevel > 0)
        {
          fprintf(stderr,
                  "TestUTCtoGPS: error in LALCHARDestroyVector, line %i, %s\n",
                  __LINE__, LALTESTUTCTOGPSC);
          REPORTSTATUS(&status);
          return status.statusCode;
        }
      REPORTSTATUS(&status);
      LALCheckMemoryLeaks();
      return 1;
    }

  /* Set the date */
  utcDate.unixDate.tm_year = 94;
  utcDate.unixDate.tm_mon  =  6;
  utcDate.unixDate.tm_mday =  6;
  utcDate.unixDate.tm_hour = 23;
  utcDate.unixDate.tm_min  = 59;
  utcDate.unixDate.tm_sec  = 50;
  utcDate.residualNanoSeconds = 123456789;

  LALUTCtoGPS(&status, &gpsTime, &utcDate, &accuracy);
  if (status.statusCode && lalDebugLevel > 0)
    {
      fprintf(stderr,
              "TestUTCtoGPS: error in LALUTCtoGPS, line %i, %s\n",
              __LINE__, LALTESTUTCTOGPSC);
      REPORTSTATUS(&status);
      return status.statusCode;
    }
  if (lalDebugLevel > 0)
    REPORTSTATUS(&status);

  refGPS.gpsSeconds = 457574400;
  refGPS.gpsNanoSeconds = 123456789;
  LALDateString(&status, timestamp, &utcDate);
  if (status.statusCode && lalDebugLevel > 0)
    {
      fprintf(stderr,
              "TestUTCtoGPS: error in LALDateString, line %i, %s\n",
              __LINE__, LALTESTUTCTOGPSC);
      REPORTSTATUS(&status);
      return status.statusCode;
    }
  if (lalDebugLevel > 0) 
    REPORTSTATUS(&status);

  if (lalDebugLevel > 0)
    {
      fprintf(stderr, "For: %s\n", timestamp->data);
      fprintf(stderr, "  expect GPS = {%10d, %9d}\n", refGPS.gpsSeconds,
              refGPS.gpsNanoSeconds);
      fprintf(stderr, "  got    GPS = {%10d, %9d}\n", gpsTime.gpsSeconds,
              gpsTime.gpsNanoSeconds);
    }

  if (gpsTime.gpsSeconds != refGPS.gpsSeconds ||
      gpsTime.gpsNanoSeconds != refGPS.gpsNanoSeconds)
    {
      LALCHARDestroyVector(&status, &timestamp);
      if (status.statusCode && lalDebugLevel > 0)
        {
          fprintf(stderr,
                  "TestUTCtoGPS: error in LALCHARDestroyVector, line %i, %s\n",
                  __LINE__, LALTESTUTCTOGPSC);
          REPORTSTATUS(&status);
          return status.statusCode;
        }
      REPORTSTATUS(&status);
      LALCheckMemoryLeaks();
      return 1;
    }

  /* Set the date */
  utcDate.unixDate.tm_year = 94;
  utcDate.unixDate.tm_mon  =  6;
  utcDate.unixDate.tm_mday =  1;
  utcDate.unixDate.tm_hour =  0;
  utcDate.unixDate.tm_min  =  0;
  utcDate.unixDate.tm_sec  =  0;
  utcDate.residualNanoSeconds = 123456789;

  LALUTCtoGPS(&status, &gpsTime, &utcDate, &accuracy);
  if (status.statusCode && lalDebugLevel > 0)
    {
      fprintf(stderr,
              "TestUTCtoGPS: error in LALUTCtoGPS, line %i, %s\n",
              __LINE__, LALTESTUTCTOGPSC);
      REPORTSTATUS(&status);
      return status.statusCode;
    }
  if (lalDebugLevel > 0)
    REPORTSTATUS(&status);

  refGPS.gpsSeconds = 457056010;
  refGPS.gpsNanoSeconds = 123456789;
  LALDateString(&status, timestamp, &utcDate);
  if (status.statusCode && lalDebugLevel > 0)
    {
      fprintf(stderr,
              "TestUTCtoGPS: error in LALDateString, line %i, %s\n",
              __LINE__, LALTESTUTCTOGPSC);
      REPORTSTATUS(&status);
      return status.statusCode;
    }
  if (lalDebugLevel > 0)
    REPORTSTATUS(&status);

  if (lalDebugLevel > 0)
    {
      fprintf(stderr, "For: %s\n", timestamp->data);
      fprintf(stderr, "  expect GPS = {%10d, %9d}\n", refGPS.gpsSeconds,
              refGPS.gpsNanoSeconds);
      fprintf(stderr, "  got    GPS = {%10d, %9d}\n", gpsTime.gpsSeconds,
              gpsTime.gpsNanoSeconds);
    }

  if (gpsTime.gpsSeconds != refGPS.gpsSeconds ||
      gpsTime.gpsNanoSeconds != refGPS.gpsNanoSeconds)
    {
      LALCHARDestroyVector(&status, &timestamp);
      if (status.statusCode && lalDebugLevel > 0)
        {
          fprintf(stderr,
                  "TestUTCtoGPS: error in LALCHARDestroyVector, line %i, %s\n",
                  __LINE__, LALTESTUTCTOGPSC);
          REPORTSTATUS(&status);
          return status.statusCode;
        }
      REPORTSTATUS(&status);
      LALCheckMemoryLeaks();
      return 1;
    }

  /* Set the date */
  utcDate.unixDate.tm_year = 94;
  utcDate.unixDate.tm_mon  =  5;
  utcDate.unixDate.tm_mday = 30;
  utcDate.unixDate.tm_hour = 23;
  utcDate.unixDate.tm_min  = 59;
  utcDate.unixDate.tm_sec  = 58;
  utcDate.residualNanoSeconds = 123456789;

  refGPS.gpsSeconds = 457056007;
  refGPS.gpsNanoSeconds = 123456789;

  for (sec = 0; sec < 5; ++sec)
    {
      LALUTCtoGPS(&status, &gpsTime, &utcDate, &accuracy);
      if (status.statusCode && lalDebugLevel > 0)
        {
          fprintf(stderr,
                  "TestUTCtoGPS: error in LALUTCtoGPS, line %i, %s\n",
                  __LINE__, LALTESTUTCTOGPSC);
          REPORTSTATUS(&status);
          return status.statusCode;
        }
      if (lalDebugLevel > 0)
        REPORTSTATUS(&status);
      
      LALDateString(&status, timestamp, &utcDate);
      if (status.statusCode && lalDebugLevel > 0)
        {
          fprintf(stderr,
                  "TestUTCtoGPS: error in LALDateString, line %i, %s\n",
                  __LINE__, LALTESTUTCTOGPSC);
          REPORTSTATUS(&status);
          return status.statusCode;
        }
      if (lalDebugLevel > 0)
        REPORTSTATUS(&status);

      if (lalDebugLevel > 0)
        {
          fprintf(stderr, "For: %s\n", timestamp->data);
          fprintf(stderr, "  expect GPS = {%10d, %9d}\n", refGPS.gpsSeconds,
                  refGPS.gpsNanoSeconds);
          fprintf(stderr, "  got    GPS = {%10d, %9d}\n", gpsTime.gpsSeconds,
                  gpsTime.gpsNanoSeconds);
        }

      if (gpsTime.gpsSeconds != refGPS.gpsSeconds ||
          gpsTime.gpsNanoSeconds != refGPS.gpsNanoSeconds)
        {
          LALCHARDestroyVector(&status, &timestamp);
          if (status.statusCode && lalDebugLevel > 0)
            {
              fprintf(stderr,
                      "TestUTCtoGPS: error in LALCHARDestroyVector, line %i, %s\n",
                      __LINE__, LALTESTUTCTOGPSC);
              REPORTSTATUS(&status);
              return status.statusCode;
            }
          REPORTSTATUS(&status);
          LALCheckMemoryLeaks();
          return 1;
        }

      utcDate.unixDate.tm_sec++;
      if (utcDate.unixDate.tm_sec == 61)
        {
          utcDate.unixDate.tm_mon++;
          utcDate.unixDate.tm_mday = 1;
          utcDate.unixDate.tm_hour = 0;
          utcDate.unixDate.tm_min = 0;
          utcDate.unixDate.tm_sec = 0;
        }
      refGPS.gpsSeconds++;
    }

  /* Set the date */
  utcDate.unixDate.tm_year = 94;
  utcDate.unixDate.tm_mon  = 10;
  utcDate.unixDate.tm_mday = 16;
  utcDate.unixDate.tm_hour =  0;
  utcDate.unixDate.tm_min  =  0;
  utcDate.unixDate.tm_sec  =  0;
  utcDate.residualNanoSeconds = 123456789;

  LALUTCtoGPS(&status, &gpsTime, &utcDate, &accuracy);
  if (status.statusCode && lalDebugLevel > 0)
    {
      fprintf(stderr,
              "TestUTCtoGPS: error in LALUTCtoGPS, line %i, %s\n",
              __LINE__, LALTESTUTCTOGPSC);
      REPORTSTATUS(&status);
      return status.statusCode;
    }
  if (lalDebugLevel > 0)
    REPORTSTATUS(&status);

  refGPS.gpsSeconds = 468979210;
  refGPS.gpsNanoSeconds = 123456789;
  LALDateString(&status, timestamp, &utcDate);
  if (status.statusCode && lalDebugLevel > 0)
    {
      fprintf(stderr,
              "TestUTCtoGPS: error in LALDateString, line %i, %s\n",
              __LINE__, LALTESTUTCTOGPSC);
      REPORTSTATUS(&status);
      return status.statusCode;
    }
  if (lalDebugLevel > 0)
    REPORTSTATUS(&status);

  if (lalDebugLevel > 0)
    {
      fprintf(stderr, "For: %s\n", timestamp->data);
      fprintf(stderr, "  expect GPS = {%10d, %9d}\n", refGPS.gpsSeconds,
              refGPS.gpsNanoSeconds);
      fprintf(stderr, "  got    GPS = {%10d, %9d}\n", gpsTime.gpsSeconds,
              gpsTime.gpsNanoSeconds);
    }

  if (gpsTime.gpsSeconds != refGPS.gpsSeconds ||
      gpsTime.gpsNanoSeconds != refGPS.gpsNanoSeconds)
    {
      LALCHARDestroyVector(&status, &timestamp);
      if (status.statusCode && lalDebugLevel > 0)
        {
          fprintf(stderr,
                  "TestUTCtoGPS: error in LALCHARDestroyVector, line %i, %s\n",
                  __LINE__, LALTESTUTCTOGPSC);
          REPORTSTATUS(&status);
          return status.statusCode;
        }
      REPORTSTATUS(&status);
      LALCheckMemoryLeaks();
      return 1;
    }

  if (lalDebugLevel > 0)
    {
      utcDate.unixDate.tm_year = 72;
      utcDate.unixDate.tm_mon  =  0;
      utcDate.unixDate.tm_mday =  1;
      utcDate.unixDate.tm_hour =  0;
      utcDate.unixDate.tm_min  =  0;
      utcDate.unixDate.tm_sec  =  0;
      utcDate.residualNanoSeconds = 0;

      LALUTCtoGPS(&status, &gpsTime, &utcDate, &accuracy);
      if (status.statusCode && lalDebugLevel > 0)
        {
          fprintf(stderr,
                  "TestUTCtoGPS: error in LALUTCtoGPS, line %i, %s\n",
                  __LINE__, LALTESTUTCTOGPSC);
          REPORTSTATUS(&status);
          return status.statusCode;
        }
      REPORTSTATUS(&status);
      
      LALDateString(&status, timestamp, &utcDate);
      if (status.statusCode && lalDebugLevel > 0)
        {
          fprintf(stderr,
                  "TestUTCtoGPS: error in LALDateString, line %i, %s\n",
                  __LINE__, LALTESTUTCTOGPSC);
          REPORTSTATUS(&status);
          return status.statusCode;
        }
      REPORTSTATUS(&status);

      fprintf(stderr, "%s = GPS %10d\n", timestamp->data, gpsTime.gpsSeconds);
    }

  
  LALCHARDestroyVector(&status, &timestamp);
  if (status.statusCode && lalDebugLevel > 0)
    {
      fprintf(stderr,
              "TestUTCtoGPS: error in LALCHARDestroyVector, line %i, %s\n",
              __LINE__, LALTESTUTCTOGPSC);
      REPORTSTATUS(&status);
      return status.statusCode;
    }
  if (lalDebugLevel > 0)
    REPORTSTATUS(&status);
  LALCheckMemoryLeaks();
  return 0;
}
  

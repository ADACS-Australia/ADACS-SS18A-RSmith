/*
*  Copyright (C) 2013 Jolien Creighton
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

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#include <lal/Date.h>
#include <lal/LALgetopt.h>
#include <lal/LALStdlib.h>
#include <lal/LALString.h>
#include <lal/TimeSeries.h>
#include <lal/LALFrStream.h>
#include <lal/Units.h>

#define FS "\t" /* field separator */
#define RS "\n" /* record separator */
#define MAXDUR 16.0     /* max segment duration (s) */
#define MAXLEN 16384    /* max number of points in buffer */

/* globals */
LALCache *cache;
char *channel;
double t0;
double dt;

int usage(const char *program);
int parseargs(int argc, char **argv);
void output_INT2(LALFrStream * stream, const char *channame,
    LIGOTimeGPS * start, LIGOTimeGPS * end);
void output_INT4(LALFrStream * stream, const char *channame,
    LIGOTimeGPS * start, LIGOTimeGPS * end);
void output_INT8(LALFrStream * stream, const char *channame,
    LIGOTimeGPS * start, LIGOTimeGPS * end);
void output_UINT2(LALFrStream * stream, const char *channame,
    LIGOTimeGPS * start, LIGOTimeGPS * end);
void output_UINT4(LALFrStream * stream, const char *channame,
    LIGOTimeGPS * start, LIGOTimeGPS * end);
void output_UINT8(LALFrStream * stream, const char *channame,
    LIGOTimeGPS * start, LIGOTimeGPS * end);
void output_REAL4(LALFrStream * stream, const char *channame,
    LIGOTimeGPS * start, LIGOTimeGPS * end);
void output_REAL8(LALFrStream * stream, const char *channame,
    LIGOTimeGPS * start, LIGOTimeGPS * end);
void output_COMPLEX8(LALFrStream * stream, const char *channame,
    LIGOTimeGPS * start, LIGOTimeGPS * end);
void output_COMPLEX16(LALFrStream * stream, const char *channame,
    LIGOTimeGPS * start, LIGOTimeGPS * end);

int main(int argc, char *argv[])
{
    LALFrStream *stream;
    LALTYPECODE type;
    LIGOTimeGPS start;
    LIGOTimeGPS end;

    XLALSetErrorHandler(XLALAbortErrorHandler);

    parseargs(argc, argv);

    stream = XLALFrStreamCacheOpen(cache);

    /* determine the start of the output stream */
    if (t0 > 0.0)       /* use value provide by user */
        XLALGPSSetREAL8(&start, t0);
    else {      /* use start of input stream */
        XLALFrStreamTell(&start, stream);
        t0 = XLALGPSGetREAL8(&start);
    }

    /* determine the end of the output stream */
    if (dt > 0.0)       /* use value provide by user */
        XLALGPSSetREAL8(&end, t0 + dt);
    else {      /* use end of input stream */
        int mode = XLALFrStreamGetMode(stream);
        /* we're seeking to the end, so don't complain! */
        XLALFrStreamSetMode(stream,
            mode & ~LAL_FR_STREAM_TIMEWARN_MODE &
            LAL_FR_STREAM_IGNORETIME_MODE);
        XLALFrStreamSeekO(stream, 0, SEEK_END);
        XLALFrStreamTell(&end, stream);
        XLALFrStreamSetMode(stream, mode);
    }

    XLALFrStreamSeek(stream, &start);

    type = XLALFrStreamGetTimeSeriesType(channel, stream);
    switch (type) {
    case LAL_I2_TYPE_CODE:
        output_INT2(stream, channel, &start, &end);
        break;
    case LAL_I4_TYPE_CODE:
        output_INT4(stream, channel, &start, &end);
        break;
    case LAL_I8_TYPE_CODE:
        output_INT8(stream, channel, &start, &end);
        break;
    case LAL_U2_TYPE_CODE:
        output_UINT2(stream, channel, &start, &end);
        break;
    case LAL_U4_TYPE_CODE:
        output_UINT4(stream, channel, &start, &end);
        break;
    case LAL_U8_TYPE_CODE:
        output_UINT8(stream, channel, &start, &end);
        break;
    case LAL_S_TYPE_CODE:
        output_REAL4(stream, channel, &start, &end);
        break;
    case LAL_D_TYPE_CODE:
        output_REAL8(stream, channel, &start, &end);
        break;
    case LAL_C_TYPE_CODE:
        output_COMPLEX8(stream, channel, &start, &end);
        break;
    case LAL_Z_TYPE_CODE:
        output_COMPLEX16(stream, channel, &start, &end);
        break;
    default:
        fprintf(stderr, "unsupported channel type\n");
        return 1;
    }

    XLALFrStreamClose(stream);
    return 0;
}

int parseargs(int argc, char **argv)
{
    struct LALoption long_options[] = {
        {"help", no_argument, 0, 'h'},
        {"channel", required_argument, 0, 'c'},
        {"frame-cache", required_argument, 0, 'f'},
        {"frame-glob", required_argument, 0, 'g'},
        {"start-time", required_argument, 0, 's'},
        {"duration", required_argument, 0, 't'},
        {0, 0, 0, 0}
    };
    char args[] = "hc:f:g:s:t:";
    while (1) {
        int option_index = 0;
        int c;

        c = LALgetopt_long_only(argc, argv, args, long_options, &option_index);
        if (c == -1)    /* end of options */
            break;

        switch (c) {
        case 0:        /* if option set a flag, nothing else to do */
            if (long_options[option_index].flag)
                break;
            else {
                fprintf(stderr, "error parsing option %s with argument %s\n",
                    long_options[option_index].name, LALoptarg);
                exit(1);
            }
        case 'h':      /* help */
            usage(argv[0]);
            exit(0);
        case 'c':      /* channel */
            channel = XLALStringDuplicate(LALoptarg);
            break;
        case 'f':      /* frame-cache */
            cache = XLALCacheImport(LALoptarg);
            break;
        case 'g':      /* frame-cache */
            cache = XLALCacheGlob(NULL, LALoptarg);
            break;
        case 's':      /* start-time */
            t0 = atof(LALoptarg);
            break;
        case 't':      /* duration */
            dt = atof(LALoptarg);
            break;
        case '?':
        default:
            fprintf(stderr, "unknown error while parsing options\n");
            exit(1);
        }
    }

    if (LALoptind < argc) {
        fprintf(stderr, "extraneous command line arguments:\n");
        while (LALoptind < argc)
            fprintf(stderr, "%s\n", argv[LALoptind++]);
        exit(1);
    }

    /* sanity check parameters */

    if (!channel) {
        fprintf(stderr, "must specify a channel\n");
        usage(argv[0]);
        exit(1);
    }
    if (!cache) {
        fprintf(stderr, "must specify a frame cache or frame files\n");
        usage(argv[0]);
        exit(1);
    }

    return 0;
}

int usage(const char *program)
{
    fprintf(stderr, "usage: %s [options]\n", program);
    fprintf(stderr, "options:\n");
    fprintf(stderr, "\t-h, --help                   \tprint this message and exit\n");
    fprintf(stderr, "\t-c, CHAN --channel=CHAN      \tchannel name CHAN\n");
    fprintf(stderr, "\t-f CACHE, --frame-cache=CACHE\tframe cache file CACHE\n");
    fprintf(stderr, "\t-g GLOB, --frame-glob=GLOB   \tframe file glob string GLOB\n");
    fprintf(stderr, "\t-s T0, --start-time=T0       \tGPS start time T0 (s)\n");
    fprintf(stderr, "\t-t DT, --duration=DT         \tduration DT (s)\n");
    return 0;
}

#define DEFINE_OUTPUT_FUNCTION(laltype, format, ...) \
void output_ ## laltype (LALFrStream *stream, const char *channame, LIGOTimeGPS *start, LIGOTimeGPS *end) \
{ \
    double remaining; \
    laltype ## TimeSeries *series; \
    /* create zero-length time series */ \
    series = XLALCreate ## laltype ## TimeSeries(channame, start, 0.0, 0.0, &lalDimensionlessUnit, 0); \
    XLALFrStreamGet ## laltype ## TimeSeriesMetadata(series, stream); \
    while ((remaining = XLALGPSDiff(end, start)) > series->deltaT) { \
        size_t length; \
        size_t i; \
        remaining = (remaining > MAXDUR ? MAXDUR : remaining); \
        length = (size_t)floor(remaining / series->deltaT); \
        length = (length > MAXLEN ? MAXLEN : length); \
        if (series->data->length != length) \
            XLALResize ## laltype ## TimeSeries(series, 0, length); \
        XLALFrStreamGet ## laltype ## TimeSeries(series, stream); \
        for (i = 0; i < series->data->length; ++i) { \
            char tstr[32]; \
            laltype value = series->data->data[i]; \
            LIGOTimeGPS t = series->epoch; \
            XLALGPSAdd(&t, i * series->deltaT); \
            fprintf(stdout, "%s", XLALGPSToStr(tstr, &t)); \
            fputs(FS, stdout); \
            fprintf(stdout, format, __VA_ARGS__); \
            fputs(RS, stdout); \
        } \
        XLALGPSAdd(start, series->data->length * series->deltaT); \
    } \
    XLALDestroy ## laltype ## TimeSeries(series); \
}

/* *INDENT-OFF* */
DEFINE_OUTPUT_FUNCTION(INT2, "%" LAL_INT2_PRId, value)
DEFINE_OUTPUT_FUNCTION(INT4, "%" LAL_INT4_PRId, value)
DEFINE_OUTPUT_FUNCTION(INT8, "%" LAL_INT8_PRId, value)
DEFINE_OUTPUT_FUNCTION(UINT2, "%" LAL_INT2_PRIu, value)
DEFINE_OUTPUT_FUNCTION(UINT4, "%" LAL_INT4_PRIu, value)
DEFINE_OUTPUT_FUNCTION(UINT8, "%" LAL_INT8_PRIu, value)
DEFINE_OUTPUT_FUNCTION(REAL4, "%.7g", value)
DEFINE_OUTPUT_FUNCTION(REAL8, "%.15g", value)
DEFINE_OUTPUT_FUNCTION(COMPLEX8, "(%.7g,%.7g)", crealf(value), cimagf(value))
DEFINE_OUTPUT_FUNCTION(COMPLEX16, "(%.15g,%.15g)", creal(value), cimag(value))
/* *INDENT-ON* */

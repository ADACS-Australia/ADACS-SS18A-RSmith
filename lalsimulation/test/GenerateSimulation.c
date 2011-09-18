/*
*  Copyright (C) 2011 Nickolas Fotopoulos
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
#include <stdlib.h>
#include <time.h>

#include <lal/LALConstants.h>
#include <lal/LALDatatypes.h>
#include <lal/Date.h>
#include <lal/FrequencySeries.h>
#include <lal/LALSimInspiral.h>
#include <lal/LALSimIMR.h>
#include <lal/XLALError.h>
#include <lal/LALAdaptiveRungeKutta4.h>

typedef enum tagGSApproximant {
    GSApproximant_DEFAULT,
    GSApproximant_IMRPhenomA,
    GSApproximant_IMRPhenomB,
    GSApproximant_SpinTaylorT4,
    GSApproximant_NUM
} GSApproximant;

typedef enum tagGSDomain {
    GSDomain_TD,
    GSDomain_FD
} GSDomain;

/* internal storage is in SI units! */
typedef struct tagGSParams {
    GSApproximant approximant;/**< waveform family or "approximant" */
    GSDomain domain;          /**< flag for time or frequency domain waveform */
    int phaseO;               /**< twice PN order of the phase */
    int ampO;                 /**< twice PN order of the amplitude */
    LIGOTimeGPS *tRef;        /**< time at fRef */
    REAL8 phiRef;             /**< phase at fRef */
    REAL8 fRef;               /**< reference frequency */
    REAL8 deltaT;             /**< sampling interval */
    REAL8 deltaF;             /**< frequency resolution */
    REAL8 m1;                 /**< mass of companion 1 */
    REAL8 m2;                 /**< mass of companion 2 */
    REAL8 chi;                /**< dimensionless aligned-spin parameter */
    REAL8 f_min;              /**< start frequency */
    REAL8 f_max;              /**< end frequency */
    REAL8 distance;           /**< distance of source */
    REAL8 inclination;        /**< inclination of L relative to line of sight */
    REAL8 s1x;                /**< (x,y,z) components of spin of m1 body */
    REAL8 s1y;                /**< z-axis along line of sight, L in x-z plane */
    REAL8 s1z;                /**< dimensionless spin, Kerr bound: |s1| <= 1 */
    REAL8 s2x;                /**< (x,y,z) component ofs spin of m2 body */
    REAL8 s2y;                /**< z-axis along line of sight, L in x-z plane */
    REAL8 s2z;                /**< dimensionless spin, Kerr bound: |s1| <= 1 */
    char outname[256];        /**< file to which output should be written */
    int verbose;
} GSParams;

const char * usage =
"Generate a simulation using the lalsimulation library\n"
"\n"
"Various options are required depending on the approximant\n"
"--approximant APPROX       Supported approximants:\n"
"                             IMRPhenomA\n"
"                             IMRPhenomB\n"
"                             SpinTaylorT4\n"
"--phase-order ORD          Twice PN order of phase (e.g. ORD=7 <==> 3.5PN)\n"
"--amp-order ORD            Twice PN order of amplitude\n"
"--domain DOM               'TD' for time domain or 'FD' for frequency\n"
"                           domain; not all approximants support all domains\n"
"--tRef N                   Reference time in GPS seconds\n"
"--phiRef                   Phase at the reference frequency\n"
"--fRef FREF                Reference frequency in Hz\n"
"                           (default: FMIN)\n"
"--deltaT DT                Sampling interval in seconds\n"
"--deltaF DF                Sampling interval in seconds\n"
"--m1 M1                    Mass of the first object in solar masses\n"
"--m2 M2                    Mass of the second object in solar masses\n"
"--chi CHI                  Dimensionless aligned-spin parameter\n"
"--inclination IOTA         Angle in radians between line of sight (N) and \n"
"                           orbital angular momentum (L) at the reference\n"
"                           (default: face on)\n"
"--spin1x S1X               Vector components for spin of mass1\n"
"--spin1y S1Y               z-axis=line of sight, L in x-z plane at reference\n"
"--spin1z S1Z               Kerr limit: s1x^2 + s1y^2 + s1z^2 <= 1\n"
"--spin2x S2X               Vector components for spin of mass2\n"
"--spin2y S2Y               z-axis=line of sight, L in x-z plane at reference\n"
"--spin2z S2Z               Kerr limit: s2x^2 + s2y^2 + s2z^2 <= 1\n"
"--f-min FMIN               Frequency at which to start waveform in Hz\n"
"--f-max FMAX               Frequency at which to stop waveform in Hz\n"
"                           (default: generate as much as possible)\n"
"--distance D               Distance in Mpc\n"
"--outname FNAME            File to which output should be written (overwrites)\n"
"--verbose                  Provide this flag to add verbose output\n"
;

/* Parse command line, sanity check arguments, and return a newly
 * allocated GSParams object */
static GSParams *parse_args(ssize_t argc, char **argv) {
    ssize_t i;
    char msg[256];
    GSParams *params;
    params = (GSParams *) XLALMalloc(sizeof(GSParams));
    memset(params, 0, sizeof(GSParams));

    /* special case for no arguments (for make check) */
    if (argc == 1) {
        params->approximant = GSApproximant_IMRPhenomA;
        params->domain = GSDomain_FD;
        params->m1 = 3. * LAL_MSUN_SI;
        params->m2 = 3.1 * LAL_MSUN_SI;
        params->deltaF = 0.125;
        params->f_min = 40;
        params->distance = 100 * 1e6 * LAL_PC_SI;
    }

    /* consume command line */
    for (i = 1; i < argc; ++i) {
        if ((strcmp(argv[i], "-h") == 0) || (strcmp(argv[i], "--help") == 0)) {
            printf("%s", usage);
            if (params->tRef) XLALFree(params->tRef);
            XLALFree(params);
            exit(0);
        } else if (strcmp(argv[i], "--approximant") == 0) {
            i++;
            if (strcmp(argv[i], "IMRPhenomA") == 0)
                params->approximant = GSApproximant_IMRPhenomA;
            else if (strcmp(argv[i], "IMRPhenomB") == 0)
                params->approximant = GSApproximant_IMRPhenomB;
            else if (strcmp(argv[i], "SpinTaylorT4") == 0)
                params->approximant = GSApproximant_SpinTaylorT4;
            else {
                XLALPrintError("Error: Unknown approximant\n");
                goto fail;
            }
        } else if (strcmp(argv[i], "--domain") == 0) {
            i++;
            if (strcmp(argv[i], "TD") == 0)
                params->domain = GSDomain_TD;
            else if (strcmp(argv[i], "FD") == 0)
                params->domain = GSDomain_FD;
            else {
                XLALPrintError("Error: Unknown domain\n");
                goto fail;
            }
        } else if (strcmp(argv[i], "--phase-order") == 0) {
            params->phaseO = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--amp-order") == 0) {
            params->ampO = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--tRef") == 0) {
            params->tRef = (LIGOTimeGPS *) XLALMalloc(sizeof(LIGOTimeGPS));
            XLALINT8NSToGPS(params->tRef, 1000000000L * atof(argv[++i]));
        } else if (strcmp(argv[i], "--phiRef") == 0) {
            params->phiRef = atof(argv[++i]);
        } else if (strcmp(argv[i], "--fRef") == 0) {
            params->fRef = atof(argv[++i]);
        } else if (strcmp(argv[i], "--deltaT") == 0) {
            params->deltaT = atof(argv[++i]);
        } else if (strcmp(argv[i], "--deltaF") == 0) {
            params->deltaF = atof(argv[++i]);
        } else if (strcmp(argv[i], "--m1") == 0) {
            params->m1 = atof(argv[++i]) * LAL_MSUN_SI;
        } else if (strcmp(argv[i], "--m2") == 0) {
            params->m2 = atof(argv[++i]) * LAL_MSUN_SI;
        } else if (strcmp(argv[i], "--chi") == 0) {
            params->chi = atof(argv[++i]);
        } else if (strcmp(argv[i], "--spin1x") == 0) {
            params->s1x = atof(argv[++i]);
        } else if (strcmp(argv[i], "--spin1y") == 0) {
            params->s1y = atof(argv[++i]);
        } else if (strcmp(argv[i], "--spin1z") == 0) {
            params->s1z = atof(argv[++i]);
        } else if (strcmp(argv[i], "--spin2x") == 0) {
            params->s2x = atof(argv[++i]);
        } else if (strcmp(argv[i], "--spin2y") == 0) {
            params->s2y = atof(argv[++i]);
        } else if (strcmp(argv[i], "--spin2z") == 0) {
            params->s2z = atof(argv[++i]);
        } else if (strcmp(argv[i], "--f-min") == 0) {
            params->f_min = atof(argv[++i]);
        } else if (strcmp(argv[i], "--f-max") == 0) {
            params->f_max = atof(argv[++i]);
        } else if (strcmp(argv[i], "--distance") == 0) {
            params->distance = atof(argv[++i]) * 1e6 * LAL_PC_SI;
        } else if (strcmp(argv[i], "--inclination") == 0) {
            params->inclination = atof(argv[++i]);
        } else if (strcmp(argv[i], "--outname") == 0) {
            strncpy(params->outname, argv[++i], 256);
        } else if (strcmp(argv[i], "--verbose") == 0) {
            params->verbose = 1;
        } else {
            snprintf(msg, 256, "Error: invalid option: %s\n", argv[i]);
            XLALPrintError(msg);
            goto fail;
        }
    }

    /* generic domain checks */
    if (params->approximant == GSApproximant_DEFAULT) {
        XLALPrintError("Error: --approximant is a required parameter\n");
        goto fail;
    } else if ((params->domain == GSDomain_TD) ^ (params->deltaT > 0)) {
        XLALPrintError("Error: time-domain waveforms require --deltaT\n");
        goto fail;
    } else if ((params->domain == GSDomain_FD) ^ (params->deltaF > 0)) {
        XLALPrintError("Error: frequency-domain waveforms require --deltaF\n");
        goto fail;
    } else if ((params->phaseO < 0) || (params->phaseO > 8)) {
        XLALPrintError("Error: Invalid PN order for the phase!\n");
        goto fail;
    } else if ((params->ampO < 0) || (params->ampO > 5)) {
        XLALPrintError("Error: Invalid PN order for the amplitude!\n");
        goto fail;
    } else if ((params->m1 <= 0.) || (params->m2 <= 0.)) {
        XLALPrintError("Error: masses are required and must be positive\n");
        goto fail;
    } else if (fabs(params->chi) > 1) {
        XLALPrintError("Error: chi must be within -1 and 1\n");
        goto fail;
    } else if (params->f_min <= 0.) {
        XLALPrintError("Error: --f-min is required and must be positive\n");
        goto fail;
    } else if (params->f_max < 0.) {   /* f_max == 0 is OK */
        XLALPrintError("Error: --f-max must be positive\n");
        goto fail;
    } else if (params->distance <= 0.) {
        XLALPrintError("Error: --distance is required and must be positive\n");
        goto fail;
    } else if ((params->inclination < 0) || (params->inclination > LAL_PI)) {
        XLALPrintError("Error: --inclination must be between 0 and pi\n");
        goto fail;
    }

    /* waveform-specific checks for presence
     * tRef, masses, f_min, and distance have already been checked. */
    switch (params->approximant) {
        case GSApproximant_IMRPhenomA:
        case GSApproximant_IMRPhenomB:
        case GSApproximant_SpinTaylorT4:
            /* no additional checks required */
            break;
        default:
            XLALPrintError("Error: some lazy developer forgot to update waveform-specific checks\n");
    }

    /* fill in defaults */
    if (!params->tRef) {
        params->tRef = (LIGOTimeGPS *) XLALMalloc(sizeof(LIGOTimeGPS));
        *(params->tRef) = (LIGOTimeGPS) {0., 0.};
    }
    if (params->fRef == 0) params->fRef = params->f_min;
    if (*params->outname == '\0')
        strncpy(params->outname, "simulation.dat", 256);

    return params;

    fail:
    printf("%s", usage);
    XLALFree(params);
    exit(1);
}

static int dump_FD(FILE *f, COMPLEX16FrequencySeries *htilde) {
    ssize_t i;
    COMPLEX16 *dataPtr = htilde->data->data;

    fprintf(f, "# f htilde.re htilde.im\n");
    dataPtr = htilde->data->data;
    for (i=0; i < htilde->data->length; i++)
      fprintf(f, "%e %e %e\n", i * htilde->deltaF, dataPtr[i].re, dataPtr[i].im);
    return 0;
}

static int dump_TD(FILE *f, REAL8TimeSeries *hplus, REAL8TimeSeries *hcross) {
    ssize_t i;
    if (hplus->data->length != hcross->data->length) {
        XLALPrintError("Error: hplus and hcross are not the same length\n");
        return 1;
    } else if (hplus->deltaT != hcross->deltaT) {
        XLALPrintError("Error: hplus and hcross do not have the same sample rate\n");
        return 1;
    }

    fprintf(f, "# f hplus hcross\n");
    for (i=0; i < hplus->data->length; i++)
      fprintf(f, "%e %e %e\n", i * hplus->deltaT, hplus->data->data[i], hcross->data->data[i]);
    return 0;
}
/*
 * main
 */
int main (int argc , char **argv) {
    FILE *f;
    int status;
    int start_time;
    LIGOTimeGPS tRef;
    REAL8 LNhatx = 0., LNhaty = 0., LNhatz = 0., E1x = 0., E1y = 0., E1z = 0.;
    COMPLEX16FrequencySeries *htilde = NULL;
    REAL8TimeSeries *hplus = NULL;
    REAL8TimeSeries *hcross = NULL;
    GSParams *params;
    // For now, hardcode spin flags as 1.5PN SO + 2PN SS
    LALSpinInteraction spinFlags = LAL_SOInter | LAL_SSInter;

    /* set us up to fail hard */
    lalDebugLevel = 7;
    XLALSetErrorHandler(XLALAbortErrorHandler);

    /* parse commandline */
    params = parse_args(argc, argv);

    /* generate waveform */
    start_time = time(NULL);
    switch (params->domain) {
        case GSDomain_FD:
            switch (params->approximant) {
                case GSApproximant_IMRPhenomA:
                    XLALSimIMRPhenomAGenerateFD(&htilde, &tRef, params->phiRef, params->fRef, params->deltaF, params->m1, params->m2, params->f_min, params->f_max, params->distance);
                    break;
                case GSApproximant_IMRPhenomB:
                    XLALSimIMRPhenomBGenerateFD(&htilde, &tRef, params->phiRef, params->fRef, params->deltaF, params->m1, params->m2, params->chi, params->f_min, params->f_max, params->distance);
                    break;
                case GSApproximant_SpinTaylorT4:
                    XLALPrintError("Error: SpinTaylorT4 is not an FD waveform!\n");
                default:
                    XLALPrintError("Error: some lazy programmer forgot to add their FD waveform generation function\n");
            }
            break;
        case GSDomain_TD:
            switch (params->approximant) {
                case GSApproximant_IMRPhenomA:
                    XLALSimIMRPhenomAGenerateTD(&hplus, &hcross, &tRef, params->phiRef, params->fRef, params->deltaT, params->m1, params->m2, params->f_min, params->f_max, params->distance, params->inclination);
                    break;
                case GSApproximant_IMRPhenomB:
                    XLALSimIMRPhenomBGenerateTD(&hplus, &hcross, &tRef, params->phiRef, params->fRef, params->deltaT, params->m1, params->m2, params->chi, params->f_min, params->f_max, params->distance, params->inclination);
                    break;
                case GSApproximant_SpinTaylorT4:
                    LNhatx = sin(params->inclination);
                    LNhaty = 0.;
                    LNhatz = cos(params->inclination);
                    E1x = cos(params->inclination);
                    E1y = 0.;
                    E1z = - sin(params->inclination);
                    XLALSimInspiralSpinTaylorT4(&hplus, &hcross, &tRef, 
                            params->phiRef, 0., params->deltaT, params->m1, 
                            params->m2, params->fRef, params->distance, 
                            params->s1x, params->s1y, params->s1z, params->s2x,
                            params->s2y, params->s2z, LNhatx, LNhaty, LNhatz, 
                            E1x, E1y, E1z, spinFlags, params->phaseO, 
                            params->ampO);
                    break;
                default:
                    XLALPrintError("Error: some lazy programmer forgot to add their TD waveform generation function\n");
            }
            break;
        default:
            XLALPrintError("Error: only TD and FD waveform generation supported\n");
    }
    if (((params->domain == GSDomain_FD) && !htilde) ||
        ((params->domain == GSDomain_TD) && (!hplus || !hcross))) {
        XLALPrintError("Error: waveform generation failed\n");
        goto fail;
    }
    if (params->verbose)
        XLALPrintInfo("Generation took %.0f seconds\n", difftime(time(NULL), start_time));

    /* dump file */
    f = fopen(params->outname, "w");
    if (params->domain == GSDomain_FD)
        status = dump_FD(f, htilde);
    else
        status = dump_TD(f, hplus, hcross);
    fclose(f);
    if (status) goto fail;

    /* clean up */
    XLALFree(params->tRef);
    XLALFree(params);
    XLALDestroyCOMPLEX16FrequencySeries(htilde);
    return 0;

    fail:
    XLALFree(params->tRef);
    XLALFree(params);
    XLALDestroyCOMPLEX16FrequencySeries(htilde);
    return 1;
}

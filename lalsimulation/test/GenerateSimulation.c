/*
*  Copyright (C) 2011 Nickolas Fotopoulos, Evan Ochsner
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

#define LAL_USE_OLD_COMPLEX_STRUCTS
#include <lal/LALConstants.h>
#include <lal/LALDatatypes.h>
#include <lal/Date.h>
#include <lal/FrequencySeries.h>
#include <lal/LALSimInspiral.h>
#include <lal/LALSimIMR.h>
#include <lal/XLALError.h>
#include <lal/LALAdaptiveRungeKutta4.h>

typedef enum tagGSDomain {
    GSDomain_TD,
    GSDomain_FD
} GSDomain;

/* internal storage is in SI units! */
typedef struct tagGSParams {
    Approximant approximant;  /**< waveform family or "approximant" */
    GSDomain domain;          /**< flag for time or frequency domain waveform */
    int phaseO;               /**< twice PN order of the phase */
    int ampO;                 /**< twice PN order of the amplitude */
    REAL8 phiRef;             /**< phase at fRef */
    REAL8 fRef;               /**< reference frequency */
    REAL8 deltaT;             /**< sampling interval */
    REAL8 deltaF;             /**< frequency resolution */
    REAL8 m1;                 /**< mass of companion 1 */
    REAL8 m2;                 /**< mass of companion 2 */
    REAL8 f_min;              /**< start frequency */
    REAL8 f_max;              /**< end frequency */
    REAL8 distance;           /**< distance of source */
    REAL8 inclination;        /**< inclination of L relative to line of sight */
    REAL8 s1x;                /**< (x,y,z) components of spin of m1 body */
    REAL8 s1y;                /**< z-axis along line of sight, L in x-z plane */
    REAL8 s1z;                /**< dimensionless spin, Kerr bound: |s1| <= 1 */
    REAL8 s2x;                /**< (x,y,z) component ofs spin of m2 body */
    REAL8 s2y;                /**< z-axis along line of sight, L in x-z plane */
    REAL8 s2z;                /**< dimensionless spin, Kerr bound: |s2| <= 1 */
    REAL8 lambda1;	      /**< (tidal deformability of mass 1) / (total mass)^5 (dimensionless) */
    REAL8 lambda2;	      /**< (tidal deformability of mass 2) / (total mass)^5 (dimensionless) */
    LALSimInspiralWaveformFlags *waveFlags; /**< Set of flags to control special behavior of some waveform families */
    LALSimInspiralTestGRParam *nonGRparams; /**< Linked list of non-GR parameters. Pass in NULL for standard GR waveforms */
    int axisChoice;           /**< flag to choose reference frame for spin coordinates */
    int inspiralOnly;         /**< flag to choose if generating only the the inspiral 1 or also merger and ring-down*/
    char outname[256];        /**< file to which output should be written */
    int verbose;
} GSParams;

const char * usage =
"Generate a simulation using the lalsimulation library\n\n"
"The following options can be given (will assume a default value if omitted):\n"
"--domain DOM               'TD' for time domain (default) or 'FD' for frequency\n"
"                           domain; not all approximants support all domains\n"
"--approximant APPROX       Supported TD approximants:\n"
"                             TaylorT1 (default)\n"
"                             TaylorT2\n"
"                             TaylorT3\n"
"                             TaylorT4\n"
"                             TaylorEt\n"
"                             IMRPhenomA\n"
"                             IMRPhenomB\n"
"                             EOBNRv2\n"
"                             EOBNRv2HM\n"
"                             SEOBNRv1\n"
"                             SpinTaylorT4\n"
"                             PhenSpinTaylorRD\n"
"                           Supported FD approximants:\n"
"                             IMRPhenomA\n"
"                             IMRPhenomB\n"
"                             TaylorF2\n"
"                             TaylorF2RedSpin\n"
"                             TaylorF2RedSpinTidal\n"
"                           NOTE: Other approximants may be available if the\n"
"                           developer forgot to edit this help message\n"
"--phase-order ORD          Twice PN order of phase (default ORD=7 <==> 3.5PN)\n"
"--amp-order ORD            Twice PN order of amplitude (default 0 <==> Newt./restricted)\n"
"--phiRef                   Phase at the reference frequency (default 0)\n"
"--fRef FREF                Reference frequency in Hz\n"
"                           (default: 0)\n"
"--sample-rate SRATE        Sampling rate of TD approximants in Hz (default 4096)\n"
"--deltaF DF                Frequency bin size for FD approximants in Hz (default 1/8)\n"
"--m1 M1                    Mass of the first object in solar masses (default 10)\n"
"--m2 M2                    Mass of the second object in solar masses (default 1.4)\n"
"--inclination IOTA         Angle in radians between line of sight (N) and \n"
"                           orbital angular momentum (L) at the reference\n"
"                           (default 0, face on)\n"
"--spin1x S1X               Vector components for spin of mass1 (default all 0)\n"
"--spin1y S1Y               z-axis=line of sight, L in x-z plane at reference\n"
"--spin1z S1Z               Kerr limit: s1x^2 + s1y^2 + s1z^2 <= 1\n"
"--spin2x S2X               Vector components for spin of mass2 (default all 0)\n"
"--spin2y S2Y               z-axis=line of sight, L in x-z plane at reference\n"
"--spin2z S2Z               Kerr limit: s2x^2 + s2y^2 + s2z^2 <= 1\n"
"--interaction FLAG    Common choices: 'ALL' (default), 'NO', 'ALL_SPIN', 'TIDAL'\n"
"--tidal-lambda1 L1         (tidal deformability of mass 1) / (total mass)^5 (~4-80 for NS, 0 for BH) (default 0)\n"
"--tidal-lambda2 L2         (tidal deformability of mass 2) / (total mass)^5 (~4-80 for NS, 0 for BH) (default 0)\n"
"--f-min FMIN               Frequency at which to start waveform in Hz (default 40)\n"
"--f-max FMAX               Frequency at which to stop waveform in Hz\n"
"                           (default: generate as much as possible)\n"
"--distance D               Distance in Mpc (default 100)\n"
"--axis AXIS                for PhenSpin: 'View' (default), 'TotalJ', 'OrbitalL'\n"
"--outname FNAME            Output to file FNAME (default 'simulation.dat')\n"
"--verbose                  If included, add verbose output\n"
;

/* Parse command line, sanity check arguments, and return a newly
 * allocated GSParams object */
static GSParams *parse_args(ssize_t argc, char **argv) {
    ssize_t i;
    GSParams *params;
    params = (GSParams *) XLALMalloc(sizeof(GSParams));
    memset(params, 0, sizeof(GSParams));

    /* Set default values to the arguments */
    params->waveFlags = XLALSimInspiralCreateWaveformFlags();
    params->nonGRparams = NULL;
    params->approximant = TaylorT1;
    params->domain = GSDomain_TD;
    params->phaseO = 7;
    params->ampO = 0;
    params->phiRef = 0.;
    params->deltaT = 1./4096.;
    params->deltaF = 0.125;
    params->m1 = 10. * LAL_MSUN_SI;
    params->m2 = 1.4 * LAL_MSUN_SI;
    params->f_min = 40.;
    params->fRef = 0.;
    params->f_max = 0.; /* Generate as much as possible */
    params->distance = 100. * 1e6 * LAL_PC_SI;
    params->inclination = 0.;
    params->s1x = 0.;
    params->s1y = 0.;
    params->s1z = 0.;
    params->s2x = 0.;
    params->s2y = 0.;
    params->s2z = 0.;
    params->lambda1 = 0.;
    params->lambda2 = 0.;
    strncpy(params->outname, "simulation.dat", 256); /* output to this file */
    params->verbose = 0; /* No verbosity */

    /* consume command line */
    for (i = 1; i < argc; ++i) {
        if ((strcmp(argv[i], "-h") == 0) || (strcmp(argv[i], "--help") == 0)) {
            printf("%s", usage);
            XLALFree(params);
            exit(0);
        } else if (strcmp(argv[i], "--approximant") == 0) {
            params->approximant = XLALGetApproximantFromString(argv[++i]);
            if ( (int) params->approximant == XLAL_FAILURE) {
                XLALPrintError("Error: invalid value %s for --interaction-flag\n", argv[i]);
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
        } else if (strcmp(argv[i], "--phiRef") == 0) {
            params->phiRef = atof(argv[++i]);
        } else if (strcmp(argv[i], "--fRef") == 0) {
            params->fRef = atof(argv[++i]);
        } else if (strcmp(argv[i], "--sample-rate") == 0) {
            params->deltaT = 1./atof(argv[++i]);
        } else if (strcmp(argv[i], "--deltaF") == 0) {
            params->deltaF = atof(argv[++i]);
        } else if (strcmp(argv[i], "--m1") == 0) {
            params->m1 = atof(argv[++i]) * LAL_MSUN_SI;
        } else if (strcmp(argv[i], "--m2") == 0) {
            params->m2 = atof(argv[++i]) * LAL_MSUN_SI;
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
	} else if (strcmp(argv[i], "--interaction") == 0) {
            XLALSimInspiralSetInteraction( params->waveFlags,
                    XLALGetInteractionFromString(argv[++i]) );
            if ( (int) XLALSimInspiralGetInteraction(params->waveFlags)
                    == (int) XLAL_FAILURE) {
                XLALPrintError("Error: invalid value %s for --interaction\n", argv[i]);
                goto fail;
            }
	} else if (strcmp(argv[i], "--tidal-lambda1") == 0) {
            params->lambda1 = atof(argv[++i]);
	} else if (strcmp(argv[i], "--tidal-lambda2") == 0) {
            params->lambda2 = atof(argv[++i]);
        } else if (strcmp(argv[i], "--f-min") == 0) {
            params->f_min = atof(argv[++i]);
        } else if (strcmp(argv[i], "--f-max") == 0) {
            params->f_max = atof(argv[++i]);
        } else if (strcmp(argv[i], "--distance") == 0) {
            params->distance = atof(argv[++i]) * 1e6 * LAL_PC_SI;
        } else if (strcmp(argv[i], "--inclination") == 0) {
            params->inclination = atof(argv[++i]);
        } else if (strcmp(argv[i], "--axis") == 0) {
            XLALSimInspiralSetFrameAxis( params->waveFlags,
                    XLALGetFrameAxisFromString(argv[++i]) );
            if ( (int) XLALSimInspiralGetInteraction(params->waveFlags)
                    == (int) XLAL_FAILURE) {
                XLALPrintError("Error: invalid value %s for --axis\n", argv[i]);
                goto fail;
            }
        } else if (strcmp(argv[i], "--outname") == 0) {
            strncpy(params->outname, argv[++i], 256);
        } else if (strcmp(argv[i], "--verbose") == 0) {
            params->verbose = 1;
        } else {
            XLALPrintError("Error: invalid option: %s\n", argv[i]);
            goto fail;
        }
    }

    return params;

    fail:
    printf("%s", usage);
    XLALFree(params);
    exit(1);
}

static int dump_FD(FILE *f, COMPLEX16FrequencySeries *htilde) {
    size_t i;
    COMPLEX16 *dataPtr = htilde->data->data;

    fprintf(f, "# f htilde.re htilde.im\n");
    dataPtr = htilde->data->data;
    for (i=0; i < htilde->data->length; i++)
        fprintf(f, "%e %e %e\n", htilde->f0 + i * htilde->deltaF, 
                dataPtr[i].re, dataPtr[i].im);
    return 0;
}

static int dump_TD(FILE *f, REAL8TimeSeries *hplus, REAL8TimeSeries *hcross) {
    size_t i;
    REAL8 t0 = XLALGPSGetREAL8(&(hplus->epoch));
    if (hplus->data->length != hcross->data->length) {
        XLALPrintError("Error: hplus and hcross are not the same length\n");
        return 1;
    } else if (hplus->deltaT != hcross->deltaT) {
        XLALPrintError("Error: hplus and hcross do not have the same sample rate\n");
        return 1;
    }

    fprintf(f, "# t hplus hcross\n");
    for (i=0; i < hplus->data->length; i++)
        fprintf(f, "%e %e %e\n", t0 + i * hplus->deltaT, 
                hplus->data->data[i], hcross->data->data[i]);
    return 0;
}
/*
 * main
 */
int main (int argc , char **argv) {
    FILE *f;
    int status;
    int start_time;
    COMPLEX16FrequencySeries *htilde = NULL;
    REAL8TimeSeries *hplus = NULL;
    REAL8TimeSeries *hcross = NULL;
    GSParams *params;
	
    /* set us up to fail hard */
    lalDebugLevel = 7;
    XLALSetErrorHandler(XLALAbortErrorHandler);

    /* parse commandline */
    params = parse_args(argc, argv);

    /* generate waveform */
    start_time = time(NULL);
    switch (params->domain) {
        case GSDomain_FD:
            XLALSimInspiralChooseFDWaveform(&htilde, params->phiRef, 
                    params->deltaF, params->m1, params->m2, params->s1x, 
                    params->s1y, params->s1z, params->s2x, params->s2y, 
                    params->s2z, params->f_min, params->f_max, 
                    params->distance, params->inclination, params->lambda1, 
                    params->lambda2, params->waveFlags, params->nonGRparams,
                    params->ampO, params->phaseO, params->approximant);
            break;
        case GSDomain_TD:
            XLALSimInspiralChooseTDWaveform(&hplus, &hcross, params->phiRef, 
                    params->deltaT, params->m1, params->m2, params->s1x, 
                    params->s1y, params->s1z, params->s2x, params->s2y, 
                    params->s2z, params->f_min, params->fRef, 
                    params->distance, params->inclination, params->lambda1, 
                    params->lambda2, params->waveFlags, params->nonGRparams,
                    params->ampO, params->phaseO, params->approximant);
            break;
        default:
            XLALPrintError("Error: domain must be either TD or FD\n");
    }
    if (params->verbose)
        XLALPrintInfo("Generation took %.0f seconds\n", 
                difftime(time(NULL), start_time));
    if (((params->domain == GSDomain_FD) && !htilde) ||
        ((params->domain == GSDomain_TD) && (!hplus || !hcross))) {
        XLALPrintError("Error: waveform generation failed\n");
        goto fail;
    }

    /* dump file */
    f = fopen(params->outname, "w");
    if (params->domain == GSDomain_FD)
        status = dump_FD(f, htilde);
    else
        status = dump_TD(f, hplus, hcross);
    fclose(f);
    if (status) goto fail;

    /* clean up */
    XLALSimInspiralDestroyWaveformFlags(params->waveFlags);
    XLALSimInspiralDestroyTestGRParam(params->nonGRparams);
    XLALFree(params);
    XLALDestroyCOMPLEX16FrequencySeries(htilde);
    return 0;

    fail:
    XLALSimInspiralDestroyWaveformFlags(params->waveFlags);
    XLALSimInspiralDestroyTestGRParam(params->nonGRparams);
    XLALFree(params);
    XLALDestroyCOMPLEX16FrequencySeries(htilde);
    return 1;
}

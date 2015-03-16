/*
 * stopp_bayes.c - SGWB Standalone Analysis Pipeline
 *               - Bayesian Post Processing
 *
 * Copyright (C) 2004-2006,2009,2010 Adam Mercer
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
 * 02110-1301, USA
 */

/**
 * \file
 * \ingroup lalapps_inspiral
 *
 *
 * <dl>
 * <dt>Name</dt><dd>
 * <tt>lalapps_stopp_bayes</tt> --- Bayesian Stochastic Post Processing.</dd>
 *
 * <dt>Synopsis</dt><dd>
 * <tt>lalapps_stopp_bayes</tt> <i>options</i> <i>xml files</i>
 * <tt>--help</tt>
 * <tt>--version</tt>
 * <tt>--verbose</tt>
 * <tt>--cat-only</tt>
 * <tt>--analyse-only</tt>
 * <tt>--powerlaw-pdf</tt>
 * <tt>--text</tt>
 * <tt>--output</tt> <i>FILE</i>
 * <tt>--confidence</tt> <i>LEVEL</i></dd>
 *
 * <dt>Description</dt><dd>
 * <tt>lalapps_stopp_bayes</tt> performs Bayesian post processing upon output
 * from the main search code <tt>lalapps_stochastic</tt>.</dd>
 *
 * <dt>Options</dt><dd>
 * <dl>
 * <dt><tt>--help</tt></dt><dd>
 * Display usage information</dd>
 * <dt><tt>--version</tt></dt><dd>
 * Display version information</dd>
 * <dt><tt>--verbose</tt></dt><dd>
 * Verbose mode</dd>
 * <dt><tt>--cat-only</tt></dt><dd>
 * Only cat XML files together, don't perform post processing</dd>
 * <dt><tt>--analyse-only</tt></dt><dd>
 * Only perform post processing</dd>
 * <dt><tt>--powerlaw-pdf</tt></dt><dd>
 * Calculcate powerlaw PDF</dd>
 * <dt><tt>--text</tt></dt><dd>
 * Output file as text</dd>
 * <dt><tt>--output</tt> <i>FILE</i></dt><dd>
 * Write output data to <i>FILE</i></dd>
 * <dt><tt>--confidence</tt> <i>LEVEL</i></dt><dd>
 * Set confidence to <i>LEVEL</i> for calculating upperlimit</dd>
 * </dl></dd>
 *
 * <dt>Example</dt><dd>
 * An example usage of <tt>lalapps_stopp_bayes</tt> can be seen below.
 *
 * \code
 * > lalapps_stopp_bayes --output S3-H1L1-STOCHASTIC.xml \
 * >   H1L1-STOCHASTIC-753601044-753601242.xml \
 * >   H1L1-STOCHASTIC-753620042-753620352.xml \
 * >   H1L1-STOCHASTIC-753638864-753639462.xml \
 * >   H1L1-STOCHASTIC-753785374-753785707.xml \
 * >   H1L1-STOCHASTIC-753791744-753792342.xml
 * \endcode</dd>
 *
 * <dt>Author</dt><dd>
 * Adam Mercer</dd>
 * </dl>
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

#include <lal/Date.h>
#include <lal/LALgetopt.h>
#include <lal/LIGOLwXML.h>
#include <lal/LIGOLwXMLStochasticRead.h>
#include <lal/LIGOMetadataTables.h>

#include <gsl/gsl_sf_erf.h>

#include <lalapps.h>
#include <LALAppsVCSInfo.h>

/* verbose flag */
extern int vrbflg;


/* cvs info */
#define PROGRAM_NAME "stopp_bayes"

#define USAGE \
  "Usage: " PROGRAM_NAME " [options] [xml files]\n"\
  " --help                       display this message\n"\
  " --version                    display version\n"\
  " --verbose                    verbose mode\n"\
  " --cat-only                   only cat xml files\n"\
  " --analyse-only               only combine statistics\n"\
  " --powerlaw-pdf               calculate powerlaw pdf\n"\
  " --text                       output file as text\n"\
  " --output FILE                write output data to FILE\n"\
  " --confidence LEVEL           set confidence to LEVEL\n"

/* helper functions */

/* function to return the inverse complimentary error function */
static double stopp_erfcinv(double y)
{
  /*
   * based on dierfc() by Takuya OOURA:
   *
   * http://momonga.t.u-tokyo.ac.jp/~ooura/gamerf.html
   *
   * Copyright(C) 1996 Takuya OOURA (email: ooura@mmm.t.u-tokyo.ac.jp).
   * You may use, copy, modify this code for any purpose and
   * without fee. You may distribute this ORIGINAL package.
   */
  double s, t, u, w, x, z;

  z = y;

  if (y > 1)
    z = 2 - y;

  w = 0.916461398268964 - log(z);
  u = sqrt(w);
  s = (log(u) + 0.488826640273108) / w;
  t = 1 / (u + 0.231729200323405);
  x = u * (1 - s * (s * 0.124610454613712 + 0.5)) - \
      ((((-0.0728846765585675 * t + 0.269999308670029) * t + \
      0.150689047360223) * t + 0.116065025341614) * t + \
      0.499999303439796) * t;
  t = 3.97886080735226 / (x + 3.97886080735226);
  u = t - 0.5;
  s = (((((((((0.00112648096188977922 * u + \
      1.05739299623423047e-4) * u - 0.00351287146129100025) * u - \
      7.71708358954120939e-4) * u + 0.00685649426074558612) * u + \
      0.00339721910367775861) * u - 0.011274916933250487) * u - \
      0.0118598117047771104) * u + 0.0142961988697898018) * u + \
      0.0346494207789099922) * u + 0.00220995927012179067;
  s = ((((((((((((s * u - 0.0743424357241784861) * u - \
      0.105872177941595488) * u + 0.0147297938331485121) * u + \
      0.316847638520135944) * u + 0.713657635868730364) * u + \
      1.05375024970847138) * u + 1.21448730779995237) * u + \
      1.16374581931560831) * u + 0.956464974744799006) * u + \
      0.686265948274097816) * u + 0.434397492331430115) * u + \
      0.244044510593190935) * t - z * exp(x * x - 0.120782237635245222);
  x += s * (x * s + 1);

  if (y > 1)
    x = -x;

  return x;
}

/* main program loop */
INT4 main(INT4 argc, CHAR *argv[])
{
  /* status */
  LALStatus status = blank_status;

  /* LALgetopt flags */
  static int text_flag;
  static int cat_flag;
  static int analyse_flag;
  static int powerlaw_flag;

  /* counters */
  INT4 i, j;

  /* combined statistics variables */
  REAL8 numerator = 0;
  REAL8 denominator = 0;
  REAL8 yOpt = 0;
  REAL8 sigmaOpt = 0;
  REAL8 confidence = 0.95;
  REAL8 zeta;
  REAL8 upperlimit;

  /* pdf */
  REAL8 exponent;
  REAL8 pdf[100];
  REAL8 min_omega;
  REAL8 max_omega;
  REAL8 min_alpha = -1;
  REAL8 max_alpha = 1;
  REAL8 omega;
  REAL8 alpha;

  /* powerlaw pdf */
  REAL8 pdf_powerlaw[100][100];
  REAL8 freq;
  REAL8 freq_ref = 100;
  REAL8 omega_numerator;
  REAL8 omega_denominator;
  REAL8 sigma2_denominator;
  REAL8 omega_hat[100];
  REAL8 sigma2_omega_hat[100];

  /* program option variables */
  CHAR *outputFileName = NULL;

  /* xml data structures */
  LIGOLwXMLStream xmlStream;
  INT4 numSegments = 0;
  StochasticTable *stochHead = NULL;
  StochasticTable *thisStoch = NULL;
  MetadataTable outputTable;
  StochasticTable **stochHandle = NULL;

  /* text output file */
  FILE *out;
  FILE *pdf_out;
  FILE *omega_out;
  FILE *sigma_out;

  /* parse command line arguments */
  while (1)
  {
    /* LALgetopt arguments */
    static struct LALoption long_options[] =
    {
      /* options that set a flag */
      {"verbose", no_argument, &vrbflg, 1},
      {"text", no_argument, &text_flag, 1},
      {"cat-only", no_argument, &cat_flag, 1},
      {"analyse-only", no_argument, &analyse_flag, 1},
      {"powerlaw-pdf", no_argument, &powerlaw_flag, 1},
      /* options that don't set a flag */
      {"help", no_argument, 0, 'h'},
      {"version", no_argument, 0, 'v'},
      {"output", required_argument, 0, 'o'},
      {"confidence", required_argument, 0, 'c'},
      {0, 0, 0, 0}
    };
    int c;

    /* LALgetopt_long stores the option index here. */
    int option_index = 0;
    size_t LALoptarg_len;

    c = LALgetopt_long_only(argc, argv, "hvo:c:", long_options, &option_index);

    /* detect the end of the options */
    if (c == - 1)
    {
      /* end of options, break loop */
      break;
    }

    switch (c)
    {
      case 0:
        /* If this option set a flag, do nothing else now. */
        if (long_options[option_index].flag != 0)
        {
          break;
        }
        else
        {
          fprintf(stderr, "error parseing option %s with argument %s\n", \
              long_options[option_index].name, LALoptarg);
          exit(1);
        }
        break;

      case 'h':
        fprintf(stdout, USAGE);
        exit(0);
        break;

      case 'v':
        /* display version info and exit */
        fprintf(stdout, "Stochastic Post Processing: Bayesian\n");
        XLALOutputVersionString(stderr,0);
        exit(0);
        break;

      case 'o':
        /* create storage for the output file name */
        LALoptarg_len = strlen(LALoptarg) + 1;
        outputFileName = (CHAR *)calloc(LALoptarg_len, sizeof(CHAR));
        memcpy(outputFileName, LALoptarg, LALoptarg_len);
        break;

      case 'c':
        /* confidence level */
        confidence = atof(LALoptarg);
        if ((confidence >= 1) || (confidence <= 0))
        {
          fprintf(stderr, "invalid argument to --%s\n" \
              "confidence must be between 0 and 1, exclusive " \
              "(%.2f specified)\n", long_options[option_index].name, \
              confidence);
          exit(1);
        }
        break;

      case '?':
        exit(1);
        break;

      default:
        fprintf(stderr, "Unknown error while parsing options\n");
        exit(1);
    }
  }

  /* read in the input data from the rest of the arguments */
  if (LALoptind < argc)
  {
    for (i = LALoptind; i < argc; ++i)
    {
      struct stat infileStatus;

      /* if the named file does not exist, exit with an error */
      if (stat(argv[i], &infileStatus) == -1)
      {
        fprintf(stderr, "Error opening input file \"%s\"\n", argv[i]);
        exit(1);
      }

      if (!stochHead)
      {
        stochHandle = &stochHead;
      }
      else
      {
        stochHandle = &thisStoch->next;
      }

      /* read in the stochastic table */
      numSegments = LALStochasticTableFromLIGOLw(stochHandle, argv[i]);

      if (numSegments < 0)
      {
        fprintf(stderr, "Unable to read stochastic_table from \"%s\"\n", \
            argv[i]);
        exit(1);
      }
      else if (numSegments > 0)
      {
        if (vrbflg)
        {
          fprintf(stdout, "Read in %d segments from file \"%s\"\n", \
              numSegments, argv[i]);
        }

        /* scroll to end of list */
        thisStoch = *stochHandle;
        while (thisStoch->next)
        {
          thisStoch = thisStoch->next;
        }
      }
    }
  }

  if (!cat_flag)
  {
    /* combine statistics */
    for (thisStoch = stochHead; thisStoch; thisStoch = thisStoch->next)
    {
      numerator += thisStoch->cc_stat / (thisStoch->cc_sigma * \
          thisStoch->cc_sigma);
      denominator += 1./(thisStoch->cc_sigma * thisStoch->cc_sigma);
    }
    yOpt = (1./stochHead->duration.gpsSeconds) * (numerator / denominator);
    sigmaOpt = (1./stochHead->duration.gpsSeconds) * (1./sqrt(denominator));

    /* report point estimate and sigma */
    fprintf(stdout, "yOpt       = %e\n", yOpt);
    fprintf(stdout, "sigmaOpt   = %e\n", sigmaOpt);

    /* calculate upperlimit */
    zeta = yOpt / (sqrt(2) * sigmaOpt);
    upperlimit = yOpt + (sqrt(2) * sigmaOpt * \
        stopp_erfcinv((1 - confidence) * gsl_sf_erfc(-zeta)));
    fprintf(stdout, "upperlimit = %e\n", upperlimit);
  }

  /* calculate pdf */
  if (!powerlaw_flag)
  {
    /* pdf for constant spectra */
    min_omega = 0;
    max_omega = yOpt + (3 * sigmaOpt);
    for (i = 0; i < 100; i++)
    {
      omega = min_omega + (((i - 1)/99.) * (max_omega - min_omega));
      exponent = ((omega - yOpt) / sigmaOpt) * ((omega - yOpt) / sigmaOpt);
      pdf[i] = exp(-0.5 * exponent);
    }
  }
  else
  {
    /* pdf for power law spectra */
    min_omega = 0;
    max_omega = 1; /*(10 * yOpt)/stochHead->duration.gpsSeconds;*/
    min_alpha = -4;
    max_alpha = 4;

    /* loop for \Omega_R */
    for (i = 0; i < 100; i++)
    {
      /* loop for \alpha */
      for (j = 0; j < 100; j++)
      {
        omega = min_omega + ((i/99.) * (max_omega - min_omega));
        alpha = min_alpha + ((j/99.) * (max_alpha - min_alpha));

        /* initialise numerator */
        omega_numerator = 0;
        omega_denominator = 0;
        sigma2_denominator = 0;

        /* loop over segments */
        for (thisStoch = stochHead; thisStoch; thisStoch = thisStoch->next)
        {
          /* get frequency of middle of the band */
          freq = thisStoch->f_min + ((thisStoch->f_max - \
                thisStoch->f_min) / 2.);

          /* \hat{\Omega}_R */
          omega_numerator += (thisStoch->cc_stat / (thisStoch->cc_sigma * \
                thisStoch->cc_sigma)) * pow((freq/freq_ref), alpha);
          omega_denominator += (1. / (thisStoch->cc_sigma * \
                thisStoch->cc_sigma)) * pow((freq/freq_ref), 2 * alpha);

          /* sigma^2_{\hat{\Omega}_R} */
          sigma2_denominator += (1. / (thisStoch->cc_sigma * \
                thisStoch->cc_sigma)) * pow((freq/freq_ref), 2 * alpha);
        }

        /* construct \hat{\Omega}_R */
        omega_hat[j] = omega_numerator / (stochHead->duration.gpsSeconds * \
            omega_denominator);

        /* construct sigma^2_{\hat{\Omega}_R} */
        sigma2_omega_hat[j] = 1. / (stochHead->duration.gpsSeconds * \
              stochHead->duration.gpsSeconds * sigma2_denominator);

        /* construct pdf */
        pdf_powerlaw[i][j] = exp(-0.5 * ((omega - omega_hat[j]) / \
              sqrt(sigma2_omega_hat[j])) * ((omega - omega_hat[j]) / \
                sqrt(sigma2_omega_hat[j])));
      }
    }
  }

  if (!cat_flag)
  {
    if (powerlaw_flag)
    {
      /* open omega and sigma output files */
      if ((omega_out = fopen("omega.dat", "w")) == NULL)
      {
        fprintf(stderr, "Can't open file for omega output...\n");
        exit(1);
      }
      if ((sigma_out = fopen("sigma.dat", "w")) == NULL)
      {
        fprintf(stderr, "Can't open file for sigma output...\n");
        exit(1);
      }

      /* save out omega and sigma */
      for (j = 0; j < 100; j++)
      {
        alpha = min_alpha + ((j/99.) * (max_alpha - min_alpha));
        fprintf(omega_out, "%e %e\n", alpha, omega_hat[j]);
        fprintf(sigma_out, "%e %e\n", alpha, sqrt(sigma2_omega_hat[j]));
      }

      /* close files */
      fclose(omega_out);
      fclose(sigma_out);
    }

    /* save out pdf */
    if ((pdf_out = fopen("pdf.dat", "w")) == NULL)
    {
      fprintf(stderr, "Can't open file for pdf output...\n");
      exit(1);
    }
    if (powerlaw_flag)
    {
      for (i = 0; i < 100; i++)
      {
        for (j = 0; j < 100; j++)
        {
          omega = min_omega + ((i/99.) * (max_omega - min_omega));
          alpha = min_alpha + ((j/99.) * (max_alpha - min_alpha));
          fprintf(pdf_out, "%e %e %e\n", omega, alpha, pdf_powerlaw[i][j]);
        }

        /* gnuplot */
        fprintf(pdf_out, "\n");
      }
    }
    else
    {
      for (i = 0; i < 100; i++)
      {
        omega = min_omega + (((i - 1)/99.) * (max_omega - min_omega));
        fprintf(pdf_out, "%e %e\n", omega, pdf[i]);
      }
    }
    fclose(pdf_out);
  }

  if (!analyse_flag)
  {
    /* output as text file */
    if (text_flag)
    {
      /* open output file */
      if ((out = fopen(outputFileName, "w")) == NULL)
      {
        fprintf(stderr, "Can't open file \"%s\" for output...\n", \
            outputFileName);
        exit(1);
      }

      /* write details of events */
      for (thisStoch = stochHead; thisStoch; thisStoch = thisStoch->next)
      {
        fprintf(out, "%d %e %e\n", thisStoch->start_time.gpsSeconds, \
            thisStoch->cc_stat, thisStoch->cc_sigma);
      }

      /* close output file */
      fclose(out);
    }

    /* output as xml file */
    else
    {
      /* open xml file stream */
      memset(&xmlStream, 0, sizeof(LIGOLwXMLStream));
      LAL_CALL(LALOpenLIGOLwXMLFile(&status, &xmlStream, outputFileName), \
          &status);

      /* write stochastic table */
      if (stochHead)
      {
        outputTable.stochasticTable = stochHead;
        LAL_CALL(LALBeginLIGOLwXMLTable(&status, &xmlStream, \
              stochastic_table), &status);
        LAL_CALL(LALWriteLIGOLwXMLTable(&status, &xmlStream, outputTable, \
              stochastic_table), &status);
        LAL_CALL(LALEndLIGOLwXMLTable(&status, &xmlStream), &status);
      }

      /* close xml file */
      LAL_CALL(LALCloseLIGOLwXMLFile(&status, &xmlStream), &status);
    }
  }

  /* check for memory leaks and exit */
  LALCheckMemoryLeaks();
  exit(0);
}

/*
 * vim: et
 */

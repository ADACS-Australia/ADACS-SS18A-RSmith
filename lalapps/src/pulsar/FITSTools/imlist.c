//
// From https://heasarc.gsfc.nasa.gov/docs/software/fitsio/cexamples.html:
//
// FITS Tools: Handy FITS Utilities that illustrate how to use CFITSIO
// -------------------------------------------------------------------
//
// These are working programs written in ANSI C that illustrate how one can
// easily read, write, and modify FITS files using the CFITSIO library. Most of
// these programs are very short, containing only a few 10s of lines of
// executable code or less, yet they perform quite useful operations on FITS
// files. Copy the programs to your local machine, then compile, and link them
// with the CFITSIO library. A short description of how to use each program can
// be displayed by executing the program without any command line arguments.
//
// You may freely modify, reuse, and redistribute these programs as you wish. It
// is often easier to use one of these programs as a template when writing a new
// program, rather than coding the new program completely from scratch.
//

#include <config.h>
#include <string.h>
#include <stdio.h>

#if defined(HAVE_LIBCFITSIO)
#include <fitsio.h>
#else
#error CFITSIO library is not available
#endif

#if !defined(PAGER) || !defined(HAVE_POPEN) || !defined(HAVE_PCLOSE)
#define popen(...) stdout
#define pclose(...)
#endif

int main(int argc, char *argv[])
{
  fitsfile *fptr = 0;   /* FITS file pointer, defined in fitsio.h */
  int status = 0;   /* CFITSIO status value MUST be initialized to zero! */
  int bitpix = 0, naxis = 0, ii = 0;
  long naxes[2] = {1,1}, fpixel[2] = {1,1};
  double *pixels = 0;
  char format[20], hdformat[20];

  if (argc != 2) {
    fprintf(stderr, "Usage:  imlist filename[ext][section filter] \n");
    fprintf(stderr, "\n");
    fprintf(stderr, "List the the pixel values in a FITS image \n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Example: \n");
    fprintf(stderr, "  imlist image.fits                    - list the whole image\n");
    fprintf(stderr, "  imlist image.fits[100:110,400:410]   - list a section\n");
    fprintf(stderr, "  imlist table.fits[2][bin (x,y) = 32] - list the pixels in\n");
    fprintf(stderr, "         an image constructed from a 2D histogram of X and Y\n");
    fprintf(stderr, "         columns in a table with a binning factor = 32\n");
    return(0);
  }

  FILE *fout = popen(PAGER, "w");
  if (fout == NULL) {
    fprintf(stderr, "Could not execute '%s'\n", PAGER);
    return(1);
  }

  if (!fits_open_file(&fptr, argv[1], READONLY, &status))
  {
    if (!fits_get_img_param(fptr, 2, &bitpix, &naxis, naxes, &status) )
    {
      if (naxis > 2 || naxis == 0)
        fprintf(stderr, "Error: only 1D or 2D images are supported\n");
      else
      {
        /* get memory for 1 row */
        pixels = (double *) malloc(naxes[0] * sizeof(double));

        if (pixels == NULL) {
          fprintf(stderr, "Memory allocation error\n");
          return(1);
        }

        if (bitpix > 0) {  /* set the default output format string */
          strcpy(hdformat, "   %7d");
          strcpy(format,   "   %7.0g");
        } else {
          strcpy(hdformat, "   %15d");
          strcpy(format,   "   %15.5g");
        }

        /* loop over all the rows in the image, top to bottom */
        for (fpixel[1] = naxes[1]; fpixel[1] >= 1; fpixel[1]--)
        {
          if (fits_read_pix(fptr, TDOUBLE, fpixel, naxes[0], NULL,
                            pixels, NULL, &status) )  /* read row of pixels */
            break;  /* jump out of loop on error */

          for (ii = 0; ii < naxes[0]; ii++)
            fprintf(fout, format, pixels[ii]);   /* print each value  */
          fprintf(fout, "\n");                    /* terminate line */
        }
        free(pixels);
      }
    }
    fits_close_file(fptr, &status);
  }

  pclose(fout);

  if (status) fits_report_error(stderr, status); /* print any error message */
  return(status);
}

/*
* Copyright (C) 2010 Reinhard Prix
*  Copyright (C) 2007 Jolien Creighton, Reinhard Prix
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

#include <lal/Date.h>
#include <lal/LALString.h>

#include <lal/UserInput.h>

typedef struct
{
  REAL8 argNum;
  CHAR * argStr;
  BOOLEAN argBool;
  INT4 argInt;
  BOOLEAN argB2;
  CHAR *string2;	// will be read from config-file
  INT4 dummy;
  LIGOTimeGPS epochGPS;
  LIGOTimeGPS epochMJDTT;
  REAL8 longHMS;
  REAL8 longRad;
  REAL8 latDMS;
  REAL8 latRad;
} UserInput_t;

/**
 * some basic consistency checks of the (XLAL) UserInput module, far from exhaustive,
 * but should be enough to catch big obvious malfunctions
 */
int
main(int argc, char *argv[])
{
  int i, my_argc = 8;
  char **my_argv;
  const char *argv_in[] = { "progname", "--argNum=1", "--argStr=xyz", "--argBool=true", "-a", "1", "-b", "@" TEST_DATA_DIR "ConfigFileSample.cfg" };
  UserInput_t XLAL_INIT_DECL(my_uvars);

  XLAL_CHECK ( argc == 1, XLAL_EINVAL, "No input arguments allowed.\n");

  my_argv = XLALCalloc ( my_argc, sizeof(char*) );
  for (i=0; i < my_argc; i ++ )
    {
      my_argv[i] = XLALCalloc ( 1, strlen(argv_in[i])+1);
      strcpy ( my_argv[i], argv_in[i] );
    }

  /* ---------- Register all test user-variables ---------- */
  UserInput_t *uvar = &my_uvars;
  uvar->string2 = XLALStringDuplicate ( "this is the default value");

  XLAL_CHECK ( XLALregREALUserStruct( argNum, 0, UVAR_REQUIRED, "Testing float argument") == XLAL_SUCCESS, XLAL_EFUNC );
  XLAL_CHECK ( XLALregSTRINGUserStruct( argStr, 0, UVAR_REQUIRED, "Testing string argument") == XLAL_SUCCESS, XLAL_EFUNC );
  XLAL_CHECK ( XLALregBOOLUserStruct( argBool, 0, UVAR_REQUIRED, "Testing bool argument") == XLAL_SUCCESS, XLAL_EFUNC );
  XLAL_CHECK ( XLALregINTUserStruct( argInt, 'a', UVAR_REQUIRED, "Testing INT argument") == XLAL_SUCCESS, XLAL_EFUNC );
  XLAL_CHECK ( XLALregINTUserStruct( dummy,  'c', UVAR_OPTIONAL, "Testing INT argument") == XLAL_SUCCESS, XLAL_EFUNC );
  XLAL_CHECK ( XLALregBOOLUserStruct( argB2, 'b', UVAR_REQUIRED, "Testing short-option bool argument") == XLAL_SUCCESS, XLAL_EFUNC );
  XLAL_CHECK ( XLALregSTRINGUserStruct( string2, 0, UVAR_REQUIRED, "Testing another string argument") == XLAL_SUCCESS, XLAL_EFUNC );
  XLAL_CHECK ( XLALregEPOCHUserStruct( epochGPS, 0, UVAR_REQUIRED, "Testing epoch given as GPS time") == XLAL_SUCCESS, XLAL_EFUNC );
  XLAL_CHECK ( XLALregEPOCHUserStruct( epochMJDTT, 0, UVAR_REQUIRED, "Testing epoch given as MJD(TT) time") == XLAL_SUCCESS, XLAL_EFUNC );
  XLAL_CHECK ( XLALregRAJUserStruct( longHMS, 0, UVAR_REQUIRED, "Testing RAJ(HMS) argument") == XLAL_SUCCESS, XLAL_EFUNC );
  XLAL_CHECK ( XLALregRAJUserStruct( longRad, 0, UVAR_REQUIRED, "Testing RAJ(rad) argument") == XLAL_SUCCESS, XLAL_EFUNC );
  XLAL_CHECK ( XLALregDECJUserStruct( latDMS, 0, UVAR_REQUIRED, "Testing DECJ(DMS) argument") == XLAL_SUCCESS, XLAL_EFUNC );
  XLAL_CHECK ( XLALregDECJUserStruct( latRad, 0, UVAR_REQUIRED, "Testing DECJ(rad) argument") == XLAL_SUCCESS, XLAL_EFUNC );

  /* ---------- now read all input from commandline and config-file ---------- */
  XLAL_CHECK ( XLALUserVarReadAllInput ( my_argc, my_argv ) == XLAL_SUCCESS, XLAL_EFUNC );

  /* ---------- test help-string generation */
  CHAR *helpstr;
  XLAL_CHECK ( (helpstr = XLALUserVarHelpString ( argv[0])) != NULL, XLAL_EFUNC );
  XLALFree ( helpstr );

  /* ---------- test log-generation */
  CHAR *logstr;
  XLAL_CHECK ( ( logstr = XLALUserVarGetLog (   UVAR_LOGFMT_CFGFILE )) != NULL, XLAL_EFUNC );
  XLALFree ( logstr );

  /* ---------- test values were read in correctly ---------- */
  XLAL_CHECK ( uvar->argNum == 1, XLAL_EFAILED, "Failed to read in argNum\n" );
  XLAL_CHECK ( strcmp ( uvar->argStr, "xyz" ) == 0, XLAL_EFAILED, "Failed to read in argStr\n" );
  XLAL_CHECK ( uvar->argBool, XLAL_EFAILED, "Failed to read in argBool\n" );
  XLAL_CHECK ( uvar->argInt == 1, XLAL_EFAILED, "Failed to read in argInt\n" );
  XLAL_CHECK ( uvar->argB2, XLAL_EFAILED, "Failed to read in argB2\n" );
  XLAL_CHECK ( strcmp ( uvar->string2, "this is also possible, and # here does nothing; and neither does semi-colon " ) == 0, XLAL_EFAILED, "Failed to read in string2\n" );

  char buf1[256], buf2[256];
  XLAL_CHECK ( XLALGPSCmp ( &uvar->epochGPS, &uvar->epochMJDTT ) == 0, XLAL_EFAILED, "GPS epoch %s differs from MJD(TT) epoch %s\n",
               XLALGPSToStr ( buf1, &uvar->epochGPS), XLALGPSToStr ( buf2, &uvar->epochMJDTT ) );

  REAL8 diff, tol = 3e-15;
  XLAL_CHECK ( (diff = fabs(uvar->longHMS - uvar->longRad)) < tol, XLAL_EFAILED, "longitude(HMS) = %.16g differs from longitude(rad) = %.16g by %g > tolerance\n", uvar->longHMS, uvar->longRad, diff, tol );
  XLAL_CHECK ( (diff = fabs(uvar->latDMS - uvar->latRad)) < tol, XLAL_EFAILED, "latitude(HMS) = %.16g differs from latitude(rad) = %.16g by %g > tolerance\n", uvar->latDMS, uvar->latRad, diff, tol );

  /* ----- cleanup ---------- */
  XLALDestroyUserVars();
  for (i=0; i < my_argc; i ++ ) {
    XLALFree ( my_argv[i] );
  }
  XLALFree ( my_argv );

  LALCheckMemoryLeaks();

  return XLAL_SUCCESS;

} // main()

dnl $Id$
ifelse(TYPECODE,`Z',`define(`TYPE',`COMPLEX16')define(`SIZE',`8')')dnl
ifelse(TYPECODE,`C',`define(`TYPE',`COMPLEX8')define(`SIZE',`4')')dnl
ifelse(TYPECODE,`D',`define(`TYPE',`REAL8')define(`SIZE',`8')')dnl
ifelse(TYPECODE,`S',`define(`TYPE',`REAL4')define(`SIZE',`4')')dnl
ifelse(TYPECODE,`I2',`define(`TYPE',`INT2')define(`SIZE',`2')')dnl
ifelse(TYPECODE,`I4',`define(`TYPE',`INT4')define(`SIZE',`4')')dnl
ifelse(TYPECODE,`I8',`define(`TYPE',`INT8')define(`SIZE',`8')')dnl
ifelse(TYPECODE,`U2',`define(`TYPE',`UINT2')define(`SIZE',`2')')dnl
ifelse(TYPECODE,`U4',`define(`TYPE',`UINT4')define(`SIZE',`4')')dnl
ifelse(TYPECODE,`U8',`define(`TYPE',`UINT8')define(`SIZE',`8')')dnl
define(`DATACODE',TYPECODE)define(`DATA',TYPE)define(`COMPLEX',`0')dnl
ifelse(TYPECODE,`Z',`define(`DATACODE',`D')define(`DATA',`REAL8')define(`COMPLEX',`1')')dnl
ifelse(TYPECODE,`C',`define(`DATACODE',`S')define(`DATA',`REAL4')define(`COMPLEX',`1')')dnl
define(`GTYPE',`format(`%sGrid',TYPE)')dnl
define(`FUNC',`format(`LAL%sReadGrid',TYPECODE)')dnl
define(`CREATE',`format(`LAL%sCreateGrid',TYPECODE)')dnl
define(`DESTROY',`format(`LAL%sDestroyGrid',TYPECODE)')dnl
define(`FMT',`format(`LAL_%s_FORMAT',DATA)')dnl
define(`STRINGTODATA',`format(`LALStringTo%s',DATACODE)')dnl
dnl
void
FUNC ( LALStatus *stat, GTYPE **grid, FILE *stream )
{
  UINT4Vector dims;        /* parameter for creating grid */
  CHARVector *line = NULL; /* current line being read */
  CHAR *start, *end;       /* start and end of a token on a line */
  UINT4 nTot = 0;          /* number of elements read */
  UINT4 n;                 /* countdown index from nTot */
  TYPE *gData = NULL;      /* pointer to data in output grid */
#if COMPLEX
  int numRead = 0;         /* number of data values read from file */
#endif

  /* Temporary storage for metadata fields: */
  CHAR name[LALNameLength] = "";
  UINT4 *dimLength = NULL;
  REAL8 *offset = NULL;
  REAL8 *interval = NULL;
  LALUnit sampleUnits = lalDimensionlessUnit;
  LALUnit *dimUnits = NULL;
  UINT4 nDimLength = 0, nOffset = 0, nInterval = 0, nDimUnits = 0;

  INITSTATUS( stat, "FUNC", STREAMGRIDINPUTC );
  ATTATCHSTATUSPTR( stat );

  /* Default values. */
  name[0] = '\0';
  sampleUnits = lalDimensionlessUnit;

  /* Check for valid input arguments. */
  ASSERT( stream, stat, STREAMINPUTH_ENUL, STREAMINPUTH_MSGENUL );
  ASSERT( grid, stat, STREAMINPUTH_ENUL, STREAMINPUTH_MSGENUL );
  ASSERT( !(*grid), stat, STREAMINPUTH_EOUT, STREAMINPUTH_MSGEOUT );

  /*******************************************************************
   * PARSE METADATA HEADER                                           *
   *******************************************************************/

  /* Skip over blank lines; start points to the first non-whitespace
     character (or '\0' if there are none). */
  TRY( LALCHARReadVector( stat->statusPtr, &line, stream ), stat );
  start = line->data;
  while ( isspace( *start ) ) {
    start++;
    if ( *start == '\0' ) {
      TRY( LALCHARDestroyVector( stat->statusPtr, &line ), stat );
      TRY( LALCHARReadVector( stat->statusPtr, &line, stream ), stat );
      start = line->data;
    }
  }

  /* As long as '#' is the first non-whitespace character of any
     nonblank lines... */
  while ( *start == '#' ) {
    CHAR *startValue; /* start of substring giving metadata value */
    CHAR *endValue;   /* end of substring giving metadata value */

    /* Skip to the start of the metadata field tag. */
    do
      start++;
    while ( isspace( *start ) );

    /* Mark the end of the tag and the start of the metadata value. */
    end = start;
    while ( !isspace( *end ) && *end != '=' && *end != '\0' )
      end++;
    startValue = end;
    while ( isspace( *startValue ) || *startValue == '=' )
      startValue++;
    if ( startValue != end ) {
      *end = '\0';

      /* Parse name field. */
      if ( !strcmp( start, "name" ) ) {
	if ( *startValue != '"' )
	  LALWarning( stat, LALREADGRIDC_HEADER "name" );
	else
	  LALLiteralToString( stat->statusPtr, name, startValue,
			      LALNameLength );
      }

      /* Parse sampleUnits field. */
      else if ( !strcmp( start, "sampleUnits" ) ) {
	CHARVector unitString;
	if ( *startValue != '"' )
	  LALWarning( stat, LALREADGRIDC_HEADER "name" );
	else {
	  endValue = ++startValue;
	  while ( *endValue != '"' && *endValue != '\n' &&
		  *endValue != '\0' )
	    endValue++;
	  if ( *endValue != '"' )
	    LALWarning( stat, LALREADGRIDC_HEADER "name" );
	  else {
	    *endValue = '\0';
	    unitString.length = strlen( startValue ) + 1;
	    unitString.data = startValue;
	    LALParseUnitString( stat->statusPtr, &sampleUnits, &unitString );
	    if ( stat->statusPtr->statusCode == UNITSH_EPARSE ) {
#ifndef NDEBUG
	      if ( lalDebugLevel & LALERROR ) {
		LALPrintError( "\tCONTINUE: Ignoring preceding error\n" );
		DETATCHSTATUSPTR( stat );
		ATTATCHSTATUSPTR( stat );
	      }
#endif
	      LALWarning( stat, LALREADGRIDC_HEADER "sampleUnits" );
	    } else {
	      BEGINFAIL( stat ) {
		CLEANUP;
	      } ENDFAIL( stat );
	    }
	  }
	}
      }

      /* Parse dimLength field. */
      else if ( !strcmp( start, "dimLength" ) ) {
	UINT4 *data;
	U4Buffer *here, *head;
	here = head = (U4Buffer *)LALMalloc( sizeof(U4Buffer) );
	if ( !here ) {
	  CLEANUP;
	  ABORT( stat, STREAMINPUTH_EMEM, STREAMINPUTH_MSGEMEM );
	}
	here->next = NULL;
	data = here->U4;
	n = BUFFSIZE;
	endValue = startValue;
	/* Read components into the buffer list. */
	do {
	  do {
	    LALStringToU4( stat->statusPtr, data++,
			   startValue = endValue, &endValue );
	    BEGINFAIL( stat ) {
	      FREEU4BUFFER( head );
	      CLEANUP;
	    } ENDFAIL( stat );
	  } while ( endValue != startValue && --n );
	  if ( !n ) {
	    here->next = (U4Buffer *)LALMalloc( sizeof(U4Buffer) );
	    here = here->next;
	    if ( !here ) {
	      FREEU4BUFFER( head );
	      CLEANUP;
	      ABORT( stat, STREAMINPUTH_EMEM, STREAMINPUTH_MSGEMEM );
	    }
	    here->next = NULL;
	    data = here->U4;
	    nTot += n = BUFFSIZE;
	  } else
	    nTot += BUFFSIZE - n;
	} while ( endValue != startValue );
	/* Copy components into dimLength vector. */
	if ( nTot == 0 )
	  LALWarning( stat, LALREADGRIDC_HEADER "dimLength" );
	else {
	  if ( dimLength )
	    LALFree( dimLength );
	  if ( !( dimLength = LALMalloc( nTot*sizeof(UINT4) ) ) ) {
	    FREEU4BUFFER( head );
	    CLEANUP;
	    ABORT( stat, STREAMINPUTH_EMEM, STREAMINPUTH_MSGEMEM );
	  }
	  nDimLength = nTot;
	  here = head;
	  data = dimLength;
	  n = BUFFSIZE;
	  if ( n > nTot )
	    n = nTot;
	  while ( n ) {
	    memcpy( data, here->U4, n*sizeof(UINT4) );
	    data += n;
	    nTot -= n;
	    here = here->next;
	    n = BUFFSIZE;
	    if ( n > nTot )
	      n = nTot;
	  }
	}
	/* dimLength is complete, so reset everything else. */
	FREEU4BUFFER( head );
	nTot = 0;
      }

      /* Parse offset field. */
      else if ( !strcmp( start, "offset" ) ) {
	REAL8 *data;
	DBuffer *here, *head;
	here = head = (DBuffer *)LALMalloc( sizeof(DBuffer) );
	if ( !here ) {
	  CLEANUP;
	  ABORT( stat, STREAMINPUTH_EMEM, STREAMINPUTH_MSGEMEM );
	}
	here->next = NULL;
	data = here->D;
	n = BUFFSIZE;
	endValue = startValue;
	/* Read components into the buffer list. */
	do {
	  do {
	    LALStringToD( stat->statusPtr, data++,
			  startValue = endValue, &endValue );
	    BEGINFAIL( stat ) {
	      FREEDBUFFER( head );
	      CLEANUP;
	    } ENDFAIL( stat );
	  } while ( endValue != startValue && --n );
	  if ( !n ) {
	    here->next = (DBuffer *)LALMalloc( sizeof(DBuffer) );
	    here = here->next;
	    if ( !here ) {
	      FREEDBUFFER( head );
	      CLEANUP;
	      ABORT( stat, STREAMINPUTH_EMEM, STREAMINPUTH_MSGEMEM );
	    }
	    here->next = NULL;
	    data = here->D;
	    nTot += n = BUFFSIZE;
	  } else
	    nTot += BUFFSIZE - n;
	} while ( endValue != startValue );
	/* Copy components into offset vector. */
	if ( nTot == 0 )
	  LALWarning( stat, LALREADGRIDC_HEADER "offset" );
	else {
	  if ( offset )
	    LALFree( offset );
	  if ( !( offset = LALMalloc( nTot*sizeof(REAL8) ) ) ) {
	    FREEDBUFFER( head );
	    CLEANUP;
	    ABORT( stat, STREAMINPUTH_EMEM, STREAMINPUTH_MSGEMEM );
	  }
	  nOffset = nTot;
	  here = head;
	  data = offset;
	  n = BUFFSIZE;
	  if ( n > nTot )
	    n = nTot;
	  while ( n ) {
	    memcpy( data, here->D, n*sizeof(REAL8) );
	    data += n;
	    nTot -= n;
	    here = here->next;
	    n = BUFFSIZE;
	    if ( n > nTot )
	      n = nTot;
	  }
	}
	/* offset is complete, so reset everything else. */
	FREEDBUFFER( head );
	nTot = 0;
      }

      /* Parse interval field. */
      else if ( !strcmp( start, "interval" ) ) {
	REAL8 *data;
	DBuffer *here, *head;
	here = head = (DBuffer *)LALMalloc( sizeof(DBuffer) );
	if ( !here ) {
	  CLEANUP;
	  ABORT( stat, STREAMINPUTH_EMEM, STREAMINPUTH_MSGEMEM );
	}
	here->next = NULL;
	data = here->D;
	n = BUFFSIZE;
	endValue = startValue;
	/* Read components into the buffer list. */
	do {
	  do {
	    LALStringToD( stat->statusPtr, data++,
			  startValue = endValue, &endValue );
	    BEGINFAIL( stat ) {
	      FREEDBUFFER( head );
	      CLEANUP;
	    } ENDFAIL( stat );
	  } while ( endValue != startValue && --n );
	  if ( !n ) {
	    here->next = (DBuffer *)LALMalloc( sizeof(DBuffer) );
	    here = here->next;
	    if ( !here ) {
	      FREEDBUFFER( head );
	      CLEANUP;
	      ABORT( stat, STREAMINPUTH_EMEM, STREAMINPUTH_MSGEMEM );
	    }
	    here->next = NULL;
	    data = here->D;
	    nTot += n = BUFFSIZE;
	  } else
	    nTot += BUFFSIZE - n;
	} while ( endValue != startValue );
	/* Copy components into interval vector. */
	if ( nTot == 0 )
	  LALWarning( stat, LALREADGRIDC_HEADER "interval" );
	else {
	  if ( interval )
	    LALFree( interval );
	  if ( !( interval = LALMalloc( nTot*sizeof(REAL8) ) ) ) {
	    FREEDBUFFER( head );
	    CLEANUP;
	    ABORT( stat, STREAMINPUTH_EMEM, STREAMINPUTH_MSGEMEM );
	  }
	  nInterval = nTot;
	  here = head;
	  data = interval;
	  n = BUFFSIZE;
	  if ( n > nTot )
	    n = nTot;
	  while ( n ) {
	    memcpy( data, here->D, n*sizeof(REAL8) );
	    data += n;
	    nTot -= n;
	    here = here->next;
	    n = BUFFSIZE;
	    if ( n > nTot )
	      n = nTot;
	  }
	}
	/* interval is complete, so reset everything else. */
	FREEDBUFFER( head );
	nTot = 0;
      }

      /* Parse dimUnits field. */
      else if ( !strcmp( start, "dimUnits" ) ) {
	CHARVector unitString;
	LALUnit *data;
	UnitBuffer *here, *head;
	here = head = (UnitBuffer *)LALMalloc( sizeof(UnitBuffer) );
	if ( !here ) {
	  CLEANUP;
	  ABORT( stat, STREAMINPUTH_EMEM, STREAMINPUTH_MSGEMEM );
	}
	here->next = NULL;
	data = here->Unit;
	n = BUFFSIZE;
	endValue = startValue + 1;
	if ( *startValue != '\0' ) {
	  while ( *endValue != '"' && *endValue != '\0' )
	    endValue++;
	}
	/* Read components into the buffer list. */
	while ( *startValue == '"' && *endValue == '"' ) {
	  while ( *startValue == '"' && *endValue == '"' && n-- ) {
	    *endValue = '\0';
	    startValue++;
	    unitString.length = strlen( startValue ) + 1;
	    unitString.data = startValue;
	    LALParseUnitString( stat->statusPtr, data++, &unitString );
	    if ( stat->statusPtr->statusCode == UNITSH_EPARSE ) {
#ifndef NDEBUG
	      if ( lalDebugLevel & LALERROR ) {
		LALPrintError( "\tCONTINUE: Ignoring preceding error\n" );
		DETATCHSTATUSPTR( stat );
		ATTATCHSTATUSPTR( stat );
	      }
#endif
	      startValue = endValue;
	    } else {
	      BEGINFAIL( stat ) {
		FREEUNITBUFFER( head );
		CLEANUP;
	      } ENDFAIL( stat );
	      startValue = endValue + 1;
	      while ( isspace( *startValue ) )
		startValue++;
	      endValue = startValue + 1;
	      if ( *startValue != '\0' ) {
		while ( *endValue != '"' && *endValue != '\0' )
		  endValue++;
	      }
	    }
	  }
	  if ( !n ) {
	    here->next = (UnitBuffer *)LALMalloc( sizeof(UnitBuffer) );
	    here = here->next;
	    if ( !here ) {
	      FREEUNITBUFFER( head );
	      CLEANUP;
	      ABORT( stat, STREAMINPUTH_EMEM, STREAMINPUTH_MSGEMEM );
	    }
	    here->next = NULL;
	    data = here->Unit;
	    nTot += n = BUFFSIZE;
	  } else
	    nTot += BUFFSIZE - n;
	}
	/* Copy components into dimUnits vector. */
	if ( nTot == 0 )
	  LALWarning( stat, LALREADGRIDC_HEADER "dimUnits" );
	else {
	  if ( dimUnits )
	    LALFree( dimUnits );
	  if ( !( dimUnits = LALMalloc( nTot*sizeof(LALUnit) ) ) ) {
	    FREEUNITBUFFER( head );
	    CLEANUP;
	    ABORT( stat, STREAMINPUTH_EMEM, STREAMINPUTH_MSGEMEM );
	  }
	  nDimUnits = nTot;
	  here = head;
	  data = dimUnits;
	  n = BUFFSIZE;
	  if ( n > nTot )
	    n = nTot;
	  while ( n ) {
	    memcpy( data, here->Unit, n*sizeof(LALUnit) );
	    data += n;
	    nTot -= n;
	    here = here->next;
	    n = BUFFSIZE;
	    if ( n > nTot )
	      n = nTot;
	  }
	}
	/* dimUnits is complete, so reset everything else. */
	FREEUNITBUFFER( head );
	nTot = 0;
      }

      /* Parse datatype field. */
#ifndef NDEBUG
      else if ( !strcmp( start, "datatype" ) ) {
	if ( lalDebugLevel & LALWARNING ) {
	  endValue = startValue;
	  while ( !isspace( *endValue ) && *endValue != '\0' )
	    endValue++;
	  *endValue = '\0';
	  if ( strcmp( startValue, "GTYPE" ) ) {
	    LALWarning( stat, "GTYPE data expected" );
	    LALPrintError( "\t%s data being read\n", startValue );
	  }
	}
      }
#endif

      /* No other recognized tags; ignore anything else. */
    }

    /* Read in next line, skipping over blank lines. */
    TRY( LALCHARDestroyVector( stat->statusPtr, &line ), stat );
    TRY( LALCHARReadVector( stat->statusPtr, &line, stream ), stat );
    start = line->data;
    while ( isspace( *start ) ) {
      start++;
      if ( *start == '\0' ) {
	TRY( LALCHARDestroyVector( stat->statusPtr, &line ), stat );
	TRY( LALCHARReadVector( stat->statusPtr, &line, stream ), stat );
	start = line->data;
      }
    }
  }


  /*******************************************************************
   * PARSE DATA                                                      *
   *******************************************************************/

  end = start;

  /* Make sure that all the dimensions are consistently given. */
  if ( !nDimLength || !nOffset || !nInterval ) {
    CLEANUP;
    ABORT( stat, STREAMINPUTH_EDLEN, STREAMINPUTH_MSGEDLEN );
  } else if ( nDimLength < nOffset || nInterval != nOffset ||
	      ( nDimUnits && ( nDimUnits != nOffset ) ) ) {
    CLEANUP;
    ABORT( stat, STREAMINPUTH_EDLEN, STREAMINPUTH_MSGEDLEN );
  }

  /* Create the grid structure and fill in the metadata. */
  dims.length = nDimLength;
  dims.data = dimLength;
  CREATE ( stat->statusPtr, grid, &dims, nOffset );
  BEGINFAIL( stat ) {
    CLEANUP;
  } ENDFAIL( stat );
  LALFree( dimLength );
  memcpy( (*grid)->offset->data, offset, nOffset*sizeof(REAL8) );
  LALFree( offset );
  memcpy( (*grid)->interval->data, interval, nInterval*sizeof(REAL8) );
  LALFree( interval );
  if ( dimUnits ) {
    memcpy( (*grid)->dimUnits, dimUnits, nDimUnits*sizeof(REAL8) );
    LALFree( dimUnits );
  } else {
    UINT4 i; /* an index */
    for ( i = 0; i < nOffset; i++ )
      (*grid)->dimUnits[i] = lalDimensionlessUnit;
  }
  memcpy( (*grid)->name, name, LALNameLength*sizeof(CHAR) );
  (*grid)->sampleUnits = sampleUnits;
  gData = (*grid)->data->data;

  /* Begin reading the line in memory. */
  nTot = 1;
  for ( n = 0; n < (*grid)->data->dimLength->length; n++ )
    nTot *= (*grid)->data->dimLength->data[n];
#if COMPLEX
  numRead = 0;
  do {
    STRINGTODATA ( stat->statusPtr, &(gData->re), start = end,
		   &end );
    BEGINFAIL( stat ) {
      TRY( LALCHARDestroyVector( stat->statusPtr, &line ), stat );
      TRY( DESTROY ( stat->statusPtr, grid ), stat );
    } ENDFAIL( stat );
    if ( start != end ) {
      numRead = 1;
      STRINGTODATA ( stat->statusPtr, &(gData->im), start = end,
		     &end );
      BEGINFAIL( stat ) {
	TRY( LALCHARDestroyVector( stat->statusPtr, &line ), stat );
	TRY( DESTROY ( stat->statusPtr, grid ), stat );
      } ENDFAIL( stat );
      if ( start != end ) {
	numRead = 0;
	gData++;
      }
    }
  } while ( end != start && --nTot );
#else
  do {
    STRINGTODATA ( stat->statusPtr, gData++, start = end, &end );
    BEGINFAIL( stat ) {
      TRY( LALCHARDestroyVector( stat->statusPtr, &line ), stat );
      TRY( DESTROY ( stat->statusPtr, grid ), stat );
    } ENDFAIL( stat );
  } while ( end != start && --nTot );
  gData--;
#endif

  /* Read remaining data with fscanf(). */
  TRY( LALCHARDestroyVector( stat->statusPtr, &line ), stat );
#if COMPLEX
  if ( numRead == 1 ) {
    if ( fscanf( stream, "%" FMT, &(gData->im) ) != 1 ) {
      TRY( DESTROY ( stat->statusPtr, grid ), stat );
      ABORT( stat, STREAMINPUTH_ESLEN, STREAMINPUTH_MSGESLEN );
    }
    gData++;
    nTot--;
  }
  while ( nTot-- ) {
    if ( fscanf( stream, "%" FMT, &(gData->re) ) != 1 ||
	 fscanf( stream, "%" FMT, &(gData->im) ) != 1 ) {
      TRY( DESTROY ( stat->statusPtr, grid ), stat );
      ABORT( stat, STREAMINPUTH_ESLEN, STREAMINPUTH_MSGESLEN );
    }
    gData++;
  }
#else
  while ( nTot-- )
    if ( fscanf( stream, "%" FMT, gData++ ) != 1 ) {
      TRY( DESTROY ( stat->statusPtr, grid ), stat );
      ABORT( stat, STREAMINPUTH_ESLEN, STREAMINPUTH_MSGESLEN );
    }
#endif

  /* Data have been stored successfully.  So, clean up and exit. */
  DETATCHSTATUSPTR( stat );
  RETURN( stat );
}

/**** <lalVerbatim file="FrameSeriesCV">
 * Author: Jolien D. E. Creighton
 * $Id$
 **** </lalVerbatim> */

/**** <lalLaTeX>
 *
 * \subsection{Module \texttt{FrameSeries.c}}
 *
 * These routines are used to read/write time series data from/to frame files.
 *
 * \subsubsection*{Prototypes}
 * \input{FrameSeriesCP}
 * \idx{LALFrGetINT2TimeSeries}
 * \idx{LALFrGetINT2TimeSeriesMetadata}
 * \idx{LALFrGetINT4TimeSeries}
 * \idx{LALFrGetINT4TimeSeriesMetadata}
 * \idx{LALFrGetINT8TimeSeries}
 * \idx{LALFrGetINT8TimeSeriesMetadata}
 * \idx{LALFrGetREAL4TimeSeries}
 * \idx{LALFrGetREAL4TimeSeriesMetadata}
 * \idx{LALFrGetREAL8TimeSeries}
 * \idx{LALFrGetREAL8TimeSeriesMetadata}
 * \idx{LALFrGetCOMPLEX8TimeSeries}
 * \idx{LALFrGetCOMPLEX8TimeSeriesMetadata}
 * \idx{LALFrGetCOMPLEX16TimeSeries}
 * \idx{LALFrGetCOMPLEX16TimeSeriesMetadata}
 * \idx{LALFrGetINT2FrequencySeries}
 * \idx{LALFrGetINT2FrequencySeriesMetadata}
 * \idx{LALFrGetINT4FrequencySeries}
 * \idx{LALFrGetINT4FrequencySeriesMetadata}
 * \idx{LALFrGetINT8FrequencySeries}
 * \idx{LALFrGetINT8FrequencySeriesMetadata}
 * \idx{LALFrGetREAL4FrequencySeries}
 * \idx{LALFrGetREAL4FrequencySeriesMetadata}
 * \idx{LALFrGetREAL8FrequencySeries}
 * \idx{LALFrGetREAL8FrequencySeriesMetadata}
 * \idx{LALFrGetCOMPLEX8FrequencySeries}
 * \idx{LALFrGetCOMPLEX8FrequencySeriesMetadata}
 * \idx{LALFrGetCOMPLEX16FrequencySeries}
 * \idx{LALFrGetCOMPLEX16FrequencySeriesMetadata}
 * \idx{LALFrWriteINT2TimeSeries}
 * \idx{LALFrWriteINT4TimeSeries}
 * \idx{LALFrWriteINT8TimeSeries}
 * \idx{LALFrWriteREAL4TimeSeries}
 * \idx{LALFrWriteREAL8TimeSeries}
 * \idx{LALFrWriteCOMPLEX8TimeSeries}
 * \idx{LALFrWriteCOMPLEX16TimeSeries}
 * \idx{LALFrGetTimeSeriesType}
 *
 * \subsubsection*{Description}
 *
 * The routines
 * \texttt{LALFrGet}$\langle\mbox{datatype}\rangle$\texttt{TimeSeries()}
 * search the frame for a specified channel.  If the time series supplied has
 * data storage allocated, then the specified amount of data is filled from
 * the frame stream.  If no space has been allocated (so that the data field
 * is \texttt{NULL}), then only the channel information is returned in the
 * time series (e.g., the start time of the next data and the time step size).
 *
 * Because it is good coding practice to not tinker directly with the innards
 * of structures like time series, the behaviour described above is undesirable
 * whereby the time series reading functions do useful things when part of the
 * time series structures passed into them is not initialized.  To address
 * this, the routines
 * \texttt{LALFrGet}$\langle\mbox{datatype}\rangle$\texttt{TimeSeriesMetadata()}
 * are provided.  These routines accept a fully initialized time series
 * structure, and poplulate only the metadata from the frame stream.  New code
 * should be written to use \emph{these} functions when only the time series
 * meta data is desired.
 *
 * The routine \texttt{LALFrGetTimeSeriesType} returns the type of the time
 * series corresponding to the specified channel.  This is needed if it is not
 * known in advance what type of data is stored in the channel.
 *
 * The routines
 * \texttt{LALFrWrite}$\langle\mbox{datatype}\rangle$\texttt{TimeSeries()}
 * outputs a given time series as a new frame file with the filename
 * $\langle\mbox{source}\rangle$\verb+-+$\langle\mbox{description}\rangle$%
 * \verb+-+$\langle\mbox{GPS-seconds}\rangle$\verb+-+%
 * $\langle\mbox{duration}\rangle$\verb+.gwf+
 * where source and description are the specified frame source and description
 * identifiers, GPS-seconds is the start time of the data in
 * seconds since 0h UTC 6 Jan 1980 (GPS time origin), or the nearest second
 * before the start of the data, and duration is the number of seconds between
 * value of GPS-seconds and the GPS end time of the data, rounded up.
 *
 * \vfill{\footnotesize\input{FrameSeriesCV}}
 *
 **** </lalLaTeX> */


int rename( const char *from, const char *to );


#include <lal/LALFrameIO.h>

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <lal/LALStdio.h>
#include <lal/LALStdlib.h>
#include <lal/Date.h>
#include <lal/Units.h>
#include <lal/FrameCache.h>
#include <lal/FrameStream.h>
#include <lal/TimeSeries.h>
#include <lal/LALDetectors.h>
#include <lal/LALDatatypes.h>

NRCSID( FRAMESERIESC, "$Id$" );

/* Useful macros */
#define SECNAN_TO_I8TIME( sec, nan ) \
  ((INT8)1000000000*(INT8)(sec)+(INT8)(nan))
/* Dangerous!!!: */
#define EPOCH_TO_I8TIME( epoch ) \
  SECNAN_TO_I8TIME( (epoch).gpsSeconds, (epoch).gpsNanoSeconds )
#define SET_EPOCH( pepoch, i8time ) \
  do { INT8 t=(i8time); LIGOTimeGPS *pe=(pepoch); \
    pe->gpsSeconds=t/(INT8)1000000000; pe->gpsNanoSeconds=t%(INT8)1000000000; \
  } while( 0 )


/*
 *
 * Routine to load a FrVect associated with a given channel from a given frame.
 *
 */
static struct FrVect *loadFrVect( FrStream *stream, const char *channel )
{
  char        chan[256];
  FrAdcData  *adc;
  FrSimData  *sim;
  FrProcData *proc;
  FrChanType  type;
  FrVect     *vect;
  FrTOCts    *ts;
  double      tstart;
  strncpy( chan, channel, sizeof( chan ) - 1 );
  if ( ! stream->file->toc )
  {
    if ( FrTOCReadFull( stream->file ) == 0 )
    {
      stream->state |= LAL_FR_ERR | LAL_FR_TOC;
      return NULL;
    }
  }
  stream->file->relocation = FR_NO;
  tstart  = stream->file->toc->GTimeS[stream->pos];
  tstart += 1e-9 * stream->file->toc->GTimeN[stream->pos];
  ts = stream->file->toc->adc;
  /* scan adc data channels */
  type = LAL_ADC_CHAN;
  while ( ts && strcmp( channel, ts->name ) )
    ts = ts->next;
  if ( ! ts )
  {
    /* scan sim data channels */
    type = LAL_SIM_CHAN;
    ts = stream->file->toc->sim;
    while ( ts && strcmp( channel, ts->name ) )
      ts = ts->next;
  }
  if ( ! ts )
  {
    /* scan proc data channels */
    type = LAL_PROC_CHAN;
    ts = stream->file->toc->proc;
    while ( ts && strcmp( channel, ts->name ) )
      ts = ts->next;
  }
  if ( ! ts )
    return NULL;

  FrTOCSetPos( stream->file, ts->position[stream->pos] );
  switch ( type )
  {
    case LAL_ADC_CHAN:
      adc = FrAdcDataRead( stream->file );
      if ( ! adc ) /* only happens if memory allocation error */
        return NULL;
      adc->next = NULL;
      vect = FrVectReadNext( stream->file, tstart, chan );
      FrAdcDataFree( adc );
      break;
    case LAL_SIM_CHAN:
      sim = FrSimDataRead( stream->file );
      if ( ! sim ) /* only happens if memory allocation error */
        return NULL;
      sim->next = NULL;
      vect = FrVectReadNext( stream->file, tstart, chan );
      FrSimDataFree( sim );
      break;
    case LAL_PROC_CHAN:
      proc = FrProcDataRead( stream->file );
      if ( ! proc ) /* only happens if memory allocation error */
        return NULL;
      proc->next = NULL;
      vect = FrVectReadNext( stream->file, tstart, chan );
      FrProcDataFree( proc );
      break;
    default:
      return NULL;
  }
  if ( vect && vect->compress )
    FrVectExpand( vect );
  return vect;
}


/*
 *
 * Routine to make a 1D FrVect structure and attach it to a frame as the
 * appropriate channel type.
 *
 */
static struct FrVect *makeFrVect1D( struct FrameH *frame, int chtype,
    char *name, char *comment, char *unitx, char *unity, int datatype,
    double rate, double fshift, double dx, unsigned int npts )
{
  struct FrVect *vect = FrVectNew1D( name, datatype, npts, dx, unitx, unity );
  if ( ! vect ) return NULL;
  if ( chtype == ProcDataChannel )
  {
    struct FrProcData *proc = calloc( 1, sizeof( *proc ) );
    if ( ! proc )
    {
      FrVectFree( vect );
      return NULL;
    }
    proc->classe = FrProcDataDef();
    proc->fShift = fshift;
    proc->type = 1; /* time series */
    proc->data = vect;
    proc->next = frame->procData;
    frame->procData = proc;
    if ( ! FrStrCpy( &proc->name, name ) ) return NULL;
    if ( ! FrStrCpy( &proc->comment, comment ) ) return NULL;
  }
  else if ( chtype == ADCDataChannel )
  {
    struct FrAdcData *adc = calloc( 1, sizeof( *adc ) );
    struct FrRawData *raw = frame->rawData ? frame->rawData :
      FrRawDataNew( frame );
    if ( ! adc || ! raw )
    {
      FrVectFree( vect );
      if ( adc ) free( adc );
      return NULL;
    }
    adc->classe = FrAdcDataDef();
    adc->sampleRate = rate;
    adc->fShift = fshift;
    adc->data = vect;
    adc->next = raw->firstAdc;
    raw->firstAdc = adc;
    if ( ! FrStrCpy( &adc->name, name ) ) return NULL;
    if ( ! FrStrCpy( &adc->comment, comment ) ) return NULL;
    if ( ! FrStrCpy( &adc->units, unity ) ) return NULL;
  }
  else if ( chtype == SimDataChannel )
  {
    struct FrSimData *sim = calloc( 1, sizeof( *sim ) );
    if ( ! sim )
    {
      FrVectFree( vect );
      return NULL;
    }
    sim->classe = FrSimDataDef();
    sim->sampleRate = rate;
    sim->data = vect;
    sim->next = frame->simData;
    frame->simData = sim;
    if ( ! FrStrCpy( &sim->name, name ) ) return NULL;
    if ( ! FrStrCpy( &sim->comment, comment ) ) return NULL;
  }
  else
  {
    FrVectFree( vect );
    return NULL;
  }
  return vect;
}

static FrVect * FrVectReadInfo( FrFile *iFile, FRULONG *pos )
{
  FrVect *v;
  unsigned short type;
  FRULONG localpos;
  if ( FrFileIGoToNextRecord( iFile ) != iFile->vectorType )
    return NULL;
  v = calloc( 1, sizeof( FrVect ) );
  if ( ! v )
  {
    iFile->error = FR_ERROR_MALLOC_FAILED;
    return NULL;
  }
  FrReadHeader( iFile, v );
  FrReadSChar( iFile, &v->name );
  FrReadShortU( iFile, &v->compress );
  if ( v->compress == 256 )
    v->compress = 0; /* we will swap bytes at reading time */
  FrReadShortU( iFile, &type );
  v->type = type;
  switch ( v->type )
  {
    case FR_VECT_4R:
      v->wSize = sizeof( float );
      break;
    case FR_VECT_8R:
      v->wSize = sizeof( double );
      break;
    case FR_VECT_C:
      v->wSize = sizeof( char );
      break;
    case FR_VECT_1U:
      v->wSize = sizeof( char );
      break;
    case FR_VECT_2S:
      v->wSize = sizeof( short );
      break;
    case FR_VECT_2U:
      v->wSize = sizeof( short );
      break;
    case FR_VECT_4S:
      v->wSize = sizeof( int );
      break;
    case FR_VECT_4U:
      v->wSize = sizeof( int );
      break;
    case FR_VECT_8S:
      v->wSize = sizeof( FRLONG );
      break;
    case FR_VECT_8U:
      v->wSize = sizeof( FRLONG );
      break;
    case FR_VECT_8C:
      v->wSize = 2 * sizeof( float );
      break;
    case FR_VECT_16C:
      v->wSize = 2 * sizeof( double );
      break;
    case FR_VECT_8H:
      v->wSize = 2 * sizeof( float );
      break;
    case FR_VECT_16H:
      v->wSize = 2 * sizeof( double );
      break;
    default:
      v->wSize = 0;
  }
  if ( iFile->fmtVersion > 5 )
  {
    FrReadLong( iFile, (FRLONG *)&v->nData );
    FrReadLong( iFile, (FRLONG *)&v->nBytes );
  }
  else
  {
    unsigned int nData, nBytes;
    FrReadIntU( iFile, &nData );
    FrReadIntU( iFile, &nBytes );
    v->nData  = nData;
    v->nBytes = nBytes;
  }
  v->space = v->nData;

  /* skip the data */
  localpos = FrIOTell( iFile->frfd );
  if ( pos )
    *pos = localpos;
  localpos += v->nBytes;
  FrIOSet( iFile->frfd, localpos );
  FrReadIntU( iFile, &v->nDim );
  FrReadVL( iFile, (FRLONG**)&v->nx, v->nDim );
  FrReadVD( iFile, &v->dx, v->nDim );
  FrReadVD( iFile, &v->startX, v->nDim );
  FrReadVQ( iFile, &v->unitX, v->nDim );
  FrReadSChar( iFile, &v->unitY );
  FrReadStruct( iFile, &v->next );

  if ( pos )
    *pos = localpos;
  return v;
}

LALTYPECODE XLALFrGetTimeSeriesType( const char *channel, FrStream *stream )
{
  static const char func[] = "XLALFrGetTimeSeriesType";
  FrChanType chantype;
  FrTOCts    *ts   = NULL;
  FrProcData *proc = NULL;
  FrAdcData  *adc  = NULL;
  FrSimData  *sim  = NULL;
  int type = -1;

  if ( ! channel || ! stream )
    XLAL_ERROR( func, XLAL_EFAULT );
  if ( stream->state & LAL_FR_ERR )
    XLAL_ERROR( func, XLAL_EIO );
  if ( stream->state & LAL_FR_END )
    XLAL_ERROR( func, XLAL_EIO );

  if ( ! stream->file->toc )
  {
    if ( FrTOCReadFull( stream->file ) == NULL )
    {
      XLALPrintError( "XLAL Error - %s: could not open frame TOC %s\n",
          func, stream->file );
      stream->state |= LAL_FR_ERR | LAL_FR_TOC;
      XLAL_ERROR( func, XLAL_EIO );
    }
  }

  /* scan adc channels */
  chantype = LAL_ADC_CHAN;
  ts = stream->file->toc->adc;
  while ( ts && strcmp( channel, ts->name ) )
    ts = ts->next;

  if ( ! ts )
  {
    /* scan sim channels */
    chantype = LAL_SIM_CHAN;
    ts = stream->file->toc->sim;
    while ( ts && strcmp( channel, ts->name ) )
      ts = ts->next;
  }

  if ( ! ts )
  {
    /* scan proc channels */
    chantype = LAL_PROC_CHAN;
    ts = stream->file->toc->proc;
    while ( ts && strcmp( channel, ts->name ) )
      ts = ts->next;
  }

  if ( ts ) /* the channel was found */
  {
    FrTOCSetPos( stream->file, ts->position[0] );
    switch ( chantype )
    {
      case LAL_ADC_CHAN:
        if ( ( adc = FrAdcDataRead( stream->file ) ) )
        {
          adc->data = FrVectReadInfo( stream->file, NULL );
          type = adc->data ? adc->data->type : -1;
          FrAdcDataFree( adc );
        }
        break;
      case LAL_SIM_CHAN:
        if ( ( sim = FrSimDataRead( stream->file ) ) )
        {
          sim->data = FrVectReadInfo( stream->file, NULL );
          type = sim->data ? sim->data->type : -1;
          FrSimDataFree( sim );
        }
        break;
      case LAL_PROC_CHAN:
        if ( ( proc = FrProcDataRead( stream->file ) ) )
        {
          proc->data = FrVectReadInfo( stream->file, NULL );
          type = proc->data ? proc->data->type : -1;
          FrProcDataFree( proc );
        }
        break;
      default:
        type = -1;
    }
  }

  switch ( type )
  {
    case FR_VECT_C:
      return LAL_CHAR_TYPE_CODE;
    case FR_VECT_2S:
      return LAL_I2_TYPE_CODE;
    case FR_VECT_4S:
      return LAL_I4_TYPE_CODE;
    case FR_VECT_8S:
      return LAL_I8_TYPE_CODE;
    case FR_VECT_1U:
      return LAL_UCHAR_TYPE_CODE;
    case FR_VECT_2U:
      return LAL_U2_TYPE_CODE;
    case FR_VECT_4U:
      return LAL_U4_TYPE_CODE;
    case FR_VECT_8U:
      return LAL_U8_TYPE_CODE;
    case FR_VECT_4R:
      return LAL_S_TYPE_CODE;
    case FR_VECT_8R:
      return LAL_D_TYPE_CODE;
    case FR_VECT_C8:
      return LAL_C_TYPE_CODE;
    case FR_VECT_C16:
      return LAL_Z_TYPE_CODE;
    default:
      XLAL_ERROR( func, XLAL_ETYPE );
  }

  XLAL_ERROR( func, XLAL_ETYPE );
}



/* little helper function for getting number of points in a channel */
int XLALFrGetVectorLength ( CHAR *name, FrStream *stream )
{
  static const char func[] = "XLALFrGetVectorLength";
  struct FrVect	*vect;
  int ret = -1;

  if ( stream->state & LAL_FR_ERR )
    XLAL_ERROR( func, XLAL_EIO );
  if ( stream->state & LAL_FR_END )
    XLAL_ERROR( func, XLAL_EIO );

  vect = loadFrVect( stream, name );
  if ( ! vect || ! vect->data )
    XLAL_ERROR( func, XLAL_ENAME ); /* couldn't find channel */

  ret = vect->nData;

  FrVectFree(vect);

  return ret;

}


/* <lalVerbatim file="FrameSeriesCP"> */
void
LALFrGetTimeSeriesType(
    LALStatus   *status,
    LALTYPECODE *output,
    FrChanIn    *chanin,
    FrStream    *stream
    )
{ /* </lalVerbatim> */
  int type;
  INITSTATUS( status, "LALFrGetTimeSeriesType", FRAMESERIESC );

  ASSERT( output, status, FRAMESTREAMH_ENULL, FRAMESTREAMH_MSGENULL );
  ASSERT( stream, status, FRAMESTREAMH_ENULL, FRAMESTREAMH_MSGENULL );
  ASSERT( chanin, status, FRAMESTREAMH_ENULL, FRAMESTREAMH_MSGENULL );
  ASSERT( chanin->name, status, FRAMESTREAMH_ENULL, FRAMESTREAMH_MSGENULL );

  if ( stream->state & LAL_FR_ERR )
  {
    ABORT( status, FRAMESTREAMH_ERROR, FRAMESTREAMH_MSGERROR );
  }
  if ( stream->state & LAL_FR_END )
  {
    ABORT( status, FRAMESTREAMH_EDONE, FRAMESTREAMH_MSGEDONE );
  }

  type = XLALFrGetTimeSeriesType( chanin->name, stream );
  if ( type < 0 )
  {
    int errnum = xlalErrno;
    XLALClearErrno();
    switch ( errnum )
    {
      case XLAL_EIO:
        ABORT( status, FRAMESTREAMH_EREAD, FRAMESTREAMH_MSGEREAD );
      case XLAL_ETYPE:
        ABORT( status, FRAMESTREAMH_ETYPE, FRAMESTREAMH_MSGETYPE );
      default:
        ABORTXLAL( status );
    }
  }

  *output = type;
  RETURN( status );
}


static int copy_FrVect_to_REAL8( REAL8 *data, struct FrVect *vect, size_t ncpy, size_t offset )
{
	size_t i;
	if ( ncpy + offset > vect->nData )
		return -1;
	switch ( vect->type ) {
		case FR_VECT_C:
			for ( i = 0; i < ncpy; ++i )
			       	data[i] = vect->data[offset + i];
			break;
		case FR_VECT_1U:
			for ( i = 0; i < ncpy; ++i )
			       	data[i] = vect->dataU[offset + i];
			break;
		case FR_VECT_2U:
			for ( i = 0; i < ncpy; ++i )
			       	data[i] = vect->dataUS[offset + i];
			break;
		case FR_VECT_4U:
			for ( i = 0; i < ncpy; ++i )
			       	data[i] = vect->dataUI[offset + i];
			break;
		case FR_VECT_8U:
			for ( i = 0; i < ncpy; ++i )
			       	data[i] = vect->dataUL[offset + i];
			break;
		case FR_VECT_2S:
			for ( i = 0; i < ncpy; ++i )
			       	data[i] = vect->dataS[offset + i];
			break;
		case FR_VECT_4S:
			for ( i = 0; i < ncpy; ++i )
			       	data[i] = vect->dataI[offset + i];
			break;
		case FR_VECT_8S:
			for ( i = 0; i < ncpy; ++i )
			       	data[i] = vect->dataL[offset + i];
			break;
		case FR_VECT_4R:
			for ( i = 0; i < ncpy; ++i )
			       	data[i] = vect->dataF[offset + i];
			break;
		case FR_VECT_8R:
			for ( i = 0; i < ncpy; ++i )
			       	data[i] = vect->dataD[offset + i];
			break;
		default:
			return -1;
	}
	return ncpy;
}

REAL8TimeSeries * XLALFrInputREAL8TimeSeries( FrStream *stream, const char *channel, const LIGOTimeGPS *start, REAL8 duration, size_t lengthlimit )
{
	static const char func[] = "XLALFrInputREAL8TimeSeries";
	const REAL8 fuzz = 0.1 / 16384.0; /* smallest discernable unit of time */
	struct FrVect *vect;
	REAL8TimeSeries *series;
	LIGOTimeGPS epoch;
	UINT4  need;
	UINT4  noff;
	UINT4  ncpy;
	REAL8 *dest;
	INT8   tnow;
	INT8   tbeg;
	INT8   tend;
	REAL8  rate;
	int    code;

	if ( stream->state & LAL_FR_ERR )
		XLAL_ERROR_NULL( func, XLAL_EIO );
	if ( stream->state & LAL_FR_END )
		XLAL_ERROR_NULL( func, XLAL_EIO );

	/* seek to the correct place */
	if ( XLALFrSeek( stream, start ) )
		XLAL_ERROR_NULL( func, XLAL_EIO );

	vect = loadFrVect( stream, channel );
	if ( ! vect || ! vect->data )
		XLAL_ERROR_NULL( func, XLAL_ENAME ); /* couldn't find channel */

	tnow = EPOCH_TO_I8TIME( stream->epoch );
#	if defined FR_VERS && FR_VERS >= 5000
	tbeg = 1e9 * vect->GTime;
#	else
	tbeg = SECNAN_TO_I8TIME( vect->GTimeS, vect->GTimeN );
#	endif
	if ( tnow + 1000 < tbeg ) { /* added 1000 ns to account for double precision */
		FrVectFree(vect);
		XLAL_ERROR_NULL( func, XLAL_ETIME ); /* invalid time offset */
	}

	/* compute number of points offset very carefully:
	 * if current time is within fuzz of a sample, get that sample;
	 * otherwise get the sample just after the requested time */
	rate = 1.0 / vect->dx[0];
	noff = ceil( ( 1e-9 * ( tnow - tbeg ) - fuzz ) * rate );
	if ( noff > vect->nData ) {
		FrVectFree(vect);
		XLAL_ERROR_NULL( func, XLAL_ETIME ); /* invalid time offset */
	}
	need = duration * rate;
	if ( lengthlimit && (lengthlimit < need) )
		need = lengthlimit;

	/* adjust current time to be exactly the first sample
	 * (rounded to nearest nanosecond) */
	tnow = tbeg + floor( 1e9 * noff * vect->dx[0] + 0.5 );
	SET_EPOCH( &epoch, tnow );

	series = XLALCreateREAL8TimeSeries( channel, &epoch, 0.0, vect->dx[0], &lalADCCountUnit, need );
	if ( ! series )
		XLAL_ERROR_NULL( func, XLAL_EFUNC );
	dest = series->data->data;

	/* number of points to copy */
	ncpy = ( vect->nData - noff < need ) ? ( vect->nData - noff ) : need;
	code = copy_FrVect_to_REAL8( dest, vect, ncpy, noff );
	if ( code < 0 ) { /* fails if vect has complex data type */
		if(vect) FrVectFree(vect);
		XLALDestroyREAL8TimeSeries( series );
		XLAL_ERROR_NULL( func, XLAL_ETYPE ); /* data has wrong type */
	}

	FrVectFree(vect);
	vect = NULL;

	dest += ncpy;
	need -= ncpy;

	/* if still data remaining */
	while ( need ) {
		if ( XLALFrNext( stream ) < 0 ) {
			if(vect) FrVectFree(vect);
			XLALDestroyREAL8TimeSeries( series );
			XLAL_ERROR_NULL( func, XLAL_EFUNC );
		}
		if ( stream->state & LAL_FR_END ) {
			if(vect) FrVectFree(vect);
			XLALDestroyREAL8TimeSeries( series );
			XLAL_ERROR_NULL( func, XLAL_EIO );
		}

		/* load more data */
		vect = loadFrVect( stream, series->name );
		if ( ! vect || ! vect->data ) { /* channel missing */
			XLALDestroyREAL8TimeSeries( series );
			XLAL_ERROR_NULL( func, XLAL_ENAME );
		}

		if ( stream->state & LAL_FR_GAP ) { /* failure: gap in data */
			if(vect) FrVectFree(vect);
			XLALDestroyREAL8TimeSeries( series );
			XLAL_ERROR_NULL( func, XLAL_ETIME );
		}

		/* copy data */
		ncpy = vect->nData < need ? vect->nData : need;
		code = copy_FrVect_to_REAL8( dest, vect, ncpy, 0 );
		if ( code < 0 ) { /* fails if vect has complex data type */
			if(vect) FrVectFree(vect);
			XLALDestroyREAL8TimeSeries( series );
			XLAL_ERROR_NULL( func, XLAL_ETYPE );
		}

		FrVectFree(vect);
		vect=NULL;

		dest += ncpy;
		need -= ncpy;
	}

	/* update stream start time very carefully:
	 * start time must be the exact time of the next sample, rounded to the
	 * nearest nanosecond */
	SET_EPOCH( &stream->epoch, EPOCH_TO_I8TIME( series->epoch ) + (INT8)floor( 1e9 * series->data->length * series->deltaT + 0.5 ) );

	/* check to see that we are still within current frame */
	tend  = SECNAN_TO_I8TIME( stream->file->toc->GTimeS[stream->pos], stream->file->toc->GTimeN[stream->pos] );
	tend += (INT8)floor( 1e9 * stream->file->toc->dt[stream->pos] );
	if ( tend <= EPOCH_TO_I8TIME( stream->epoch ) ) {
		int keepmode = stream->mode;
		LIGOTimeGPS keep;
		keep = stream->epoch;
		/* advance a frame -- failure is benign */
		stream->mode |= LAL_FR_IGNOREGAP_MODE;
		XLALFrNext( stream );
		if ( ! stream->state & LAL_FR_GAP )
			stream->epoch = keep;
		stream->mode = keepmode;
	}

	if ( stream->state & LAL_FR_ERR ) {
		XLALDestroyREAL8TimeSeries( series );
		XLAL_ERROR_NULL( func, XLAL_EIO );
	}

	return series;
}


define(`TYPE',`COMPLEX16')
include(`FrameSeriesRead.m4')
include(`FrameSeriesWrite.m4')

define(`TYPE',`COMPLEX8')
include(`FrameSeriesRead.m4')
include(`FrameSeriesWrite.m4')

define(`TYPE',`REAL8')
include(`FrameSeriesRead.m4')
include(`FrameSeriesWrite.m4')

define(`TYPE',`REAL4')
include(`FrameSeriesRead.m4')
include(`FrameSeriesWrite.m4')

define(`TYPE',`INT8')
include(`FrameSeriesRead.m4')
include(`FrameSeriesWrite.m4')

define(`TYPE',`INT4')
include(`FrameSeriesRead.m4')
include(`FrameSeriesWrite.m4')

define(`TYPE',`INT2')
include(`FrameSeriesRead.m4')
include(`FrameSeriesWrite.m4')

/*
*  Copyright (C) 2007 Andres C. Rodriguez, Sukanta Bose, Alexander Dietz, Duncan Brown, Jolien Creighton, Kipp Cannon, Lisa M. Goggin, Patrick Brady, Robert Adam Mercer, Saikat Ray-Majumder, Anand Sengupta, Stephen Fairhurst, Xavier Siemens, Sean Seader, Thomas Cokelaer
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

/*----------------------------------------------------------------------- 
 * 
 * File Name: LIGOLwXML.c
 *
 * Author: Brown, D. A.
 * 
 * Revision: $Id$
 * 
 *-----------------------------------------------------------------------
 */

#if 0
<lalVerbatim file="LIGOLwXMLCV">
Author: Brown, D. A.
$Id$
</lalVerbatim> 
#endif

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <lal/LALStdlib.h>
#include <lal/LALStdio.h>
#include <lal/FileIO.h>
#include <lal/LALVersion.h>
#include <lal/LIGOMetadataTables.h>
#include <lal/LIGOLwXML.h>
#include <lal/XLALError.h>

#ifdef fputs
#	undef fputs
#endif
#define fputs XLALFilePuts
#ifdef fprintf
#	undef fprintf
#endif
#define fprintf XLALFilePrintf
#include <lal/LIGOLwXMLHeaders.h>
#include <lal/LIGOLwXMLInspiralHeaders.h>


NRCSID( LIGOLWXMLC, "$Id$" );

#if 0
<lalLaTeX>
\subsection{Module \texttt{LIGOLwXML.c}}

Routines to write LIGO metadata database structures to LIGO lightweight XML
files.

\subsubsection*{Prototypes}
\input{LIGOLwXMLCP}
\idx{LALOpenLIGOLwXMLFile()}
\idx{LALCloseLIGOLwXMLFile()}
\idx{LALBeginLIGOLwXMLTable()}
\idx{LALEndLIGOLwXMLTable()}
\idx{LALWriteLIGOLwXMLTable()}

\subsubsection*{Description}

The routine \verb+LALOpenLIGOLwXMLFile+ calls the C standard library function
\verb+fopen+ to open a file specified by the \verb+path+ argument. The file is
truncated to zero length if already exists. The standard LIGO lightweight XML
header, \verb+LIGOLW_XML_HEADER+ given in LIGOLwXMLHeaders.h, is then written
to the file and the the pointer to the file stream is returned in the
\verb+xml->fp+ argument.

The routine \verb+LALCloseLIGOLwXMLFile+ prints the standard LIGO lightweight
XML footer, \verb+LIGOLW_XML_FOOTER+ given in LIGOLwXMLHeaders.h, and closes
the file stream pointed to by \verb+xml->fp+.

The routine \verb+LALBeginLIGOLwXMLTable+ prints the table header.  The type of
table to begin is specified by the \verb+table+ argument.  The appropriate
headers are again contained in LIGOLwXMLHeaders.h and contain the table name as
well as the names and data types of each of the columns in the table.  In
addition, it sets \verb+xml->first+ to 1 and \verb+xml->table+ to the requested
table. 

The routine \verb+LALEndLIGOLwXMLTable+ prints the table footer.  This is the
same for all tables, and given by \verb+LIGOLW_XML_TABLE_FOOTER+ in
LIGOLwXMLHeaders.h.  Additionally, \verb+xml->table+ is set to \verb+no_table+.

The routine \verb+LALWriteLIGOLwXMLTable+ writes the content of the xml table.
The type of table to be written is specified by \verb+table+.  The contents of
the table should be stored as a linked list in \verb+tablePtr->table+.  The data
is written using the row format for the specified table given in
LIGOLwXMLHeaders.h. 


\subsubsection*{Algorithm}

None.

\subsubsection*{Uses}

\verb+fopen()+
\verb+fprintf()+
\verb+fclose()+

\subsubsection*{Notes}

In order to change a table definition in LAL, changes must be made in
several places.  It is necessary to update the structure which is used to store
the information in memory as well as the reading and writing codes.  Below is a
list of all the files which must be updated.
\begin{itemize}
\item  Update the LAL table definition in \verb+LIGOMetaDataTables.h+

\item  Update the LIGOLwXML writing code:

\begin{enumerate}
\item  Change the table header written at to the LIGOLwXML file.  This is
\verb+#define+d in \verb+LIGOLwXMLHeaders.h+.  For example, to change the 
\verb+sngl_inspiral+ table, you must edit \verb+LIGOLW_XML_SNGL_INSPIRAL+.

\item Change the row format of the LIGOLwXML file.  This is \verb+#define+d in
\verb+LIGOLwXMLHeaders.h+.  For example, to change the \verb+ sngl_inspiral+
table, you must edit \verb+SNGL_INSPIRAL_ROW+.

\item Change the fprintf command which writes the table rows.  This is contained
in \verb+LIGOLwXML.c+.  

\end{enumerate}

\item Update the LIGOLwXML reading code:

\begin{enumerate}

\item Add/remove columns from the table directory of the table in question.
This is contained in \verb+LIGOLwXMLRead.c+, either in
\verb+LALCreateMetaTableDir+ or in the specific reading function.

\item Check that all columns read in from the XML table are stored in memory.
This requires editing the table specific reading codes in
\verb+LIGOLwXMLRead.c+.

\end{enumerate}

\end{itemize}


\vfill{\footnotesize\input{LIGOLwXMLCV}}

</lalLaTeX>
#endif

/* JC: ISO C89 COMPILERS ARE REQUIRED TO SUPPORT STRINGS UP TO 509 CHARS LONG;
 * MANY OF THE STRINGS IN THE ORIGINAL MACROS WERE LONGER.  TO FIX I CHANGED
 * THE ORIGINAL MACROS TO BE A SERIES OF FPUTS COMMANDS AND PREFIXED THEM
 * WITH PRINT_.  I RENAMED FPRINTF TO MYFPRINTF AND THEN CREATED THIS MACRO
 * TO PREFIX THE PRINT_ TO THE PREVIOUS NAME SO IT EXPANDS AS THE NEW MACRO.
 *
 * IF YOU DON'T LIKE IT, FIX IT, BUT MAKE SURE THAT THE STRINGS ARE SMALLER
 * THAN 509 CHARS.
 */
#define myfprintf(fp,oldmacro) PRINT_ ## oldmacro(fp)

/* <lalVerbatim file="LIGOLwXMLCP"> */
void
LALOpenLIGOLwXMLFile (
    LALStatus          *status,
    LIGOLwXMLStream    *xml,
    const CHAR         *path
    )
/* </lalVerbatim> */
{
  /*  open the file and print the xml header */
  INITSTATUS( status, "LALOpenLIGOLwXMLFile", LIGOLWXMLC );
  ASSERT( xml, status, LIGOLWXMLH_ENULL, LIGOLWXMLH_MSGENULL );
  ASSERT( ! xml->fp, status, LIGOLWXMLH_ENNUL, LIGOLWXMLH_MSGENNUL );
  xml->fp = XLALFileOpen( path, "w" );
  if ( ! xml->fp )
  {
    ABORT( status, LIGOLWXMLH_EOPEN, LIGOLWXMLH_MSGEOPEN );
  }
  myfprintf( xml->fp, LIGOLW_XML_HEADER );
  xml->table = no_table;
  RETURN( status );
}

/* <lalVerbatim file="LIGOLwXMLCP"> */
void
LALCloseLIGOLwXMLFile (
    LALStatus          *status,
    LIGOLwXMLStream    *xml
    )
/* </lalVerbatim> */
{
  /* print the xml footer and close the file handle */
  INITSTATUS( status, "LALCloseLIGOLwXMLFile", LIGOLWXMLC );
  ASSERT( xml, status, LIGOLWXMLH_ENULL, LIGOLWXMLH_MSGENULL );
  ASSERT( xml->fp, status, LIGOLWXMLH_ENULL, LIGOLWXMLH_MSGENULL );
  if ( xml->table != no_table )
  {
    ABORT( status, LIGOLWXMLH_ECLOS, LIGOLWXMLH_MSGECLOS );
  }
  myfprintf( xml->fp, LIGOLW_XML_FOOTER );
  XLALFileClose( xml->fp );
  xml->fp = NULL;
  RETURN( status );
}

/* <lalVerbatim file="LIGOLwXMLCP"> */
void
LALBeginLIGOLwXMLTable (
    LALStatus           *status,
    LIGOLwXMLStream     *xml,
    MetadataTableType    table
    )
/* </lalVerbatim> */
{
  /* print the header for the xml table */
  INITSTATUS( status, "LALBeginLIGOLwXMLTable", LIGOLWXMLC );
  ASSERT( xml, status, LIGOLWXMLH_ENULL, LIGOLWXMLH_MSGENULL );
  ASSERT( xml->fp, status, LIGOLWXMLH_ENULL, LIGOLWXMLH_MSGENULL );
  if ( xml->table != no_table )
  {
    ABORT( status, LIGOLWXMLH_EBGNT, LIGOLWXMLH_MSGEBGNT );
  }

  switch( table )
  {
    case no_table:
      ABORT( status, LIGOLWXMLH_ENTAB, LIGOLWXMLH_MSGENTAB );
      break;
    case process_table:
      myfprintf( xml->fp, LIGOLW_XML_PROCESS );
      break;
    case process_params_table:
      myfprintf( xml->fp, LIGOLW_XML_PROCESS_PARAMS );
      break;
    case search_summary_table:
      myfprintf( xml->fp, LIGOLW_XML_SEARCH_SUMMARY );
      break;
    case search_summvars_table:
      myfprintf( xml->fp, LIGOLW_XML_SEARCH_SUMMVARS );
      break;
    case sngl_burst_table:
      myfprintf( xml->fp, LIGOLW_XML_SNGL_BURST );
      break;
    case sngl_inspiral_table:
      myfprintf( xml->fp, LIGOLW_XML_SNGL_INSPIRAL );
      break;
    case sngl_inspiral_table_bns:
      myfprintf( xml->fp, LIGOLW_XML_SNGL_INSPIRAL_BNS );
      break;
    case sngl_inspiral_table_bcv:
      myfprintf( xml->fp, LIGOLW_XML_SNGL_INSPIRAL_BCV );
      break;
    case sngl_ringdown_table:
      myfprintf( xml->fp, LIGOLW_XML_SNGL_RINGDOWN );
      break;
    case multi_inspiral_table:
      myfprintf( xml->fp, LIGOLW_XML_MULTI_INSPIRAL );
      break;
    case sim_inspiral_table:
      myfprintf( xml->fp, LIGOLW_XML_SIM_INSPIRAL );
      break;
    case sim_ringdown_table:
      myfprintf( xml->fp, LIGOLW_XML_SIM_RINGDOWN );
      break;
    case summ_value_table:
      myfprintf( xml->fp, LIGOLW_XML_SUMM_VALUE );
      break;
    case sim_inst_params_table:
      myfprintf( xml->fp, LIGOLW_XML_SIM_INST_PARAMS );
      break;
    case stochastic_table:
      myfprintf( xml->fp, LIGOLW_XML_STOCHASTIC );
      break;
    case ext_triggers_table:
      myfprintf( xml->fp, LIGOLW_XML_EXT_TRIGGERS);
      break;
    case filter_table:
      myfprintf( xml->fp, LIGOLW_XML_FILTER );
      break;
    default:
      ABORT( status, LIGOLWXMLH_EUTAB, LIGOLWXMLH_MSGEUTAB );
  }
  xml->first = 1;
  xml->rowCount = 0;
  xml->table = table;
  RETURN( status );
}

/* <lalVerbatim file="LIGOLwXMLCP"> */
void
LALEndLIGOLwXMLTable (
    LALStatus           *status,
    LIGOLwXMLStream     *xml
    )
/* </lalVerbatim> */
{
  /* print the header for the xml table */
  INITSTATUS( status, "LALEndLIGOLwXMLTable", LIGOLWXMLC );
  ASSERT( xml, status, LIGOLWXMLH_ENULL, LIGOLWXMLH_MSGENULL );
  ASSERT( xml->fp, status, LIGOLWXMLH_ENULL, LIGOLWXMLH_MSGENULL );
  if ( xml->table == no_table )
  {
    ABORT( status, LIGOLWXMLH_EENDT, LIGOLWXMLH_MSGEENDT );
  }
  myfprintf( xml->fp, LIGOLW_XML_TABLE_FOOTER );
  xml->table = no_table;
  RETURN( status );
}

/* macro to print a comma on subsequent table rows */
#define FIRST_TABLE_ROW \
  if ( xml->first ) \
{ \
  xml->first = 0; \
} else \
{ \
  fprintf( xml->fp, ",\n" ); \
}

/* <lalVerbatim file="LIGOLwXMLCP"> */
void
LALWriteLIGOLwXMLTable (
    LALStatus           *status,
    LIGOLwXMLStream     *xml,
    MetadataTable        tablePtr,
    MetadataTableType    table
    )
/* </lalVerbatim> */
{
  /* print contents of the database struct into the xml table */
  INITSTATUS( status, "LALWriteLIGOLwXMLTable", LIGOLWXMLC );
  ASSERT( xml, status, LIGOLWXMLH_ENULL, LIGOLWXMLH_MSGENULL );
  ASSERT( xml->fp, status, LIGOLWXMLH_ENULL, LIGOLWXMLH_MSGENULL );
  if ( xml->table == no_table )
  {
    ABORT( status, LIGOLWXMLH_ETNOP, LIGOLWXMLH_MSGETNOP );
  }
  if ( xml->table != table )
  {
    ABORT( status, LIGOLWXMLH_ETMSM, LIGOLWXMLH_MSGETMSM );
  }
  switch( table )
  {
    case no_table:
      ABORT( status, LIGOLWXMLH_ENTAB, LIGOLWXMLH_MSGENTAB );
      break;
    case process_table:
      while( tablePtr.processTable )
      {
        FIRST_TABLE_ROW
          fprintf( xml->fp, PROCESS_ROW,
              tablePtr.processTable->program,
              tablePtr.processTable->version,
              tablePtr.processTable->cvs_repository,
              tablePtr.processTable->cvs_entry_time.gpsSeconds,
              tablePtr.processTable->comment,
              tablePtr.processTable->is_online,
              tablePtr.processTable->node,
              tablePtr.processTable->username,
              tablePtr.processTable->unix_procid,
              tablePtr.processTable->start_time.gpsSeconds,
              tablePtr.processTable->end_time.gpsSeconds,
              tablePtr.processTable->jobid,
              tablePtr.processTable->domain,
              tablePtr.processTable->ifos
              );
        tablePtr.processTable = tablePtr.processTable->next;
        ++(xml->rowCount);
      }
      break;
    case process_params_table:
      while( tablePtr.processParamsTable )
      {
        FIRST_TABLE_ROW
          fprintf( xml->fp, PROCESS_PARAMS_ROW,
              tablePtr.processParamsTable->program,
              tablePtr.processParamsTable->param,
              tablePtr.processParamsTable->type,
              tablePtr.processParamsTable->value
              );
        tablePtr.processParamsTable = tablePtr.processParamsTable->next;
        ++(xml->rowCount);
      }
      break;
    case search_summary_table:
      while( tablePtr.searchSummaryTable )
      {
        FIRST_TABLE_ROW
          fprintf( xml->fp, SEARCH_SUMMARY_ROW,
              lalCVSTag,
              tablePtr.searchSummaryTable->comment,
              tablePtr.searchSummaryTable->ifos,
              tablePtr.searchSummaryTable->in_start_time.gpsSeconds,
              tablePtr.searchSummaryTable->in_start_time.gpsNanoSeconds,
              tablePtr.searchSummaryTable->in_end_time.gpsSeconds,
              tablePtr.searchSummaryTable->in_end_time.gpsNanoSeconds,
              tablePtr.searchSummaryTable->out_start_time.gpsSeconds,
              tablePtr.searchSummaryTable->out_start_time.gpsNanoSeconds,
              tablePtr.searchSummaryTable->out_end_time.gpsSeconds,
              tablePtr.searchSummaryTable->out_end_time.gpsNanoSeconds,
              tablePtr.searchSummaryTable->nevents,
              tablePtr.searchSummaryTable->nnodes
              );
        tablePtr.searchSummaryTable = tablePtr.searchSummaryTable->next;
        ++(xml->rowCount);
      }
      break;
    case search_summvars_table:
      while( tablePtr.searchSummvarsTable )
      {
        FIRST_TABLE_ROW
          fprintf( xml->fp, SEARCH_SUMMVARS_ROW,
              tablePtr.searchSummvarsTable->name,
              tablePtr.searchSummvarsTable->string,
              tablePtr.searchSummvarsTable->value,
              xml->rowCount
              );
        tablePtr.searchSummvarsTable = tablePtr.searchSummvarsTable->next;
        ++(xml->rowCount);
      }
      break;
    case sngl_burst_table:
      while( tablePtr.snglBurstTable )
      {
        FIRST_TABLE_ROW
          fprintf( xml->fp, SNGL_BURST_ROW,
              tablePtr.snglBurstTable->ifo,
              tablePtr.snglBurstTable->search,
              tablePtr.snglBurstTable->channel,
              tablePtr.snglBurstTable->start_time.gpsSeconds,
              tablePtr.snglBurstTable->start_time.gpsNanoSeconds,
              tablePtr.snglBurstTable->peak_time.gpsSeconds,
              tablePtr.snglBurstTable->peak_time.gpsNanoSeconds,
              tablePtr.snglBurstTable->duration,
              tablePtr.snglBurstTable->central_freq,
              tablePtr.snglBurstTable->bandwidth,
              tablePtr.snglBurstTable->amplitude,
	      tablePtr.snglBurstTable->snr,
	      tablePtr.snglBurstTable->confidence,
	      tablePtr.snglBurstTable->event_id
              );
        tablePtr.snglBurstTable = tablePtr.snglBurstTable->next;
      }
      break;
    case sngl_inspiral_table:
      while( tablePtr.snglInspiralTable )
      {
        UINT8 id = 0;
        if ( tablePtr.snglInspiralTable->event_id )
        {
          id = tablePtr.snglInspiralTable->event_id->id;
        }
        FIRST_TABLE_ROW
          fprintf( xml->fp, SNGL_INSPIRAL_ROW,
              tablePtr.snglInspiralTable->ifo,
              tablePtr.snglInspiralTable->search,
              tablePtr.snglInspiralTable->channel,
              tablePtr.snglInspiralTable->end_time.gpsSeconds,
              tablePtr.snglInspiralTable->end_time.gpsNanoSeconds,
              tablePtr.snglInspiralTable->end_time_gmst,
              tablePtr.snglInspiralTable->impulse_time.gpsSeconds,
              tablePtr.snglInspiralTable->impulse_time.gpsNanoSeconds,
              tablePtr.snglInspiralTable->template_duration,
              tablePtr.snglInspiralTable->event_duration,
              tablePtr.snglInspiralTable->amplitude,
              tablePtr.snglInspiralTable->eff_distance,
              tablePtr.snglInspiralTable->coa_phase,
              tablePtr.snglInspiralTable->mass1,
              tablePtr.snglInspiralTable->mass2,
              tablePtr.snglInspiralTable->mchirp,
              tablePtr.snglInspiralTable->mtotal,
              tablePtr.snglInspiralTable->eta,
              tablePtr.snglInspiralTable->kappa,
              tablePtr.snglInspiralTable->chi,
              tablePtr.snglInspiralTable->tau0,
              tablePtr.snglInspiralTable->tau2,
              tablePtr.snglInspiralTable->tau3,
              tablePtr.snglInspiralTable->tau4,
              tablePtr.snglInspiralTable->tau5,
              tablePtr.snglInspiralTable->ttotal,
              tablePtr.snglInspiralTable->psi0,
              tablePtr.snglInspiralTable->psi3,
              tablePtr.snglInspiralTable->alpha,
              tablePtr.snglInspiralTable->alpha1,
              tablePtr.snglInspiralTable->alpha2,
              tablePtr.snglInspiralTable->alpha3,
              tablePtr.snglInspiralTable->alpha4,
              tablePtr.snglInspiralTable->alpha5,
              tablePtr.snglInspiralTable->alpha6,
              tablePtr.snglInspiralTable->beta,
              tablePtr.snglInspiralTable->f_final,
              tablePtr.snglInspiralTable->snr,
              tablePtr.snglInspiralTable->chisq,
              tablePtr.snglInspiralTable->chisq_dof,
              tablePtr.snglInspiralTable->bank_chisq,
              tablePtr.snglInspiralTable->bank_chisq_dof,
              tablePtr.snglInspiralTable->cont_chisq,
              tablePtr.snglInspiralTable->cont_chisq_dof,
              tablePtr.snglInspiralTable->sigmasq,
	      tablePtr.snglInspiralTable->rsqveto_duration,
	      tablePtr.snglInspiralTable->Gamma[0],
	      tablePtr.snglInspiralTable->Gamma[1],
	      tablePtr.snglInspiralTable->Gamma[2],
	      tablePtr.snglInspiralTable->Gamma[3],
	      tablePtr.snglInspiralTable->Gamma[4],
	      tablePtr.snglInspiralTable->Gamma[5],
	      tablePtr.snglInspiralTable->Gamma[6],
	      tablePtr.snglInspiralTable->Gamma[7],
	      tablePtr.snglInspiralTable->Gamma[8],
	      tablePtr.snglInspiralTable->Gamma[9],
              id );
        tablePtr.snglInspiralTable = tablePtr.snglInspiralTable->next;
        ++(xml->rowCount);
      }
      break;
    case sngl_inspiral_table_bns:
      while( tablePtr.snglInspiralTable )
      {
        FIRST_TABLE_ROW
          fprintf( xml->fp, SNGL_INSPIRAL_ROW_BNS,
              tablePtr.snglInspiralTable->ifo,
              tablePtr.snglInspiralTable->search,
              tablePtr.snglInspiralTable->channel,
              tablePtr.snglInspiralTable->end_time.gpsSeconds,
              tablePtr.snglInspiralTable->end_time.gpsNanoSeconds,
              tablePtr.snglInspiralTable->end_time_gmst,
              tablePtr.snglInspiralTable->template_duration,
              tablePtr.snglInspiralTable->eff_distance,
              tablePtr.snglInspiralTable->coa_phase,
              tablePtr.snglInspiralTable->mass1,
              tablePtr.snglInspiralTable->mass2,
              tablePtr.snglInspiralTable->mchirp,
              tablePtr.snglInspiralTable->mtotal,
              tablePtr.snglInspiralTable->eta,
              tablePtr.snglInspiralTable->tau0,
              tablePtr.snglInspiralTable->tau3,
              tablePtr.snglInspiralTable->ttotal,
              tablePtr.snglInspiralTable->f_final,
              tablePtr.snglInspiralTable->snr,
              tablePtr.snglInspiralTable->chisq,
              tablePtr.snglInspiralTable->chisq_dof,
              tablePtr.snglInspiralTable->bank_chisq,
              tablePtr.snglInspiralTable->bank_chisq_dof,
              tablePtr.snglInspiralTable->cont_chisq,
              tablePtr.snglInspiralTable->cont_chisq_dof,
              tablePtr.snglInspiralTable->sigmasq,
	      tablePtr.snglInspiralTable->rsqveto_duration );
        tablePtr.snglInspiralTable = tablePtr.snglInspiralTable->next;
        ++(xml->rowCount);
      }
      break;
    case sngl_inspiral_table_bcv:
      while( tablePtr.snglInspiralTable )
      {
        FIRST_TABLE_ROW
          fprintf( xml->fp, SNGL_INSPIRAL_ROW_BCV,
              tablePtr.snglInspiralTable->ifo,
              tablePtr.snglInspiralTable->search,
              tablePtr.snglInspiralTable->channel,
              tablePtr.snglInspiralTable->end_time.gpsSeconds,
              tablePtr.snglInspiralTable->end_time.gpsNanoSeconds,
              tablePtr.snglInspiralTable->end_time_gmst,
              tablePtr.snglInspiralTable->template_duration,
              tablePtr.snglInspiralTable->eff_distance,
              tablePtr.snglInspiralTable->coa_phase,
              tablePtr.snglInspiralTable->mchirp,
              tablePtr.snglInspiralTable->eta,
              tablePtr.snglInspiralTable->psi0,
              tablePtr.snglInspiralTable->psi3,
              tablePtr.snglInspiralTable->alpha,
              tablePtr.snglInspiralTable->f_final,
              tablePtr.snglInspiralTable->snr,
              tablePtr.snglInspiralTable->chisq,
              tablePtr.snglInspiralTable->chisq_dof,
              tablePtr.snglInspiralTable->bank_chisq,
              tablePtr.snglInspiralTable->bank_chisq_dof,
              tablePtr.snglInspiralTable->cont_chisq,
              tablePtr.snglInspiralTable->cont_chisq_dof,
              tablePtr.snglInspiralTable->sigmasq,
	      tablePtr.snglInspiralTable->rsqveto_duration );
        tablePtr.snglInspiralTable = tablePtr.snglInspiralTable->next;
        ++(xml->rowCount);
      }
      break;
    case sngl_ringdown_table:
      while( tablePtr.snglRingdownTable )
      {
        UINT8 id = xml->rowCount;
        if ( tablePtr.snglRingdownTable->event_id )
        {
          id = tablePtr.snglRingdownTable->event_id->id;
        }
        FIRST_TABLE_ROW
          fprintf( xml->fp, SNGL_RINGDOWN_ROW,
              tablePtr.snglRingdownTable->ifo,
              tablePtr.snglRingdownTable->channel,
              tablePtr.snglRingdownTable->start_time.gpsSeconds,
              tablePtr.snglRingdownTable->start_time.gpsNanoSeconds,
              tablePtr.snglRingdownTable->start_time_gmst,
              tablePtr.snglRingdownTable->frequency,
              tablePtr.snglRingdownTable->quality,
              tablePtr.snglRingdownTable->phase,
              tablePtr.snglRingdownTable->mass,
              tablePtr.snglRingdownTable->spin,
              tablePtr.snglRingdownTable->epsilon,
              tablePtr.snglRingdownTable->num_clust_trigs,
              tablePtr.snglRingdownTable->ds2_H1H2,
              tablePtr.snglRingdownTable->ds2_H1L1,
              tablePtr.snglRingdownTable->ds2_H2L1,
              tablePtr.snglRingdownTable->amplitude,
              tablePtr.snglRingdownTable->snr,
              tablePtr.snglRingdownTable->eff_dist,
              tablePtr.snglRingdownTable->sigma_sq,
              id
              );
        tablePtr.snglRingdownTable = tablePtr.snglRingdownTable->next;
        ++(xml->rowCount);
      }
      break;
    case multi_inspiral_table:
      while( tablePtr.multiInspiralTable )
      {
        FIRST_TABLE_ROW
          fprintf( xml->fp, MULTI_INSPIRAL_ROW,
              tablePtr.multiInspiralTable->ifos,
              tablePtr.multiInspiralTable->search,
              tablePtr.multiInspiralTable->end_time.gpsSeconds,
              tablePtr.multiInspiralTable->end_time.gpsNanoSeconds,
              tablePtr.multiInspiralTable->end_time_gmst,
              tablePtr.multiInspiralTable->impulse_time.gpsSeconds,
              tablePtr.multiInspiralTable->impulse_time.gpsNanoSeconds,
              tablePtr.multiInspiralTable->amplitude,
              tablePtr.multiInspiralTable->ifo1_eff_distance,
              tablePtr.multiInspiralTable->ifo2_eff_distance,
              tablePtr.multiInspiralTable->eff_distance,
              tablePtr.multiInspiralTable->coa_phase,
              tablePtr.multiInspiralTable->mass1,
              tablePtr.multiInspiralTable->mass2,
              tablePtr.multiInspiralTable->mchirp,
              tablePtr.multiInspiralTable->eta,
              tablePtr.multiInspiralTable->tau0,
              tablePtr.multiInspiralTable->tau2,
              tablePtr.multiInspiralTable->tau3,
              tablePtr.multiInspiralTable->tau4,
              tablePtr.multiInspiralTable->tau5,
              tablePtr.multiInspiralTable->ttotal,
              tablePtr.multiInspiralTable->ifo1_snr,
              tablePtr.multiInspiralTable->ifo2_snr,
              tablePtr.multiInspiralTable->snr,
              tablePtr.multiInspiralTable->chisq,
              tablePtr.multiInspiralTable->chisq_dof,
              tablePtr.multiInspiralTable->bank_chisq,
              tablePtr.multiInspiralTable->bank_chisq_dof,
              tablePtr.multiInspiralTable->cont_chisq,
              tablePtr.multiInspiralTable->cont_chisq_dof,
              tablePtr.multiInspiralTable->sigmasq,
              tablePtr.multiInspiralTable->ligo_axis_ra,
              tablePtr.multiInspiralTable->ligo_axis_dec,
              tablePtr.multiInspiralTable->ligo_angle,
	      tablePtr.multiInspiralTable->ligo_angle_sig,
	      tablePtr.multiInspiralTable->inclination,
              tablePtr.multiInspiralTable->polarization,
              tablePtr.multiInspiralTable->event_id->id,
	      tablePtr.multiInspiralTable->null_statistic,
              tablePtr.multiInspiralTable->h1quad.re,
              tablePtr.multiInspiralTable->h1quad.im,
              tablePtr.multiInspiralTable->h2quad.re,
              tablePtr.multiInspiralTable->h2quad.im,
              tablePtr.multiInspiralTable->l1quad.re,
              tablePtr.multiInspiralTable->l1quad.im,
              tablePtr.multiInspiralTable->v1quad.re,
              tablePtr.multiInspiralTable->v1quad.im,
              tablePtr.multiInspiralTable->g1quad.re,
              tablePtr.multiInspiralTable->g1quad.im,
              tablePtr.multiInspiralTable->t1quad.re,
              tablePtr.multiInspiralTable->t1quad.im
                );
        tablePtr.multiInspiralTable = tablePtr.multiInspiralTable->next;
        ++(xml->rowCount);
      }
      break;
    case sim_inspiral_table:
      {
      while( tablePtr.simInspiralTable )
      {
        FIRST_TABLE_ROW
          fprintf( xml->fp, SIM_INSPIRAL_ROW,
              tablePtr.simInspiralTable->waveform,
              tablePtr.simInspiralTable->geocent_end_time.gpsSeconds,
              tablePtr.simInspiralTable->geocent_end_time.gpsNanoSeconds,
              tablePtr.simInspiralTable->h_end_time.gpsSeconds,
              tablePtr.simInspiralTable->h_end_time.gpsNanoSeconds,
              tablePtr.simInspiralTable->l_end_time.gpsSeconds,
              tablePtr.simInspiralTable->l_end_time.gpsNanoSeconds,
              tablePtr.simInspiralTable->g_end_time.gpsSeconds,
              tablePtr.simInspiralTable->g_end_time.gpsNanoSeconds,
              tablePtr.simInspiralTable->t_end_time.gpsSeconds,
              tablePtr.simInspiralTable->t_end_time.gpsNanoSeconds,
              tablePtr.simInspiralTable->v_end_time.gpsSeconds,
              tablePtr.simInspiralTable->v_end_time.gpsNanoSeconds,
              tablePtr.simInspiralTable->end_time_gmst,
              tablePtr.simInspiralTable->source,
              tablePtr.simInspiralTable->mass1,
              tablePtr.simInspiralTable->mass2,
              tablePtr.simInspiralTable->mchirp,
              tablePtr.simInspiralTable->eta,
              tablePtr.simInspiralTable->distance,
              tablePtr.simInspiralTable->longitude,
              tablePtr.simInspiralTable->latitude,
              tablePtr.simInspiralTable->inclination,
              tablePtr.simInspiralTable->coa_phase,
              tablePtr.simInspiralTable->polarization,
              tablePtr.simInspiralTable->psi0,
              tablePtr.simInspiralTable->psi3,
              tablePtr.simInspiralTable->alpha,
              tablePtr.simInspiralTable->alpha1,
              tablePtr.simInspiralTable->alpha2,
              tablePtr.simInspiralTable->alpha3,
              tablePtr.simInspiralTable->alpha4,
              tablePtr.simInspiralTable->alpha5,
              tablePtr.simInspiralTable->alpha6,
              tablePtr.simInspiralTable->beta,
              tablePtr.simInspiralTable->spin1x,
              tablePtr.simInspiralTable->spin1y,
              tablePtr.simInspiralTable->spin1z,
              tablePtr.simInspiralTable->spin2x,
              tablePtr.simInspiralTable->spin2y,
              tablePtr.simInspiralTable->spin2z,
              tablePtr.simInspiralTable->theta0,
              tablePtr.simInspiralTable->phi0,
              tablePtr.simInspiralTable->f_lower,
              tablePtr.simInspiralTable->f_final, 
              tablePtr.simInspiralTable->eff_dist_h,
              tablePtr.simInspiralTable->eff_dist_l,
              tablePtr.simInspiralTable->eff_dist_g,
              tablePtr.simInspiralTable->eff_dist_t,
              tablePtr.simInspiralTable->eff_dist_v,
	      tablePtr.simInspiralTable->numrel_mode_min,
	      tablePtr.simInspiralTable->numrel_mode_max,
	      tablePtr.simInspiralTable->numrel_data,
              xml->rowCount
              );
        tablePtr.simInspiralTable = tablePtr.simInspiralTable->next;
        ++(xml->rowCount);
        }
      }
      break;
    case sim_ringdown_table:
      {
        while( tablePtr.simRingdownTable )
        {
          FIRST_TABLE_ROW
            fprintf( xml->fp, SIM_RINGDOWN_ROW,
                tablePtr.simRingdownTable->waveform,
                tablePtr.simRingdownTable->coordinates,
                tablePtr.simRingdownTable->geocent_start_time.gpsSeconds,
                tablePtr.simRingdownTable->geocent_start_time.gpsNanoSeconds,
                tablePtr.simRingdownTable->h_start_time.gpsSeconds,
                tablePtr.simRingdownTable->h_start_time.gpsNanoSeconds,
                tablePtr.simRingdownTable->l_start_time.gpsSeconds,
                tablePtr.simRingdownTable->l_start_time.gpsNanoSeconds,
                tablePtr.simRingdownTable->start_time_gmst,
                tablePtr.simRingdownTable->longitude,
                tablePtr.simRingdownTable->latitude,
                tablePtr.simRingdownTable->distance,
                tablePtr.simRingdownTable->inclination,
                tablePtr.simRingdownTable->polarization,
                tablePtr.simRingdownTable->frequency,
                tablePtr.simRingdownTable->quality,
                tablePtr.simRingdownTable->phase,
                tablePtr.simRingdownTable->mass,
                tablePtr.simRingdownTable->spin,
                tablePtr.simRingdownTable->epsilon,
                tablePtr.simRingdownTable->amplitude,
                tablePtr.simRingdownTable->eff_dist_h,
                tablePtr.simRingdownTable->eff_dist_l,
                tablePtr.simRingdownTable->hrss,
                tablePtr.simRingdownTable->hrss_h,
                tablePtr.simRingdownTable->hrss_l,
                xml->rowCount
                  );
          tablePtr.simRingdownTable = tablePtr.simRingdownTable->next;
          ++(xml->rowCount);
        }
      }
      break;
    case summ_value_table:
      while( tablePtr.summValueTable )
      {
        FIRST_TABLE_ROW
          fprintf( xml->fp, SUMM_VALUE_ROW,
              tablePtr.summValueTable->program,
              tablePtr.summValueTable->start_time.gpsSeconds,
              tablePtr.summValueTable->start_time.gpsNanoSeconds,
              tablePtr.summValueTable->end_time.gpsSeconds,
              tablePtr.summValueTable->end_time.gpsNanoSeconds,
              tablePtr.summValueTable->ifo,
              tablePtr.summValueTable->name,
              tablePtr.summValueTable->value,
              tablePtr.summValueTable->comment,
              xml->rowCount
              );
        tablePtr.snglInspiralTable = tablePtr.snglInspiralTable->next;
        ++(xml->rowCount);
      }
      break;
    case sim_inst_params_table:
      while( tablePtr.simInstParamsTable )
      {
        FIRST_TABLE_ROW
          fprintf( xml->fp, SIM_INST_PARAMS_ROW,
              tablePtr.simInstParamsTable->name,
              tablePtr.simInstParamsTable->comment,
              tablePtr.simInstParamsTable->value
              );
        tablePtr.simInstParamsTable = tablePtr.simInstParamsTable->next;
        ++(xml->rowCount);
      }
      break;
    case stochastic_table:
      while( tablePtr.stochasticTable )
      {
        FIRST_TABLE_ROW
          fprintf( xml->fp, STOCHASTIC_ROW,
              tablePtr.stochasticTable->ifo_one,
              tablePtr.stochasticTable->ifo_two,
              tablePtr.stochasticTable->channel_one,
              tablePtr.stochasticTable->channel_two,
              tablePtr.stochasticTable->start_time.gpsSeconds,
              tablePtr.stochasticTable->start_time.gpsNanoSeconds,
              tablePtr.stochasticTable->duration.gpsSeconds,
              tablePtr.stochasticTable->duration.gpsNanoSeconds,
              tablePtr.stochasticTable->f_min,
              tablePtr.stochasticTable->f_max,
              tablePtr.stochasticTable->cc_stat,
              tablePtr.stochasticTable->cc_sigma
              );
        tablePtr.stochasticTable = tablePtr.stochasticTable->next;
        ++(xml->rowCount);
      }
      break;
    case stoch_summ_table:
      while( tablePtr.stochSummTable )
      {
        FIRST_TABLE_ROW
          fprintf( xml->fp, STOCH_SUMM_ROW,
              tablePtr.stochSummTable->ifo_one,
              tablePtr.stochSummTable->ifo_two,
              tablePtr.stochSummTable->channel_one,
              tablePtr.stochSummTable->channel_two,
              tablePtr.stochSummTable->start_time.gpsSeconds,
              tablePtr.stochSummTable->start_time.gpsNanoSeconds,
              tablePtr.stochSummTable->end_time.gpsSeconds,
              tablePtr.stochSummTable->end_time.gpsNanoSeconds,
              tablePtr.stochSummTable->f_min,
              tablePtr.stochSummTable->f_max,
              tablePtr.stochSummTable->y_opt,
              tablePtr.stochSummTable->error
              );
        tablePtr.stochSummTable = tablePtr.stochSummTable->next;
        ++(xml->rowCount);
      }
      break;
    case ext_triggers_table:
      while( tablePtr.extTriggerTable )
      {
        FIRST_TABLE_ROW
          fprintf( xml->fp, EXT_TRIGGERS_ROW,
              tablePtr.extTriggerTable->det_alts,
              tablePtr.extTriggerTable->det_band,
              tablePtr.extTriggerTable->det_fluence,
              tablePtr.extTriggerTable->det_fluence_int,
              tablePtr.extTriggerTable->det_name,
              tablePtr.extTriggerTable->det_peak,
              tablePtr.extTriggerTable->det_peak_int,
              tablePtr.extTriggerTable->det_snr,
              tablePtr.extTriggerTable->email_time,
              tablePtr.extTriggerTable->event_dec,
              tablePtr.extTriggerTable->event_dec_err,
              tablePtr.extTriggerTable->event_epoch,
              tablePtr.extTriggerTable->event_err_type,
              tablePtr.extTriggerTable->event_ra,
              tablePtr.extTriggerTable->event_ra_err,
              tablePtr.extTriggerTable->start_time,
              tablePtr.extTriggerTable->start_time_ns,
              tablePtr.extTriggerTable->event_type,
              tablePtr.extTriggerTable->event_z,
              tablePtr.extTriggerTable->event_z_err,
              tablePtr.extTriggerTable->notice_comments,
              tablePtr.extTriggerTable->notice_id,
              tablePtr.extTriggerTable->notice_sequence,
              tablePtr.extTriggerTable->notice_time,
              tablePtr.extTriggerTable->notice_type,
              tablePtr.extTriggerTable->notice_url,
              tablePtr.extTriggerTable->obs_fov_dec,
              tablePtr.extTriggerTable->obs_fov_dec_width,
              tablePtr.extTriggerTable->obs_fov_ra,
              tablePtr.extTriggerTable->obs_fov_ra_width,
	      tablePtr.extTriggerTable->obs_loc_ele,
	      tablePtr.extTriggerTable->obs_loc_lat,
	      tablePtr.extTriggerTable->obs_loc_long,
	      tablePtr.extTriggerTable->ligo_fave_lho,
	      tablePtr.extTriggerTable->ligo_fave_llo,
	      tablePtr.extTriggerTable->ligo_delay,
	      tablePtr.extTriggerTable->event_number_gcn,
	      tablePtr.extTriggerTable->event_number_grb,
	      tablePtr.extTriggerTable->event_status
	    );
        tablePtr.extTriggerTable = tablePtr.extTriggerTable->next;
        ++(xml->rowCount);
      }
      break;
    case filter_table:
      while( tablePtr.filterTable )
      {
        FIRST_TABLE_ROW
          fprintf( xml->fp, FILTER_ROW,
              tablePtr.filterTable->program,
              tablePtr.filterTable->start_time,
              tablePtr.filterTable->filter_name,
              tablePtr.filterTable->comment,
              xml->rowCount
              );
        tablePtr.filterTable = tablePtr.filterTable->next;
        ++(xml->rowCount);
      }
      break;
    default:
      ABORT( status, LIGOLWXMLH_EUTAB, LIGOLWXMLH_MSGEUTAB );
  }
  RETURN( status );
}

#undef FIRST_TABLE_ROW /* undefine first table row macro */


int XLALWriteLIGOLwXMLSimBurstTable(
	LIGOLwXMLStream *xml,
	const SimBurst *sim_burst
)
{
	static const char func[] = "XLALWriteLIGOLwXMLSimBurstTable";
	const char *row_head = "\n\t\t\t";

	if(xml->table != no_table) {
		XLALPrintError("a table is still open");
		XLAL_ERROR(func, XLAL_EFAILED);
	}

	/* table header */

	XLALClearErrno();
	XLALFilePuts("\t<Table Name=\"sim_burst:table\">\n", xml->fp);
	XLALFilePuts("\t\t<Column Name=\"sim_burst:process_id\" Type=\"ilwd:char\"/>\n", xml->fp);
	XLALFilePuts("\t\t<Column Name=\"sim_burst:waveform\" Type=\"lstring\"/>\n", xml->fp);
	XLALFilePuts("\t\t<Column Name=\"sim_burst:ra\" Type=\"real_8\"/>\n", xml->fp);
	XLALFilePuts("\t\t<Column Name=\"sim_burst:dec\" Type=\"real_8\"/>\n", xml->fp);
	XLALFilePuts("\t\t<Column Name=\"sim_burst:psi\" Type=\"real_8\"/>\n", xml->fp);
	XLALFilePuts("\t\t<Column Name=\"sim_burst:time_geocent_gps\" Type=\"int_4s\"/>\n", xml->fp);
	XLALFilePuts("\t\t<Column Name=\"sim_burst:time_geocent_gps_ns\" Type=\"int_4s\"/>\n", xml->fp);
	XLALFilePuts("\t\t<Column Name=\"sim_burst:time_geocent_gmst\" Type=\"real_8\"/>\n", xml->fp);
	XLALFilePuts("\t\t<Column Name=\"sim_burst:duration\" Type=\"real_8\"/>\n", xml->fp);
	XLALFilePuts("\t\t<Column Name=\"sim_burst:frequency\" Type=\"real_8\"/>\n", xml->fp);
	XLALFilePuts("\t\t<Column Name=\"sim_burst:bandwidth\" Type=\"real_8\"/>\n", xml->fp);
	XLALFilePuts("\t\t<Column Name=\"sim_burst:q\" Type=\"real_8\"/>\n", xml->fp);
	XLALFilePuts("\t\t<Column Name=\"sim_burst:pol_ellipse_angle\" Type=\"real_8\"/>\n", xml->fp);
	XLALFilePuts("\t\t<Column Name=\"sim_burst:pol_ellipse_e\" Type=\"real_8\"/>\n", xml->fp);
	XLALFilePuts("\t\t<Column Name=\"sim_burst:amplitude\" Type=\"real_8\"/>\n", xml->fp);
	XLALFilePuts("\t\t<Column Name=\"sim_burst:hrss\" Type=\"real_8\"/>\n", xml->fp);
	XLALFilePuts("\t\t<Column Name=\"sim_burst:egw_over_rsquared\" Type=\"real_8\"/>\n", xml->fp);
	/* FIXME:  waveform is an unsigned long, but metaio doesn't support
	 * int_8u columns */
	XLALFilePuts("\t\t<Column Name=\"sim_burst:waveform_number\" Type=\"int_8s\"/>\n", xml->fp);
	XLALFilePuts("\t\t<Column Name=\"sim_burst:simulation_id\" Type=\"ilwd:char\"/>\n", xml->fp);
	XLALFilePuts("\t\t<Stream Name=\"sim_burst:table\" Type=\"Local\" Delimiter=\",\">", xml->fp);
	if(XLALGetBaseErrno())
		XLAL_ERROR(func, XLAL_EFUNC);

	/* rows */

	for(; sim_burst; sim_burst = sim_burst->next) {
		/* FIXME:  waveform is an unsigned long, but metaio doesn't
		 * support int_8u columns */
		if(XLALFilePrintf(xml->fp, "%s\"process:process_id:%ld\",\"%s\",%.16g,%.16g,%.16g,%d,%d,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g,%ld,\"sim_burst:simulation_id:%ld\"",
			row_head,
			sim_burst->process_id,
			sim_burst->waveform,
			sim_burst->ra,
			sim_burst->dec,
			sim_burst->psi,
			sim_burst->time_geocent_gps.gpsSeconds,
			sim_burst->time_geocent_gps.gpsNanoSeconds,
			sim_burst->time_geocent_gmst,
			sim_burst->duration,
			sim_burst->frequency,
			sim_burst->bandwidth,
			sim_burst->q,
			sim_burst->pol_ellipse_angle,
			sim_burst->pol_ellipse_e,
			sim_burst->amplitude,
			sim_burst->hrss,
			sim_burst->egw_over_rsquared,
			sim_burst->waveform_number,
			sim_burst->simulation_id
		) < 0)
			XLAL_ERROR(func, XLAL_EFUNC);
		row_head = ",\n\t\t\t";
	}

	/* table footer */

	if(XLALFilePuts("\n\t\t</Stream>\n\t</Table>\n", xml->fp) < 0)
		XLAL_ERROR(func, XLAL_EFUNC);

	/* done */

	return 0;
}

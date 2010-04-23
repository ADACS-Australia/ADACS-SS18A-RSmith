/*
*  Copyright (C) 2007 Alexander Dietz, Duncan Brown, Jolien Creighton, Kipp Cannon, Lisa M. Goggin, Patrick Brady, Robert Adam Mercer, Stephen Fairhurst, Thomas Cokelaer
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
 * File Name: LIGOLwXMLRead.h
 *
 * Author: Brown, D. A. and Fairhurst, S.
 *
 * Revision: $Id$
 *
 *-----------------------------------------------------------------------
 */

#if 0
<lalVerbatim file="LIGOLwXMLReadHV">
Author: Brown, D. A. and Fairhurst, S.
$Id$
</lalVerbatim>
<lalLaTeX>
\section{Header \texttt{LIGOLwXMLRead.h}}
\label{s:LIGOLwXMLRead.h}

Provides functions for reading LIGO lightweight XML files to LIGO
metadata database tables.

\subsection*{Synopsis}
\begin{verbatim}
#include <lal/LIGOLwXMLRead.h>
\end{verbatim}

</lalLaTeX>
#endif

#ifndef _LIGOLWXMLREAD_H
#define _LIGOLWXMLREAD_H

#include <stdio.h>
#include <stdlib.h>
#include <metaio.h>
#include <lal/LALDatatypes.h>
#include <lal/LALConstants.h>
#include <lal/LIGOMetadataTables.h>

#ifdef  __cplusplus
extern "C" {
#endif


NRCSID( LIGOLWXMLREADH, "$Id$" );

/* <lalLaTeX>

\subsection*{Error conditions}
\input{LIGOLwXMLReadHE}

</lalLaTeX> */

/* <lalErrTable file="LIGOLwXMLReadHE"> */
#define LIGOLWXMLREADH_ENULL 1
#define LIGOLWXMLREADH_ENNUL 2
#define LIGOLWXMLREADH_EALOC 3
#define LIGOLWXMLREADH_EUTAB 4
#define LIGOLWXMLREADH_ENCOL 5
#define LIGOLWXMLREADH_ENTAB 6
#define LIGOLWXMLREADH_EPARS 7
#define LIGOLWXMLREADH_EMTAB 8
#define LIGOLWXMLREADH_EENDT 9
#define LIGOLWXMLREADH_ETMSM 10
#define LIGOLWXMLREADH_ETNOP 11

#define LIGOLWXMLREADH_MSGENULL "Null pointer"
#define LIGOLWXMLREADH_MSGENNUL "Non-null pointer"
#define LIGOLWXMLREADH_MSGEALOC "Memory allocation error"
#define LIGOLWXMLREADH_MSGEUTAB "Unknown metadata table type"
#define LIGOLWXMLREADH_MSGENCOL "Unable to find table column"
#define LIGOLWXMLREADH_MSGENTAB "Requested table not found in file"
#define LIGOLWXMLREADH_MSGEPARS "Error parsing table"
#define LIGOLWXMLREADH_MSGEMTAB "No table type specified"
#define LIGOLWXMLREADH_MSGEENDT "Ending a table without an beginning a table"
#define LIGOLWXMLREADH_MSGETMSM "Table type mismatch"
#define LIGOLWXMLREADH_MSGETNOP "Table not begun for writing"
/* </lalErrTable> */


#if 0
<lalLaTeX>
\subsection*{Structures}

\subsubsection*{Type \texttt{tagMetaTableDirectory}}
\idx[Type]{tagMetaTableDirectory}
</lalLaTeX>
#endif

/* <lalVerbatim> */
typedef struct
tagMetaTableDirectory
{
  const CHAR *name;
  INT4   pos;
  INT4   idx;
}
MetaTableDirectory;
/* </lalVerbatim> */
#if 0
<lalLaTeX>

This structure allows for the association of entries in a MetaDataTable
with columns in an xml file.
\begin{description}
\item[\texttt{name}] The name of the column in the XML table.
\item[\texttt{pos}] The position of this column in the XML table.
\item[\texttt{idx}] The id number of the column.
\end{description}
</lalLaTeX>
#endif


#if 0
<lalLaTeX>
\newpage\input{CreateMetaTableDirC}
</lalLaTeX>
#endif

MetaTableDirectory* XLALCreateMetaTableDir(
    const MetaioParseEnv    env,
    MetadataTableType       table
    );

void
LALCreateMetaTableDir(
    LALStatus              *status,
    MetaTableDirectory    **tableDir,
    const MetaioParseEnv    env,
    MetadataTableType       table
    );

#if 0
<lalLaTeX>
\newpage\input{LIGOLwXMLReadC}
</lalLaTeX>
#endif

int
XLALLIGOLwFindColumn(
    struct MetaioParseEnvironment *env,
    const char *name,
    enum METAIO_Type type,
    int required
);

long long XLALLIGOLwParseIlwdChar(
    const struct MetaioParseEnvironment *env,
    int column_number,
    const char *ilwd_char_table_name,
    const char *ilwd_char_column_name
);

int
XLALLIGOLwHasTable(
    const char *filename,
    const char *table_name
);

ProcessTable *
XLALProcessTableFromLIGOLw (
    const char *filename
);

ProcessParamsTable *
XLALProcessParamsTableFromLIGOLw (
    const char *filename
);

/* these functions need to be lalified, but they are in support... */

SearchSummaryTable *
XLALSearchSummaryTableFromLIGOLw (
    const CHAR          *fileName
    );

#ifdef  __cplusplus
}
#endif

#endif /* _LIGOLWXMLREAD_H */

/************************************ <lalVerbatim file="ConfigFileHV">
Author: Prix, Reinhard
$Id$
************************************* </lalVerbatim> */

/**************************************************** <lalLaTeX>
\section{Header \texttt{ConfigFile.h}}
\label{s:ConfigFile.h}

Module for general config-file reading.

\subsection*{Synopsis}
\begin{verbatim}
#include <lal/ConfigFile.h>
\end{verbatim}

***************************************************** </lalLaTeX> */

#ifndef _CONFIGFILE_H  /* Double-include protection. */
#define _CONFIGFILE_H

#include <lal/LALDatatypes.h>
#include <lal/StringInput.h>

/* C++ protection. */
/* 
#ifdef  __cplusplus   
extern "C" {
#endif
*/

NRCSID( CONFIGFILEH, "$Id$");

/********************************************************** <lalLaTeX>
\subsection*{Error codes}
</lalLaTeX>
***************************************************** <lalErrTable> */
#define CONFIGFILEH_ENULL 		1
#define CONFIGFILEH_EFILE		2
#define CONFIGFILEH_EVAR		3
#define CONFIGFILEH_EFMT		4
#define CONFIGFILEH_ETOKENS		5
#define CONFIGFILEH_ENONULL		6
#define CONFIGFILEH_EUNKNOWN		8
#define CONFIGFILEH_EMEM		9
#define CONFIGFILEH_EBOOL		10

#define CONFIGFILEH_MSGENULL 		"Arguments contained an unexpected null pointer."
#define CONFIGFILEH_MSGEFILE		"File error."
#define CONFIGFILEH_MSGEVAR		"Config variable not found."
#define CONFIGFILEH_MSGEFMT		"Config variable not readable using given format-string."
#define CONFIGFILEH_MSGETOKENS		"The input ConfigData seems corrupted."
#define CONFIGFILEH_MSGENONULL		"Output pointer is not NULL"
#define CONFIGFILEH_MSGEUNKNOWN		"Unknown config-file entry found"
#define CONFIGFILEH_MSGEMEM		"Out of memory"
#define CONFIGFILEH_MSGEBOOL		"Illegal BOOLEAN entry"

/*************************************************** </lalErrTable> */


/***************************************************** <lalLaTeX>
\subsection*{Constants}
\idx[Constant]{CONFIGFILE\_IGNORE}
\idx[Constant]{CONFIGFILE\_WARN}
\idx[Constant]{CONFIGFILE\_ERROR}

The following are constants used to define various \verb+ConfigStrictness+-levels.
</lalLaTeX> */
/* <lalVerbatim> */
typedef enum {
  CONFIGFILE_IGNORE = 0,	/* ignore missing config-variable or unparsed config-entries */
  CONFIGFILE_WARN,		/* issue a warning but don't report an error. */
  CONFIGFILE_ERROR,		/* issue an error-message and report a LAL-error */
} ConfigStrictness;
/* </lalVerbatim> */
/****************************************************** <lalLaTeX>
\subsection*{Types}

\subsubsection*{Structure \texttt{LALConfigVar}}
\idx[Type]{LALConfigVar}

This structure defines a config-variable to be read in using the
general-purpose reading function \verb+LALReadConfigVariable()+:

</lalLaTeX> */
/* <lalVerbatim> */
typedef struct {
  const CHAR *varName;		/* Variable-name to be read in the config-file */
  const CHAR *fmt;		/* Format string for reading (\verb+sscanf()+-style) */
  ConfigStrictness strictness;	/* what to do if variable not found: ignore, warn, error */
} LALConfigVar;
/* </lalVerbatim> */

/* <lalLaTeX>

\subsubsection*{Structure \texttt{LALConfigData}}
\idx[Type]{LALConfigData}

This structure holds the actual config-file contents in a pre-parsed
form, namely stripped from all comments and ignored whitespaces, and
separated into lines (taking into account line-continuation).
This is used as the input structure for the config-variable reading routines.

</lalLaTeX> */
/* <lalVerbatim> */
typedef struct {
  TokenList *lines;	/* list of pre-parsed config-file lines */
  BOOLEAN *wasRead;	/* keep track of successfully read lines for strictness-checking */
} LALConfigData;
/* </lalVerbatim> */

/********************************************************** <lalLaTeX>
\vfill{\footnotesize\input{ConfigFileHV}}
\newpage\input{ConfigFileC}
\newpage\input{ConfigFileTestC}
******************************************************* </lalLaTeX> */

/* Function prototypes */
void LALLoadConfigFile (LALStatus *stat, LALConfigData **cfgdata, const CHAR *fname);
void LALDestroyConfigData (LALStatus *stat, LALConfigData **cfgdata);

void 
LALReadConfigBOOLVariable (LALStatus *stat, 
			  BOOLEAN *varp, 
			  const LALConfigData *cfgdata, 
			  const CHAR *varName, 
			  BOOLEAN *wasRead);

void
LALReadConfigINT4Variable (LALStatus *stat,
			   INT4 *varp, 
			   const LALConfigData *cfgdata, 
			   const CHAR *varName, 
			   BOOLEAN *wasRead);

void
LALReadConfigREAL8Variable (LALStatus *stat, 
			    REAL8 *varp, 
			    const LALConfigData *cfgdata, 
			    const CHAR *varName, 
			    BOOLEAN *wasRead);

void 
LALReadConfigSTRINGVariable (LALStatus *stat, 
			     CHAR **varp, 
			     const LALConfigData *cfgdata, 
			     const CHAR *varName,
			     BOOLEAN *wasRead);

void
LALReadConfigSTRINGNVariable (LALStatus *stat, 
			      CHARVector *varp,
			      const LALConfigData *cfgdata, 
			      const CHAR *varName,
			      BOOLEAN *wasRead);

void
LALReadConfigVariable (LALStatus *stat, 
		       void *varp,
		       const LALConfigData *cfgdata,
		       const LALConfigVar *param,
		       BOOLEAN *wasRead);

void LALCheckConfigReadComplete (LALStatus *stat, const LALConfigData *cfgdata, ConfigStrictness strict);

/* C++ protection. */
/*
#ifdef  __cplusplus
}
#endif  
*/

#endif  /* Double-include protection. */




/*
 * Copyright (C) 2004, 2005 Reinhard Prix
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

/** \file
 * \ingroup UserInput
 * \author Reinhard Prix
 * 
 * \brief Convenient unified handling of user-input via config-file and/or command-line.
 *
 */

/* <lalVerbatim file="UserInputCV">
Author: Prix, Reinhard
$Id$
************************************* </lalVerbatim> */

/********************************************************** <lalLaTeX>
\subsection{Module \texttt{UserInput.c}}
\label{ss:UserInput.c}

Convenient unified handling of user-input via config-file and/or command-line.

\subsubsection*{Prototypes}

\input{UserInputCP}

\subsubsection*{Description}

This module provides a very simple and convenient way to handle
user-input, wether it comes from the command-line or a config-file. 

The general procedure is the following:

\subsubsection*{Algorithm}

\subsubsection*{Uses}

\subsubsection*{Notes}

\vfill{\footnotesize\input{UserInputCV}}

******************************************************* </lalLaTeX> */
#include "getopt.h"

#include <lal/LALStdio.h>
#include <lal/UserInput.h>

NRCSID( USERINPUTC, "$Id$");

extern INT4 lalDebugLevel;

#define TRUE  (1==1)
#define FALSE (1==0)

/** Defines the type of a "user-variable": bool, int, real or string. 
 * Should be used only internally !! 
 */
typedef enum {
  UVAR_BOOL,	/**< boolean */
  UVAR_INT4,	/**< integer */
  UVAR_REAL8,	/**< float */
  UVAR_STRING	/**< string */
} UserVarType;

/** Linked list to hold the complete information about the user-variables.
 */
typedef struct tagLALUserVariable {
  const CHAR *name;	/**< full name */
  UserVarType type;	/**< type: bool, int, float or string */
  CHAR optchar;		/**< cmd-line character */
  const CHAR *help;	/**< help-string */
  void *varp;		/**< pointer to the actual C-variable */
  UserVarState state;	/**< state (empty, default, set) */
  struct tagLALUserVariable *next; /* linked list */
} LALUserVariable;

/** The module-local linked list to hold the user-variables */
static LALUserVariable UVAR_vars;	/**< empty head */

/* needed for command-line parsing */
extern char *optarg;
extern int optind, opterr, optopt;

/* ---------- internal prototypes ---------- */
static void RegisterUserVar (LALStatus *, const CHAR *name, UserVarType type, CHAR optchar, 
			     UserVarState flag, const CHAR *helpstr, void *cvar);

static void UvarValue2String (LALStatus *, CHAR **outstr, LALUserVariable *uvar);

/*---------- Function definitions ---------- */

/* these are type-specific wrappers to allow tighter type-checking! */
/* <lalVerbatim file="UserInputCP"> */
void
LALRegisterREALUserVar (LALStatus *status, 
			const CHAR *name, 
			CHAR optchar, 
			UserVarState flag,
			const CHAR *helpstr, 
			REAL8 *cvar)
{ /* </lalVerbatim> */
  RegisterUserVar (status, name, UVAR_REAL8, optchar, flag, helpstr, cvar);
}
/* <lalVerbatim file="UserInputCP"> */
void
LALRegisterINTUserVar (LALStatus *status, 
		       const CHAR *name, 
		       CHAR optchar, 
		       UserVarState flag,
		       const CHAR *helpstr, 
		       INT4 *cvar)
{ /* </lalVerbatim> */
  RegisterUserVar (status, name, UVAR_INT4, optchar, flag, helpstr, cvar);
} 

/* <lalVerbatim file="UserInputCP"> */
void
LALRegisterBOOLUserVar (LALStatus *status, 
			const CHAR *name, 
			CHAR optchar, 
			UserVarState flag,
			const CHAR *helpstr, 
			BOOLEAN *cvar)
{ /* </lalVerbatim> */
  RegisterUserVar (status, name, UVAR_BOOL, optchar, flag, helpstr, cvar);
} 

/* <lalVerbatim file="UserInputCP"> */
void
LALRegisterSTRINGUserVar (LALStatus *status, 
			  const CHAR *name, 
			  CHAR optchar, 
			  UserVarState flag,
			  const CHAR *helpstr, 
			  CHAR **cvar)
{ /* </lalVerbatim> */
  RegisterUserVar (status, name, UVAR_STRING, optchar, flag, helpstr, cvar);
} 




/** Register a user-variable with the module.
 *  Effectively put an appropriate entry into UVAR_vars
 *
 *  \note don't use this directly, as it's not type-safe!!
 *      ==> use one of the 4 wrappers: LALRegisterREALUserVar(), 
 *    LALRegisterINTUserVar(), LALRegisterBOOLUserVar(), LALRegisterSTRINGUserVar().
 *
 */
static void
RegisterUserVar (LALStatus *status, 
		 const CHAR *name, 
		 UserVarType type, 
		 CHAR optchar, 
		 UserVarState flag,
		 const CHAR *helpstr,
		 void *cvar)
{
  LALUserVariable *ptr;

  INITSTATUS( status, "LALRegisterUserVar", USERINPUTC );

  ASSERT (cvar != NULL, status, USERINPUTH_ENULL, USERINPUTH_MSGENULL);
  ASSERT (name != NULL, status, USERINPUTH_ENULL, USERINPUTH_MSGENULL);

  /* find end of uvar-list */
  ptr = &UVAR_vars;
  while (ptr->next) 
    ptr = ptr->next;

  /* create new entry */
  ptr->next = LALCalloc (1, sizeof(LALUserVariable));
  if (ptr->next == NULL) {
    ABORT (status, USERINPUTH_EMEM,  USERINPUTH_MSGEMEM);
  }
  /* set pointer to newly created entry (Note: the head remains empty!) */
  ptr = ptr->next;

  /* fill in values */
  ptr->name = name;
  ptr->type = type;
  ptr->optchar = optchar;
  ptr->help = helpstr;
  ptr->varp = cvar;
  ptr->state = flag;

  RETURN (status);

} /* LALRegisterUserVar() */



/** Free all memory associated with user-variable linked list
 */
/* <lalVerbatim file="UserInputCP"> */
void
LALDestroyUserVars (LALStatus *status)
{ /* </lalVerbatim> */

  LALUserVariable *ptr, *lastptr;

  INITSTATUS( status, "LALDestroyUserVars", USERINPUTC );

  /* step through user-variables: free list-entries and all allocated strings */
  ptr = &(UVAR_vars);
  lastptr = NULL;
  while ( (ptr=ptr->next) != NULL)
    {
      /* is an allocated string here? */
      if ( (ptr->type == UVAR_STRING) && (*(CHAR**)(ptr->varp) != NULL) ) 
	{
	  LALFree ( *(CHAR**)(ptr->varp) );
	  *(CHAR**)(ptr->varp) = NULL;		/* IMPORTANT: reset user-variable to NULL ! */
	}

      /* free list-entry behind us (except for the head) */
      if (lastptr) {
	LALFree (lastptr);
	lastptr=NULL;
      }

      lastptr = ptr;

    } /* while ptr->next */

  if (lastptr) {
    LALFree (lastptr);
    lastptr=NULL;
  }

  /* clean head */
  memset (&UVAR_vars, 0, sizeof(UVAR_vars));
  
  RETURN(status);
} /* LALDestroyUserVars() */


/** Parse command-line into UserVariable array
 */
/* <lalVerbatim file="UserInputCP"> */
void
LALUserVarReadCmdline (LALStatus *status, 
		     int argc, char *argv[]) 	  /* command-line contents */
{ /* </lalVerbatim> */
  INT4 c;
  UINT4 pos;
  UINT4 numvars;
  LALUserVariable *ptr = NULL;
  char optstring[512] = "\0";	/* string of short-options, should be easily enough */
  struct option *long_options;
  int longindex = -1;

  INITSTATUS( status, "LALUserVarReadCmdline", USERINPUTC );
  ATTATCHSTATUSPTR (status);

  ASSERT (argv, status, USERINPUTH_ENULL, USERINPUTH_MSGENULL);
  ASSERT (UVAR_vars.next, status, USERINPUTH_ENOUVARS,  USERINPUTH_MSGENOUVARS);

  /* build optstring of short-options */
  ptr = &UVAR_vars;	/* set to empty head */
  pos = 0;
  numvars = 0;

  /* special treatment of head, which could contain the debug-option: 
   * NOTE: this one will *NOT* be treated by the remaining function, 
   * (as the head is always skipped), but it has to be in the optstring 
   * to avoid an error if specified on the command-line. 
   * Treatment of debug-option reading has to be done separately using
   * UVARgetDebugLevel() 
   */
  if ( (ptr->help != NULL) && (ptr->optchar != 0) )
    {
      numvars ++;
      optstring[pos++] = ptr->optchar;
      optstring[pos++] = ':';		/* requires an argument */
    } /* if debug-option */

  /* treat the remaining "normal" entries */
  while ( (ptr = ptr->next) != NULL )
    {
      numvars ++;			/* counter number of user-variables */
      if (ptr->optchar == 0)		/* if no short-option given, ignore */
	continue;
      optstring[pos++] = ptr->optchar;
      optstring[pos++] = ':';		/* everything but bool takes an argument */
      if (ptr->type == UVAR_BOOL)	/* but for BOOL its optional */
	optstring[pos++] = ':';
    } /* while ptr->next */
  optstring[pos] = '\0';

  /* fill option-struct for long-options */
  long_options = LALCalloc (1, (numvars+1) * sizeof(struct option));
  ptr = &UVAR_vars;	/* start again from beginning */
  pos = 0;
  while ( (ptr= ptr->next) != NULL)
    {
      if (ptr->name == NULL)		/* if no long-option given, ignore */
	continue;
      long_options[pos].name = ptr->name;
      long_options[pos].has_arg = (ptr->type == UVAR_BOOL) ? optional_argument : required_argument;
      long_options[pos].flag = NULL;	/* get val returned from getopt_long() */
      long_options[pos].val = 0;	/* we use longindex to find long-options */
      pos ++;
    } /* while ptr->next */
  /* null-terminate array */
  long_options[pos].name = 0;
  long_options[pos].has_arg = 0;
  long_options[pos].flag = 0;
  long_options[pos].val = 0;


  /* NOTE: in case we get called several times, we have to make sure here that getopt() gets 
   * properly reset/initialized. We do this using the (undocumented) feature of GNU getopt
   * of setting optind to 0. As we're linking our private version of GNU getopt, this should be
   * guaranteed to work.
   * 
   * Bruce's notes: read getopt_long() source code, and in particular
   * _getopt_internal() to see what is initialized.
   */
  optind = 0; 	/* reset getopt(), getopt_long() */

  /* parse the command-line */
  while ( (c = getopt_long(argc, argv, optstring, long_options, &longindex)) != -1 )
    {
      if (c == '?') {
	LALPrintError ("\nERROR: unkown command-line option encountered\n");
	LALPrintError ("see '%s --help' for usage-help\n\n", argv[0]);
	ABORT (status, USERINPUTH_EOPT, USERINPUTH_MSGEOPT);
      }
      if (c != 0) 	/* find short-option character */
	{
	  ptr = &UVAR_vars;
	  do {
	    if (c == ptr->optchar)
	      break;
	  } while ( (ptr=ptr->next) != NULL);
	} /* if short-option */
      else	/* find long-option: returned in longindex */
	{
	  ptr = &UVAR_vars;
	  while ( (ptr=ptr->next) != NULL)
	    if ( !strcmp (long_options[longindex].name, ptr->name) )
	      break;
	} /* if long-option */

      if (ptr == NULL) {	/* should not be possible: nothing found at all... */
	LALPrintError ("WARNING: failed to find option.. this points to a coding-error!\n");
	ABORT (status, USERINPUTH_EOPT, USERINPUTH_MSGEOPT);
      }

      /* if we found the debug-switch, ignore it (has been handled already */
      if (ptr == &UVAR_vars)
	continue;

      switch (ptr->type)
	{
	  INT2 ans;
	case UVAR_BOOL:
	  ans = -1;

	  /* subtlety with optional argument: it's not necessarily found in the *same* argv-entry
	   * eg, if no '=' was used, so we have to check for that case by hand: */

	  /* if the next entry is not an option, take it as an argument */
	  if (optarg == NULL && (optind<argc) && (argv[optind][0]!='-') && (argv[optind][0]!='@') )	 
	    optarg = argv[optind];

	  if ( optarg == NULL )	/* no argument found at all: defaults to TRUE */
	    {
	      ans = 1;
	    }
	  else	/* parse bool-argument: should be consistent with bool-parsing in ConfigFile!! */
	    {
	      /* get rid of case ambiguities */
	      TRY (LALLowerCaseString (status->statusPtr, optarg), status);

	      if      ( !strcmp(optarg, "yes") || !strcmp(optarg, "true") || !strcmp(optarg,"1") )
		ans = 1;
	      else if ( !strcmp (optarg, "no") || !strcmp(optarg,"false") || !strcmp(optarg,"0") )
		ans = 0;
	      else {	/* failed to parse BOOL properly */
		LALPrintError ("\nIllegal bool-value `%s`\n\n", optarg);
		ABORT (status, USERINPUTH_ECMDLARG, USERINPUTH_MSGECMDLARG);
	      }
	    } /* parse bool-argument */
	  
	  /* only set if we properly parsed something */
	  if (ans != -1) {
	    *(BOOLEAN*)(ptr->varp)  = (BOOLEAN)ans;
	    ptr->state |= UVAR_WAS_SET;
	  }

	  break;

	case UVAR_INT4:
	  if ( 1 != sscanf ( optarg, "%" LAL_INT4_FORMAT, (INT4*)(ptr->varp)) )
	    {
	      LALPrintError ("\nIllegal INT4 commandline argument to --%s: '%s'\n\n", 
			     ptr->name, optarg);
	      ABORT (status, USERINPUTH_ECMDLARG, USERINPUTH_MSGECMDLARG);
	    }

	  ptr->state |= UVAR_WAS_SET;
	  break;

	case UVAR_REAL8:
	  if ( 1 != sscanf ( optarg, "%" LAL_REAL8_FORMAT, (REAL8*)(ptr->varp)) )
	    {
	      LALPrintError ("\nIllegal REAL8 commandline argument to --%s: '%s'\n\n", 
			     ptr->name, optarg);
	      ABORT (status, USERINPUTH_ECMDLARG, USERINPUTH_MSGECMDLARG);
	    }

	  ptr->state |= UVAR_WAS_SET;
	  break;

	case UVAR_STRING:
	  if (!optarg) {	/* should not be possible, but let's be paranoid */
	    ABORT (status, USERINPUTH_ENULL, USERINPUTH_MSGENULL);
	  }
	  if ( *(CHAR**)(ptr->varp) != NULL) {	 /* something allocated here before? */
	    LALFree ( *(CHAR**)(ptr->varp) );
	    (*(CHAR**)(ptr->varp))=NULL;
	  }

	  *(CHAR**)(ptr->varp) = LALCalloc (1, strlen(optarg) + 1);
	  if ( *(CHAR**)(ptr->varp) == NULL) {
	    ABORT (status, USERINPUTH_EMEM, USERINPUTH_MSGEMEM);
	  }
	  strcpy ( *(CHAR**)(ptr->varp), optarg);
	  ptr->state |= UVAR_WAS_SET;
	  break; 

	default:
	  LALPrintError ("ERROR: unkown UserVariable-type encountered... points to a coding error!\n");
	  ABORT (status, USERINPUTH_ENULL, USERINPUTH_MSGENULL);
	  break;

	} /* switch ptr->type */

    } /* while getopt_long() */

  LALFree (long_options);
  long_options=NULL;

  DETATCHSTATUSPTR (status);
  RETURN (status);

} /* LALUserVarReadCmdline() */

/** Read config-variables from cfgfile and parse into input-structure.
 *
 * An error is reported if the config-file reading fails, but the 
 * individual variable-reads are treated as optional
 */
/* <lalVerbatim file="UserInputCP"> */
void
LALUserVarReadCfgfile (LALStatus *status, 
		       const CHAR *cfgfile) 	   /* name of config-file */
{/* </lalVerbatim> */
  LALParsedDataFile *cfg = NULL;
  CHAR *stringbuf;
  LALUserVariable *ptr;
  BOOLEAN wasRead;

  INITSTATUS( status, "LALUserVarReadCfgfile", USERINPUTC );
  ATTATCHSTATUSPTR (status);

  ASSERT (UVAR_vars.next, status, USERINPUTH_ENOUVARS,  USERINPUTH_MSGENOUVARS);

  TRY (LALParseDataFile (status->statusPtr, &cfg, cfgfile), status);

  /* step through all user-variable: read those with names from config-file */
  ptr = &UVAR_vars;
  while ( (ptr=ptr->next) != NULL)
    {
      if (ptr->name == NULL)	/* ignore name-less user-variable */
	continue;

      wasRead = FALSE;

      switch (ptr->type)
	{
	case UVAR_BOOL:
	  TRY(LALReadConfigBOOLVariable(status->statusPtr, ptr->varp, cfg, ptr->name, &wasRead), status);
	  if (wasRead)
	    ptr->state |= UVAR_WAS_SET;
	  break;
	case UVAR_INT4:
	  TRY(LALReadConfigINT4Variable(status->statusPtr, ptr->varp, cfg, ptr->name, &wasRead),status);
	  if (wasRead)
	    ptr->state |= UVAR_WAS_SET;
	  break;
	case UVAR_REAL8:
	  TRY(LALReadConfigREAL8Variable(status->statusPtr, ptr->varp, cfg, ptr->name, &wasRead),status);
	  if (wasRead)
	    ptr->state |= UVAR_WAS_SET;
	  break;
	case UVAR_STRING:
	  stringbuf = NULL;
	  TRY(LALReadConfigSTRINGVariable(status->statusPtr,&stringbuf,cfg,ptr->name,&wasRead),status);
	  if ( wasRead && stringbuf)	/* did we find something? */
	    {
	      if ( *(CHAR**)(ptr->varp) != NULL) {	 /* something allocated here before? */
		LALFree ( *(CHAR**)(ptr->varp) );
		( *(CHAR**)(ptr->varp) )=NULL;
	      }

	      *(CHAR**)(ptr->varp) = stringbuf;
	      ptr->state |= UVAR_WAS_SET;
	    } /* if stringbuf */
	  break;

	default:
	  LALPrintError ("ERROR: unkown UserVariable-type encountered...points to a coding error!\n");
	  ABORT (status, USERINPUTH_ENULL, USERINPUTH_MSGENULL);
	  break;

	} /* switch ptr->type */

    } /* while ptr->next */

  /* ok, that should be it: check if there were more definitions we did not read */
  TRY (LALCheckConfigReadComplete (status->statusPtr, cfg, CONFIGFILE_WARN), status);

  TRY( LALDestroyParsedDataFile (status->statusPtr, &cfg), status);

  DETATCHSTATUSPTR(status);
  RETURN (status);

} /* LALUserVarReadCfgfile() */


#define UVAR_MAXHELPLINE  512	/* max length of one help-line */
#define UVAR_MAXDEFSTR    100 	/* max length of default-string */
/** Assemble all help-info from uvars into a help-string.
 */
/* <lalVerbatim file="UserInputCP"> */
void
LALUserVarHelpString (LALStatus *status, 
		      CHAR **helpstring, /* output: allocated here! */
		      const CHAR *progname)
{ /* </lalVerbatim> */

  CHAR strbuf[UVAR_MAXHELPLINE];	/* should be enough for one line...*/
  CHAR defaultstr[UVAR_MAXDEFSTR]; 	/* for display of default-value */
  CHAR optstr[10];			/* display of opt-char */
  /* we need strings for UVAR_BOOL, UVAR_INT4, UVAR_REAL8, UVAR_STRING: */
  const CHAR *typestr[] = {"BOOL", "INT", "REAL", "STRING"}; 
  LALUserVariable *ptr;
  LALUserVariable *helpptr = NULL;	/* pointer to help-option */
  CHAR *helpstr = NULL;
  size_t newlen = 0;
  BOOLEAN haveDevOpt = 0;	/* have we got any 'developer'-options */

  INITSTATUS (status, "LALUserVarHelpString", USERINPUTC);
  ATTATCHSTATUSPTR (status);

  ASSERT (UVAR_vars.next, status, USERINPUTH_ENOUVARS,  USERINPUTH_MSGENOUVARS);
  ASSERT (helpstring != NULL, status, USERINPUTH_ENULL, USERINPUTH_MSGENULL);
  ASSERT ( *helpstring == NULL, status, USERINPUTH_ENONULL, USERINPUTH_MSGENONULL);

  /* prepare first lines of help-string: info about config-file reading */
  newlen = 0;
  sprintf (strbuf, "Usage: %s [@ConfigFile] [options], where options are:\n\n", progname);
  if ( (helpstr = LALCalloc (1, strlen(strbuf) + 1)) == NULL) {
    ABORT (status, USERINPUTH_EMEM, USERINPUTH_MSGEMEM);
  }
  strcpy (helpstr, strbuf);
  newlen += strlen (strbuf) + 1;

  /* put together the help-string. Allocate memory on-the-fly... */
  ptr = &UVAR_vars;

  /* special treatment of debug-option in the head (if present) */
  if ( ptr->help && ptr->optchar )
    {

      sprintf (strbuf, "  -%c    %-15s %-6s   %s [%d] \n",
	       ptr->optchar, " ", typestr[ptr->type], ptr->help, *(INT4*)(ptr->varp) );
      newlen += strlen (strbuf);
      helpstr = LALRealloc (helpstr, newlen);
      if ( helpstr == NULL) {
	ABORT (status, USERINPUTH_EMEM, USERINPUTH_MSGEMEM);
      }

      strcat (helpstr, strbuf);	/* add this line to the helpstring */
    }

  /* FIRST PASS: treat all "normal" entries excluding DEVELOPER-options */
  while ( (ptr=ptr->next) != NULL)
    {
      if (ptr->state & UVAR_DEVELOPER)
	continue;	/* skip developer-options to be treated a second pass */

      if (ptr->state & UVAR_REQUIRED)
	strcpy (defaultstr, "REQUIRED");
      else if (ptr->state & UVAR_HELP)
	{
	  helpptr = ptr;	/* keep a pointer to the help-option for later */
	  strcpy (defaultstr, "");
	}
      else /* write the current default-value into a string */
	{
	  CHAR *valstr = NULL;
	  TRY ( UvarValue2String(status->statusPtr, &valstr, ptr), status);
	  strncpy (defaultstr, valstr, UVAR_MAXDEFSTR);	/* cut short for default-entry */
	  defaultstr[UVAR_MAXDEFSTR-1] = 0;
	  LALFree (valstr);
	  valstr=NULL;
	}
      
      if (ptr->optchar != 0)
	sprintf (optstr, "-%c,", ptr->optchar);
      else
	strcpy (optstr, "   ");

      LALSnprintf (strbuf, UVAR_MAXHELPLINE,  "  %s --%-15s %-6s   %s [%s] \n", 
		   optstr,
		   ptr->name ? ptr->name : "-NONE-", 
		   typestr[ptr->type], 
		   ptr->help ? ptr->help : "-NONE-",
		   defaultstr);

      /* now increase allocated memory by the right amount */
      newlen += strlen (strbuf);
      helpstr = LALRealloc (helpstr, newlen);
      if ( helpstr == NULL) {
	ABORT (status, USERINPUTH_EMEM, USERINPUTH_MSGEMEM);
      }

      strcat (helpstr, strbuf);	/* add this line to the helpstring */

    } /* while ptr->next */

  /* ---------- SECOND PASS through user-options: 
   * show DEVELOPER-options only if lalDebugLevel >= 1 
   */
  if ( lalDebugLevel == 0)	/* only give instructions as to how to see developer-options */
    {
      CHAR buf[256];
      if ( UVAR_vars.optchar )
	sprintf (buf, "(e.g. --%s -%c1)", helpptr->name, UVAR_vars.optchar);
      else
	sprintf (buf, " ");

      sprintf (strbuf, "\n ----- Hint: use help with lalDebugLevel > 0 %s to see all 'developer-options' ----- \n", buf);
      newlen += strlen (strbuf);
      helpstr = LALRealloc (helpstr, newlen);
      if ( helpstr == NULL) {
	ABORT (status, USERINPUTH_EMEM, USERINPUTH_MSGEMEM);
      }

      strcat (helpstr, strbuf);	/* add this line to the helpstring */
    }
  else	/* lalDebugLevel > 0 */
    {
      strcpy(strbuf, 
	     "\n   ---------- The following are 'Developer'-options not useful " 
	     "for most users:----------\n\n");
      newlen += strlen (strbuf);
      helpstr = LALRealloc (helpstr, newlen);
      if ( helpstr == NULL) {
	ABORT (status, USERINPUTH_EMEM, USERINPUTH_MSGEMEM);
      }

      strcat (helpstr, strbuf);	/* add this line to the helpstring */
  
      ptr = &UVAR_vars;
      while ( (ptr=ptr->next) != NULL )
	{
	  CHAR *valstr = NULL;

	  if ( ! (ptr->state & UVAR_DEVELOPER) )	/* only treat developer-options */
	    continue;
	  
	  haveDevOpt = 1;
	  
	  TRY ( UvarValue2String(status->statusPtr, &valstr, ptr), status);
	  strncpy (defaultstr, valstr, UVAR_MAXDEFSTR);	/* cut short for default-entry */
	  defaultstr[UVAR_MAXDEFSTR-1] = 0;
	  LALFree (valstr);
	  valstr = NULL;
	  
	  if (ptr->optchar != 0)
	    sprintf (optstr, "-%c,", ptr->optchar);
	  else
	    strcpy (optstr, "   ");
	  
	  LALSnprintf (strbuf, UVAR_MAXHELPLINE,  "  %s --%-15s %-6s   %s [%s] \n", 
		       optstr,
		       ptr->name ? ptr->name : "-NONE-", 
		       typestr[ptr->type], 
		       ptr->help ? ptr->help : "-NONE-",
		       defaultstr);
	  
	  /* now increase allocated memory by the right amount */
	  newlen += strlen (strbuf);
	  helpstr = LALRealloc (helpstr, newlen);
	  if ( helpstr == NULL) {
	    ABORT (status, USERINPUTH_EMEM, USERINPUTH_MSGEMEM);
	  }
	  
	  strcat (helpstr, strbuf);	/* add this line to the helpstring */
	  
	} /* while ptr->next: 2nd PASS for DEVELOPER-options */
      
      if ( !haveDevOpt )	/* no developer-options found: say something */
	{
	  strcpy(strbuf, "   -- NONE --\n\n");
	  newlen += strlen (strbuf);
	  helpstr = LALRealloc (helpstr, newlen);
	  if ( helpstr == NULL) {
	    ABORT (status, USERINPUTH_EMEM, USERINPUTH_MSGEMEM);
	  }
	  
	  strcat (helpstr, strbuf);	/* add this line to the helpstring */
	} /* if !haveDevOpt */

    } /* if lalDebugLevel: output developer-options */

  /* finished: return final helpstring */
  *helpstring = helpstr;

  DETATCHSTATUSPTR(status);
  RETURN(status);

} /* LALUserVarHelpString() */


/** Put all the pieces together, and basically does everything: 
 * get config-filename from cmd-line (if found),
 * then interpret config-file and then the command-line
 */
/* <lalVerbatim file="UserInputCP"> */
void
LALUserVarReadAllInput (LALStatus *status, int argc, char *argv[])
{ /* </lalVerbatim> */

  INT4 i;
  CHAR* fname = NULL;
  CHAR *tmp;
  LALUserVariable *ptr;

  INITSTATUS( status, "LALUserVarReadAllInput", USERINPUTC);

  ASSERT (UVAR_vars.next, status, USERINPUTH_ENOUVARS,  USERINPUTH_MSGENOUVARS);

  ATTATCHSTATUSPTR (status); 

  /* pre-process command-line: have we got a config-file ? */
  for (i=1; i < argc; i++)
    {
      tmp = argv[i];
      if ( *tmp == '@' )
	{
	  if (fname != NULL) {
	    ABORT (status, USERINPUTH_EONECONFIG, USERINPUTH_MSGEONECONFIG);
	  }

	  tmp ++;
	  if ( (fname = LALCalloc (1, strlen(tmp) + 5 )) == NULL) {
	    ABORT (status, USERINPUTH_EMEM,  USERINPUTH_MSGEMEM);
	  }
	  /* NOTE: if the filename given is not a relative or absolute path, 
	   * we want to ensure it is interpreted relative to the CURRENT directory,
	   * NOT relative to LAL_DATA_PATH (if set), as we cannot rely on it containg
	   * the local directory.
	   *
	   * ==> therefore we ensure that the path is relative to "./" in that case.
	   */
	  if ( (tmp[0] != '.') && (tmp[0] != '/') )
	    sprintf (fname, "./%s", tmp);
	  else
	    strcpy (fname, tmp);
	
	} /* if argument starts with '@' */

    } /* for i < argc */


  /* if config-file specified, read from that first */
  if (fname) {
    TRY (LALUserVarReadCfgfile (status->statusPtr, fname), status);
    LALFree (fname);
    fname=NULL;
  }

  /* now do proper cmdline parsing: overloads config-file settings */
  TRY (LALUserVarReadCmdline (status->statusPtr, argc, argv), status);

  /* now check if help-string was requested */
  ptr = &UVAR_vars;
  while ( (ptr=ptr->next) != NULL)
    if ( (ptr->state & UVAR_HELP) && (ptr->state & UVAR_WAS_SET) )
      {
	CHAR *helpstring = NULL;
	TRY (LALUserVarHelpString (status->statusPtr, &helpstring, argv[0]), status);
	printf ("\n%s\n", helpstring);
	LALFree (helpstring);
	helpstring=NULL;
	DETATCHSTATUSPTR (status);
	RETURN (status);
      } /* if help requested */

  /* check that all required input-variables have been specified */
  TRY (LALUserVarCheckRequired (status->statusPtr), status);


  DETATCHSTATUSPTR (status);
  RETURN (status);
  
} /* LALReadUserInput() */


/** Has this user-variable been set by the user?
 * return -1 on error, TRUE/FALSE otherwise
 */
/* <lalVerbatim file="UserInputCP"> */
INT4
LALUserVarWasSet (const void *cvar)
{ /* </lalVerbatim> */

  LALUserVariable *ptr;

  if (!cvar)
    return (-1);


  /* find this varname in the list of user-variables */
  ptr = &UVAR_vars;
  while ( (ptr = ptr->next) != NULL) 
    if ( ptr->varp == cvar)
      break;

  if (ptr == NULL) {
    LALPrintError ("Variable passed to UVARwasSet is not a registered Usser-variable\n");
    return (-1);
  }
  
  /* we found it: has it been set by user? */
  return ( (ptr->state & UVAR_WAS_SET) != 0 );

} /* LALUserVarWasSet() */

/** Check that all required user-variables have been set successfully.
 * Print error if not 
 */
void
LALUserVarCheckRequired (LALStatus *status)
{
  LALUserVariable *ptr;  

  INITSTATUS( status, "LALUserVarCheckRequired", USERINPUTC);

  /* go through list of uvars */
  ptr = &UVAR_vars;
  while ( (ptr = ptr->next) != NULL) 
    if ( (ptr->state & UVAR_REQUIRED) && !(ptr->state & UVAR_WAS_SET))
      {
	LALPrintError ("\nRequired user-variable `%s` has not been specified!\n\n", ptr->name);
	ABORT (status, USERINPUTH_ENOTSET, USERINPUTH_MSGENOTSET);
      }

  RETURN (status);

} /* LALUserVarCheckRequired() */

/** Handle the delicate setting of lalDebuglevel.
 */
void
LALGetDebugLevel (LALStatus *status, int argc, char *argv[], CHAR optchar)
{
  static const char *help = "set lalDebugLevel";
  static INT4 defaultDebugLevel;
  INT4 i;
  CHAR *ptr;

  INITSTATUS( status, "UVARgetDebugLevel", USERINPUTC);

  ASSERT (argv, status,  USERINPUTH_ENULL, USERINPUTH_MSGENULL);
  ASSERT (UVAR_vars.next == NULL, status, USERINPUTH_EDEBUG,  USERINPUTH_MSGEDEBUG);
  ASSERT (UVAR_vars.varp == NULL, status, USERINPUTH_EDEBUG,  USERINPUTH_MSGEDEBUG);

  /* "register" the debug-level variable in the head of the UVAR-list, 
   * to avoid any mallocs. We need this to show up in the help-string */
  UVAR_vars.name = NULL;
  UVAR_vars.type = UVAR_INT4;
  UVAR_vars.optchar = optchar;
  UVAR_vars.help = help;

  defaultDebugLevel = lalDebugLevel;
  UVAR_vars.varp = &defaultDebugLevel;	/* trick: use to store default-value (for help-string) */

  UVAR_vars.state = UVAR_OPTIONAL;
  UVAR_vars.next = NULL;
  
  /* the command-line has to be processed by hand for this... ! */
  for (i=1; i < argc; i++)
    {
      if ( (argv[i][0] == '-') && (argv[i][1] == optchar) )
	{
	  if (argv[i][2] != '\0')
	    ptr = argv[i]+2;
	  else
	    ptr = argv[i+1];

	  if ( (ptr==NULL) || (sscanf ( ptr, "%d", &lalDebugLevel) != 1) ) {
	    LALPrintError ("setting debug-level `-%c` requires an argument\n", optchar);
	    ABORT (status, USERINPUTH_EOPT, USERINPUTH_MSGEOPT);
	  }
	  break;
	} /* if debug-switch found */
      
    } /* for i < argc */

  RETURN (status);

} /* LALGetDebugLevel() */

/** Return a log-string representing the <em>complete</em> user-input.
 * <em>NOTE:</em> we only record user-variables that have been set
 * by the user.
 * \param[out] **outstr	the string containing the user-input record.
 * \param[in] format	return as config-file or command-line
 */
void
LALUserVarGetLog (LALStatus *status, CHAR **logstr,  UserVarLogFormat format)
{
  LALUserVariable *ptr = NULL;
  CHAR *record = NULL;
  CHAR *valstr;		/* buffer to hold value-string */
  CHAR *append;
  UINT4 len, appendlen;

  INITSTATUS( status, "LALUserVarGetLog", USERINPUTC);
  ATTATCHSTATUSPTR(status);

  ASSERT (logstr, status, USERINPUTH_ENULL, USERINPUTH_MSGENULL);
  ASSERT (*logstr == NULL, status, USERINPUTH_ENONULL,USERINPUTH_MSGENONULL);

  /* initialize return-string */
  record = LALMalloc (1);
  record[0] = 0;
  len = 0;

  ptr = &UVAR_vars;
  while ( (ptr = ptr->next) )   /* we skip the possible lalDebugLevel-entry for now (FIXME?) */
    {
      if ( (ptr->state & UVAR_WAS_SET) == FALSE )	/* skip unset variables */
	continue;

      valstr = NULL;
      TRY ( UvarValue2String (status->statusPtr, &valstr, ptr), status);

      appendlen = strlen(ptr->name) + strlen(valstr) + 10; 
      if ( (append = LALMalloc(appendlen)) == NULL) {
	ABORT (status,  USERINPUTH_EMEM,  USERINPUTH_MSGEMEM);
      }

      switch (format)
	{
	case UVAR_LOGFMT_CFGFILE:
	  sprintf (append, "%s = %s\n", ptr->name, valstr);
	  break;
	case UVAR_LOGFMT_CMDLINE:
	  sprintf (append, " --%s=%s", ptr->name, valstr);
	  break;
	default:
	  ABORT (status, USERINPUTH_ERECFORMAT, USERINPUTH_MSGERECFORMAT);
	  break;
	} /* switch (format) */

      len += strlen(append);
      if ( (record = LALRealloc (record, len+1)) == NULL ) {
	ABORT (status,  USERINPUTH_EMEM,  USERINPUTH_MSGEMEM);
      }

      strcat (record, append);

      LALFree (valstr);
      valstr=NULL;
      LALFree (append);
      append=NULL;
    } /* while ptr->next */
  
  *logstr = record;

  DETATCHSTATUSPTR(status);
  RETURN (status);

} /* LALUserVarGetLog() */


/* Return the value of the given UserVariable as a string.
 * For INTERNAL use only!
 */
void
UvarValue2String (LALStatus *status, CHAR **outstr, LALUserVariable *uvar)
{
  CHAR *str = NULL;
  CHAR *ptr;
  CHAR buf[512];	/* buffer for producing non-string values */

  INITSTATUS( status, "Value2String", USERINPUTC);

  ASSERT (outstr, status, USERINPUTH_ENULL, USERINPUTH_MSGENULL);
  ASSERT (*outstr == NULL, status, USERINPUTH_ENONULL,USERINPUTH_MSGENONULL);
  ASSERT (uvar, status, USERINPUTH_ENULL, USERINPUTH_MSGENULL);
  ASSERT (uvar->varp, status, USERINPUTH_ENULL, USERINPUTH_MSGENULL);

  switch (uvar->type)	      /* info on default-value for this variable */
    {
    case UVAR_BOOL:
      sprintf (buf, *(BOOLEAN*)(uvar->varp) ? "TRUE" : "FALSE");
      break;
    case UVAR_INT4:
      sprintf (buf, "%" LAL_INT4_FORMAT, *(INT4*)(uvar->varp) );
      break;
    case UVAR_REAL8:
      if (*(REAL8*)(uvar->varp) == 0)
	strcpy (buf, "0.0");	/* makes it more explicit that's it a REAL */
      else
	sprintf (buf, "%g", *(REAL8*)(uvar->varp) );
      break;
    case UVAR_STRING:
      ptr = *(CHAR**)(uvar->varp);
      if ( ptr != NULL )
	{
	  if ( (str = LALMalloc ( strlen(ptr) + 3 )) == NULL) {
	    ABORT (status,  USERINPUTH_EMEM,  USERINPUTH_MSGEMEM);
	  }
	  sprintf (str, "\"%s\"", ptr);
	}
      else
	strcpy (buf, "NULL");
      break;
      
    default:
      LALPrintError ("ERROR: unkown UserVariable-type encountered... this points to a coding error!\n");
      ABORT (status, USERINPUTH_ENULL, USERINPUTH_MSGENULL);
      break;
      
    } /* switch uvar->type */

  if (str == NULL)
    {
      if ( (str = LALMalloc (strlen(buf) + 1)) == NULL) {
	ABORT (status,  USERINPUTH_EMEM,  USERINPUTH_MSGEMEM);
      }
      strcpy (str, buf);
    }
  
  *outstr = str;
	
  RETURN (status);

} /* UvarValue2String() */


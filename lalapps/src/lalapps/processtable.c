/*
 * Copyright (C) 2007-2012 Duncan Brown, Jolien Creighton, Kipp Cannon,
 * Reinhard Prix, Bernd Machenschalk
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation; either version 2 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with with program; see the file COPYING. If not, write to the Free
 * Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
 * 02111-1307  USA
 */

#ifndef _WIN32

#define _GNU_SOURCE

#include <config.h>

#include <ctype.h>
#include <pwd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <time.h>
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

#include <lal/Date.h>
#include <lal/LIGOMetadataTables.h>
#include <lal/XLALError.h>


#include <lalapps.h>
#include "processtable.h"


/**
 * Parses a cvs keyword string in the form $keyword:value$, returning a
 * newly-malloc()'ed string containing just the value.  Leading and
 * trailing whitespace is removed from the value string.  Returns NULL on
 * parse or malloc() failure.
 */


static char *cvs_get_keyword_value(const char *cvs_string)
{
	const char *value_start;
	const char *value_end;
	char *value;

	/*
	 * check for input
	 */

	if(!cvs_string)
		return NULL;

	/*
	 * string must start with '$'
	 */

	value_start = strchr(cvs_string, '$');
	if(!value_start)
		return NULL;

	/*
	 * keyword ends at ':'
	 */

	value_end = strchr(value_start, ':');
	if(!value_end)
	  {
	    /*
	     * allow for unset keyword
	     */
	    value_start = strchr(value_start, '$');
	    if(!value_start)
	      return NULL;
	    else
	      --value_start;
	  }
	else
	  {
	    value_start = value_end;
	  }

	/*
	 * skip leading white space
	 */

	while(isspace(*++value_start));
	if(!*value_start)
		return NULL;

	/*
	 * string must end with '$'
	 */

	value_end = strchr(value_start, '$');
	if(!value_end)
		return NULL;

	/*
	 * skip trailing white space
	 */

	while(isspace(*--value_end));
	value_end++;
	if(value_end - value_start < 0)
		value_end = value_start;

	/*
	 * extract value.  +1 for the '\0' to be added
	 */

	value = malloc((value_end - value_start + 1) * sizeof(*value));
	if(!value)
		return NULL;
	memcpy(value, value_start, (value_end - value_start) * sizeof(*value));
	value[value_end - value_start] = '\0';

	/*
	 * done
	 */

	return value;
}


/**
 * Populate a pre-allocated ProcessTable structure.
 */


int XLALPopulateProcessTable(
	ProcessTable *ptable,
	const char *program_name,
	const char *cvs_revision,
	const char *cvs_source,
	const char *cvs_date,
	long process_id
)
{
	char *cvs_keyword_value;
	uid_t uid;
#if 0
	struct passwd *pw;
#endif
	struct tm utc;

	/*
	 * program name entry
	 */

	snprintf(ptable->program, LIGOMETA_PROGRAM_MAX, "%s", program_name);

	/*
	 * cvs version
	 */

	cvs_keyword_value = cvs_get_keyword_value(cvs_revision);
	if(!cvs_keyword_value) {
		XLALPrintError("%s(): cannot parse \"%s\"\n", __func__, cvs_revision);
		XLAL_ERROR(XLAL_EINVAL);
	}
	snprintf(ptable->version, LIGOMETA_VERSION_MAX, "%s", cvs_keyword_value);
	free(cvs_keyword_value);

	/*
	 * cvs repository
	 */

	cvs_keyword_value = cvs_get_keyword_value(cvs_source);
	if(!cvs_keyword_value) {
		XLALPrintError("%s(): cannot parse \"%s\"\n", __func__, cvs_source);
		XLAL_ERROR(XLAL_EINVAL);
	}
	snprintf(ptable->cvs_repository, LIGOMETA_CVS_REPOSITORY_MAX, "%s", cvs_keyword_value);
	free(cvs_keyword_value);

	/*
	 * cvs check-in time
	 */

	cvs_keyword_value = cvs_get_keyword_value(cvs_date);
	if(!cvs_keyword_value) {
		XLALPrintError("%s(): cannot parse \"%s\"\n", __func__, cvs_date);
		XLAL_ERROR(XLAL_EINVAL);
	}
	if(!strptime(cvs_keyword_value, "%Y/%m/%d %T", &utc))
	  {
	    if(!strptime(cvs_keyword_value, "%Y-%m-%d %T", &utc))
	      {
		XLALPrintError("%s(): cannot parse \"%s\"\n", __func__, cvs_keyword_value);
		free(cvs_keyword_value);
		XLAL_ERROR(XLAL_EINVAL);
	      }
	  }
	free(cvs_keyword_value);
	XLALClearErrno();
	XLALGPSSet(&ptable->cvs_entry_time, XLALUTCToGPS(&utc), 0);
	if(XLALGetBaseErrno())
		XLAL_ERROR(XLAL_EFUNC);

	/*
	 * comment
	 */

	snprintf(ptable->comment, LIGOMETA_COMMENT_MAX, " ");

	/*
	 * online flag and domain
	 */

	ptable->is_online = 0;
	snprintf(ptable->domain, LIGOMETA_DOMAIN_MAX, "lalapps");

	/*
	 * unix process id, username, host, and process_id
	 */

	ptable->unix_procid = getpid();
	if(!ptable->unix_procid)
		ptable->unix_procid = getppid();
	if(gethostname(ptable->node, LIGOMETA_NODE_MAX) < 0) {
		perror("could not determine host name");
		XLAL_ERROR(XLAL_ESYS);
	}
	uid = geteuid();
#if 0
	if(!(pw = getpwuid(uid)))
#endif
		snprintf(ptable->username, LIGOMETA_USERNAME_MAX, "%d", uid);
#if 0
	else
		snprintf(ptable->username, LIGOMETA_USERNAME_MAX, "%s", pw->pw_name);
#endif
	ptable->process_id = process_id;

	/*
	 * done
	 */

	return 0;
}

#else /* _WIN32 */

#include <stdio.h>

int XLALPopulateProcessTable(
	ProcessTable *ptable,
	const char *program_name,
	const char *cvs_revision,
	const char *cvs_source,
	const char *cvs_date,
	long process_id
) {
  fprintf(stderr, "XLALPopulateProcessTable() not implemented for WIN32\n");
  return 1;
}

#endif /* __WIN32 */

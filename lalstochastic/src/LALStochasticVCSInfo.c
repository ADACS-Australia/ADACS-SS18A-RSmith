/*
 * LALStochasticVCSInfo.c - LALStochastic VCS Information
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with with program; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston,
 * MA 02111-1307 USA
 *
 * Copyright (C) 2009,2010 Adam Mercer
 */

#include <lal/LALVCSInfo.h>

#include <lal/LALStochasticConfig.h>
#include <lal/LALStochasticVCSInfo.h>

/* global variables for vcs information */
const char *lalStochasticVCSVersion = LALSTOCHASTIC_VERSION;
const char *lalStochasticVCSId = LALSTOCHASTIC_VCS_ID;
const char *lalStochasticVCSDate = LALSTOCHASTIC_VCS_DATE;
const char *lalStochasticVCSBranch = LALSTOCHASTIC_VCS_BRANCH;
const char *lalStochasticVCSTag = LALSTOCHASTIC_VCS_TAG;
const char *lalStochasticVCSAuthor = LALSTOCHASTIC_VCS_AUTHOR;
const char *lalStochasticVCSCommitter = LALSTOCHASTIC_VCS_COMMITTER;
const char *lalStochasticVCSStatus = LALSTOCHASTIC_VCS_STATUS;

/* global variables for vcs information - identable */
const char *lalStochasticVCSIdentId = LALSTOCHASTIC_VCS_IDENT_ID;
const char *lalStochasticVCSIdentDate = LALSTOCHASTIC_VCS_IDENT_DATE;
const char *lalStochasticVCSIdentBranch = LALSTOCHASTIC_VCS_IDENT_BRANCH;
const char *lalStochasticVCSIdentTag = LALSTOCHASTIC_VCS_IDENT_TAG;
const char *lalStochasticVCSIdentAuthor = LALSTOCHASTIC_VCS_IDENT_AUTHOR;
const char *lalStochasticVCSIdentCommitter = LALSTOCHASTIC_VCS_IDENT_COMMITTER;
const char *lalStochasticVCSIdentStatus = LALSTOCHASTIC_VCS_IDENT_STATUS;

/* vcs information structure */
struct tagLALVCSInfo lalStochasticVCSInfo = { \
  LALSTOCHASTIC_VCS_NAME, \
  LALSTOCHASTIC_VERSION, \
  LALSTOCHASTIC_VCS_ID, \
  LALSTOCHASTIC_VCS_DATE, \
  LALSTOCHASTIC_VCS_BRANCH, \
  LALSTOCHASTIC_VCS_TAG, \
  LALSTOCHASTIC_VCS_AUTHOR, \
  LALSTOCHASTIC_VCS_COMMITTER, \
  LALSTOCHASTIC_VCS_STATUS \
};

/*
 * vim: tw=0 ts=2 et
 */

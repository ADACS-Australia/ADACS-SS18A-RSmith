/*
*  Copyright (C) 2010 Evan Goetz
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

#ifndef __CANDIDATES_H__
#define __CANDIDATES_H__

#include "TwoSpectTypes.h"


candidate * new_candidate(void);
void free_candidate(candidate *cand);
void loadCandidateData(candidate *out, REAL8 fsig, REAL8 period, REAL8 moddepth, REAL4 ra, REAL4 dec, REAL8 stat, REAL8 snr, REAL8 prob);
void clusterCandidates(candidate *out[], candidate *in[], ffdataStruct *ffdata, inputParamsStruct *params, REAL8Vector *ffplanenoise, REAL8Vector *fbinaveratios, INT4 numofcandidates, INT4 option);





#endif



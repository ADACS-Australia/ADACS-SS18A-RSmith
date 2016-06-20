/*
 *  Copyright (C) 2016 John Veitch
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


#ifndef LALInferenceHDF5_h
#define LALInferenceHDF5_h

#include <lal/LALInference.h>
#include <lal/H5FileIO.h>

int XLALInferenceVariablesArray2H5Group(LALH5File *h5file, LALInferenceVariables *const *const varsArray, UINT4 N, const char *GroupName);

const char LALInferenceHDF5PosteriorGroupName[]="posterior_samples";

#endif /* LALInferenceHDF5_h */

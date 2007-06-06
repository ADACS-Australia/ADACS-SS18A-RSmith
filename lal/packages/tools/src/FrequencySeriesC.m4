dnl $Id$
dnl
dnl Copyright (C) 2007  Kipp Cannon
dnl
dnl This program is free software; you can redistribute it and/or modify it
dnl under the terms of the GNU General Public License as published by the
dnl Free Software Foundation; either version 2 of the License, or (at your
dnl option) any later version.
dnl
dnl This program is distributed in the hope that it will be useful, but
dnl WITHOUT ANY WARRANTY; without even the implied warranty of
dnl MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
dnl Public License for more details.
dnl
dnl You should have received a copy of the GNU General Public License along
dnl with this program; if not, write to the Free Software Foundation, Inc.,
dnl 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

define(`SERIESTYPE',DATATYPE`FrequencySeries')
define(`SEQUENCETYPE',DATATYPE`Sequence')
void `XLALDestroy'SERIESTYPE (
	SERIESTYPE *series
)
{
	if(series)
		`XLALDestroy'SEQUENCETYPE (series->data);
	XLALFree(series);
}


void `LALDestroy'SERIESTYPE (
	LALStatus *status,
	SERIESTYPE *series
)
{
	INITSTATUS(status, "`LALDestroy'SERIESTYPE", FREQUENCYSERIESC);
	`XLALDestroy'SERIESTYPE (series);
	RETURN(status);
}


SERIESTYPE *`XLALCreate'SERIESTYPE (
	const CHAR *name,
	const LIGOTimeGPS *epoch,
	REAL8 f0,
	REAL8 deltaF,
	const LALUnit *sampleUnits,
	size_t length
)
{
	static const char *func = "`XLALCreate'SERIESTYPE";
	SERIESTYPE *new;
	SEQUENCETYPE *sequence;

	new = XLALMalloc(sizeof(*new));
	sequence = `XLALCreate'SEQUENCETYPE (length);
	if(!new || !sequence) {
		XLALFree(new);
		`XLALDestroy'SEQUENCETYPE (sequence);
		XLAL_ERROR_NULL(func, XLAL_ENOMEM);
	}

	if(name)
		strncpy(new->name, name, LALNameLength);
	else
		new->name[0] = '\0';
	new->epoch = *epoch;
	new->f0 = f0;
	new->deltaF = deltaF;
	new->sampleUnits = *sampleUnits;
	new->data = sequence;

	return(new);
}


void `LALCreate'SERIESTYPE (
	LALStatus *status,
	SERIESTYPE **output,
	const CHAR *name,
	LIGOTimeGPS epoch,
	REAL8 f0,
	REAL8 deltaF,
	LALUnit sampleUnits,
	size_t length
)
{
	INITSTATUS(status, "`LALCreate'SERIESTYPE", FREQUENCYSERIESC);
	ASSERT(output != NULL, status, LAL_NULL_ERR, LAL_NULL_MSG);
	*output = `XLALCreate'SERIESTYPE (name, &epoch, f0, deltaF, &sampleUnits, length);
	if(*output == NULL) {
		XLALClearErrno();
		ABORT(status, LAL_NOMEM_ERR, LAL_NOMEM_MSG);
	}
	RETURN(status);
}


SERIESTYPE *`XLALCut'SERIESTYPE (
	const SERIESTYPE *series,
	size_t first,
	size_t length
)
{
	static const char *func = "`XLALCut'SERIESTYPE";
	SERIESTYPE *new;
	SEQUENCETYPE *sequence;

	if(!series || !series->data)
		return(NULL);

	new = XLALMalloc(sizeof(*new));
	sequence = `XLALCut'SEQUENCETYPE (series->data, first, length);
	if(!new || !sequence) {
		XLALFree(new);
		`XLALDestroy'SEQUENCETYPE (sequence);
		XLAL_ERROR_NULL(func, XLAL_ENOMEM);
	}

	*new = *series;
	new->data = sequence;
	new->f0 += first * new->deltaF;

	return(new);
}


size_t `XLALResize'SERIESTYPE (
	SERIESTYPE *series,
	int first,
	size_t length
)
{
	if(!series)
		return(0);

	series->f0 += first * series->deltaF;
	return(`XLALResize'SEQUENCETYPE (series->data, first, length));
}


size_t `XLALShrink'SERIESTYPE (
	SERIESTYPE *series,
	size_t first,
	size_t length
)
{
	return(`XLALResize'SERIESTYPE (series, first, length));
}


void `LALShrink'SERIESTYPE (
	LALStatus *status,
	SERIESTYPE *series,
	size_t first,
	size_t length
)
{
	size_t new_length;

	INITSTATUS(status, "`LALShrink'SERIESTYPE", FREQUENCYSERIESC);

	ASSERT(series != NULL, status, LAL_NULL_ERR, LAL_NULL_MSG);
	ASSERT(series->data != NULL, status, LAL_NULL_ERR, LAL_NULL_MSG);
	ASSERT(first + length <= series->data->length, status, LAL_RANGE_ERR, LAL_RANGE_MSG);
	new_length = `XLALShrink'SERIESTYPE (series, first, length);
	if(new_length != length) {
		XLALClearErrno();
		ABORT(status, LAL_FAIL_ERR, LAL_FAIL_MSG);
	}
	RETURN(status);
}


SERIESTYPE *`XLALAdd'SERIESTYPE (
	SERIESTYPE *arg1,
	const SERIESTYPE *arg2
)
{
	static const char *func = "`XLALAdd'SERIESTYPE";
	int offset = (arg2->f0 - arg1->f0) / arg1->deltaF;
	REAL8 ratio = XLALUnitRatio(&arg1->sampleUnits, &arg2->sampleUnits);
	unsigned int i;

	/* make sure arguments are compatible */
	if(XLALGPSCmp(&arg1->epoch, &arg2->epoch) || (arg1->deltaF != arg2->deltaF) || XLALIsREAL8FailNaN(ratio))
		XLAL_ERROR_NULL(func, XLAL_EDATA);
	/* FIXME: generalize to relax this requirement */
	if((arg2->f0 < arg1->f0) || (offset + arg2->data->length > arg1->data->length))
		XLAL_ERROR_NULL(func, XLAL_EBADLEN);
	
	/* add arg2 to arg1, adjusting the units */
	for(i = 0; i < arg2->data->length; i++) {
		ifelse(DATATYPE, COMPLEX8,
		arg1->data->data[offset + i].re += arg2->data->data[i].re / ratio;
		arg1->data->data[offset + i].im += arg2->data->data[i].im / ratio;
		, DATATYPE, COMPLEX16,
		arg1->data->data[offset + i].re += arg2->data->data[i].re / ratio;
		arg1->data->data[offset + i].im += arg2->data->data[i].im / ratio;
		, 
		arg1->data->data[offset + i] += arg2->data->data[i] / ratio;)
	}

	return(arg1);
}


SERIESTYPE *`XLALSubtract'SERIESTYPE (
	SERIESTYPE *arg1,
	const SERIESTYPE *arg2
)
{
	static const char *func = "`XLALAdd'SERIESTYPE";
	int offset = (arg2->f0 - arg1->f0) / arg1->deltaF;
	REAL8 ratio = XLALUnitRatio(&arg1->sampleUnits, &arg2->sampleUnits);
	unsigned int i;

	/* make sure arguments are compatible */
	if(XLALGPSCmp(&arg1->epoch, &arg2->epoch) || (arg1->deltaF != arg2->deltaF) || XLALIsREAL8FailNaN(ratio))
		XLAL_ERROR_NULL(func, XLAL_EDATA);
	/* FIXME: generalize to relax this requirement */
	if((arg2->f0 < arg1->f0) || (offset + arg2->data->length > arg1->data->length))
		XLAL_ERROR_NULL(func, XLAL_EBADLEN);
	
	/* subtract arg2 from arg1, adjusting the units */
	for(i = 0; i < arg2->data->length; i++) {
		ifelse(DATATYPE, COMPLEX8,
		arg1->data->data[offset + i].re -= arg2->data->data[i].re / ratio;
		arg1->data->data[offset + i].im -= arg2->data->data[i].im / ratio;
		, DATATYPE, COMPLEX16,
		arg1->data->data[offset + i].re -= arg2->data->data[i].re / ratio;
		arg1->data->data[offset + i].im -= arg2->data->data[i].im / ratio;
		, 
		arg1->data->data[offset + i] -= arg2->data->data[i] / ratio;)
	}

	return(arg1);
}


ifelse(DATATYPE, COMPLEX8,
SERIESTYPE *`XLALConjugate'SERIESTYPE (
	SERIESTYPE *series
)
{
	`XLALConjugate'SEQUENCETYPE (series->data);
	return(series);

}
, DATATYPE, COMPLEX16,
SERIESTYPE *`XLALConjugate'SERIESTYPE (
	SERIESTYPE *series
)
{
	`XLALConjugate'SEQUENCETYPE (series->data);
	return(series);
}
,)


SERIESTYPE *`XLALMultiply'SERIESTYPE (
	SERIESTYPE *arg1,
	const SERIESTYPE *arg2
)
{
	static const char *func = "`XLALMultiply'SERIESTYPE";
	int offset = (arg2->f0 - arg1->f0) / arg1->deltaF;
	REAL8 ratio = XLALUnitRatio(&arg1->sampleUnits, &arg2->sampleUnits);
	unsigned int i;

	/* make sure arguments are compatible */
	if(XLALGPSCmp(&arg1->epoch, &arg2->epoch) || (arg1->deltaF != arg2->deltaF) || XLALIsREAL8FailNaN(ratio))
		XLAL_ERROR_NULL(func, XLAL_EDATA);
	/* FIXME: generalize to relax this requirement */
	if((arg2->f0 < arg1->f0) || (offset + arg2->data->length > arg1->data->length))
		XLAL_ERROR_NULL(func, XLAL_EBADLEN);
	
	/* multiply arg2 by arg1, adjusting the units */
	for(i = 0; i < arg2->data->length; i++) {
		ifelse(DATATYPE, COMPLEX8,
		REAL4 re = arg2->data->data[i].re / ratio;
		REAL4 im = arg2->data->data[i].im / ratio;
		arg1->data->data[offset + i].re = arg1->data->data[offset + i].re * re - arg1->data->data[offset + i].im * im;
		arg1->data->data[offset + i].im = arg1->data->data[offset + i].re * im + arg1->data->data[offset + i].im * re;
		, DATATYPE, COMPLEX16,
		REAL8 re = arg2->data->data[i].re / ratio;
		REAL8 im = arg2->data->data[i].im / ratio;
		arg1->data->data[offset + i].re = arg1->data->data[offset + i].re * re - arg1->data->data[offset + i].im * im;
		arg1->data->data[offset + i].im = arg1->data->data[offset + i].re * im + arg1->data->data[offset + i].im * re;
		, 
		arg1->data->data[offset + i] *= arg2->data->data[i] / ratio;)
	}

	return(arg1);
}


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

define(`SEQUENCETYPE',DATATYPE`Sequence')
void `XLALDestroy'SEQUENCETYPE (
	SEQUENCETYPE *sequence
)
{
	if(sequence)
		XLALFree(sequence->data);
	XLALFree(sequence);
}


SEQUENCETYPE *`XLALCreate'SEQUENCETYPE (
	size_t length
)
{
	static const char *func = "`XLALCreate'SEQUENCETYPE";
	SEQUENCETYPE *new;
	DATATYPE *data;

	new = XLALMalloc(sizeof(*new));
	data = XLALMalloc(length * sizeof(*data));
	if(!new || !data) {
		XLALFree(new);
		XLALFree(data);
		XLAL_ERROR_NULL(func, XLAL_ENOMEM);
	}

	new->data = data;
	new->length = length;

	return(new);
}


SEQUENCETYPE *`XLALCut'SEQUENCETYPE (
	SEQUENCETYPE *sequence,
	size_t first,
	size_t length
)
{
	static const char *func = "`XLALCut'SEQUENCETYPE";
	SEQUENCETYPE *new = NULL;

	if(sequence && sequence->data) {
		new = `XLALCreate'SEQUENCETYPE (length);
		if(!new)
			XLAL_ERROR_NULL(func, XLAL_EFUNC);
		memcpy(new->data, sequence->data + first, length * sizeof(*new->data));
	}

	return(new);
}


SEQUENCETYPE *`XLALCopy'SEQUENCETYPE (
	SEQUENCETYPE *sequence
)
{
	return(`XLALCut'SEQUENCETYPE (sequence, 0, sequence->length));
}


void `XLALShift'SEQUENCETYPE (
	SEQUENCETYPE *sequence,
	int count
)
{
	if(!sequence || !sequence->data || !count)
		return;

	memshift(sequence->data, sequence->length * sizeof(*sequence->data), count * (int) sizeof(*sequence->data));
	if((size_t) labs(count) >= sequence->length)
		memset(sequence->data, 0, sequence->length * sizeof(*sequence->data));
	else if(count > 0)
		memset(sequence->data, 0, count * sizeof(*sequence->data));
	else
		memset(sequence->data + sequence->length + count, 0, -count * sizeof(*sequence->data));
}


/* FIXME: this function does not take care to move and zero the least
 * number of bytes possible.  A performance gain would be realized by being
 * more careful. */
/* FIXME: this function does not conform to the XLAL error reporting
 * convention. */
size_t `XLALResize'SEQUENCETYPE (
	SEQUENCETYPE *sequence,
	int first,
	size_t length
)
{
	if(!sequence || !sequence->data)
		return(0);

	if(length > sequence->length) {
		/* need to increase memory */
		sequence->data = XLALRealloc(sequence->data, length * sizeof(*sequence->data));

		if(sequence->data) {
			memset(sequence->data + sequence->length, 0, (length - sequence->length) * sizeof(*sequence->data));
			sequence->length = length;
			`XLALShift'SEQUENCETYPE (sequence, -first);
		} else
			sequence->length = 0;
	} else {
		/* do not need to increase memory */
		`XLALShift'SEQUENCETYPE (sequence, -first);
		sequence->data = XLALRealloc(sequence->data, length * sizeof(*sequence->data));
		if(sequence->data)
			sequence->length = length;
		else
			sequence->length = 0;
	}

	return(sequence->length);
}


size_t `XLALShrink'SEQUENCETYPE (
	SEQUENCETYPE *sequence,
	size_t first,
	size_t length
)
{
	return(`XLALResize'SEQUENCETYPE (sequence, first, length));
}


ifelse(DATATYPE, COMPLEX8, `#if 0', DATATYPE, COMPLEX16, `#if 0', `#if 1')
/* FIXME: too bad we aren't C99, or we could include versions for complex
 * data types */
DATATYPE `XLAL'DATATYPE`Sum' (
	const DATATYPE *data,
	size_t first,
	size_t count
)
{
	ifelse(DATATYPE, COMPLEX8,
	`DATATYPE sum = {0.0, 0.0};'
	, DATATYPE, COMPLEX16,
	`DATATYPE sum = {0.0, 0.0};'
	,
	DATATYPE sum = 0;)

	for(data += first; count-- > 0; data++) {
		ifelse(DATATYPE, COMPLEX8,
		sum.re += (*data).re;
		sum.im += (*data).im;
		, DATATYPE, COMPLEX16,
		sum.re += (*data).re;
		sum.im += (*data).im;
		, 
		sum += *data;)
	}

	return(sum);
}
#endif


SQUAREDATATYPE `XLAL'DATATYPE`SumSquares' (
	const DATATYPE *data,
	size_t first,
	size_t count
)
{
	SQUAREDATATYPE sum = 0;

	for(data += first; count-- > 0; data++) {
		ifelse(DATATYPE, COMPLEX8,
		sum += (*data).re * (*data).re + (*data).im * (*data).im;
		, DATATYPE, COMPLEX16,
		sum += (*data).re * (*data).re + (*data).im * (*data).im;
		, 
		sum += *data * *data;)
	}

	return(sum);
}


ifelse(DATATYPE, COMPLEX8, `#if 0', DATATYPE, COMPLEX16, `#if 0', `#if 1')
/* FIXME: too bad we aren't C99, or we could include versions for complex
 * data types */
DATATYPE `XLAL'SEQUENCETYPE`Sum' (
	const SEQUENCETYPE *sequence,
	size_t first,
	size_t count
)
{
	if(first >= sequence->length)
		return(0);
	if(first + count > sequence->length)
		count = sequence->length - first;
	return(`XLAL'DATATYPE`Sum' (sequence->data, first, count));
}
#endif


SQUAREDATATYPE `XLAL'SEQUENCETYPE`SumSquares' (
	const SEQUENCETYPE *sequence,
	size_t first,
	size_t count
)
{
	if(first >= sequence->length)
		return(0);
	if(first + count > sequence->length)
		count = sequence->length - first;
	return(`XLAL'DATATYPE`SumSquares' (sequence->data, first, count));
}


ifelse(DATATYPE, COMPLEX8,
SEQUENCETYPE *`XLALConjugate'SEQUENCETYPE (
	SEQUENCETYPE *sequence
)
{
	unsigned int i;

	for(i = 0; i < sequence->length; i++)
		sequence->data[i].im = -sequence->data[i].im;

	return(sequence);
}
, DATATYPE, COMPLEX16,
SEQUENCETYPE *`XLALConjugate'SEQUENCETYPE (
	SEQUENCETYPE *sequence
)
{
	unsigned int i;

	for(i = 0; i < sequence->length; i++)
		sequence->data[i].im = -sequence->data[i].im;

	return(sequence);
}
,)

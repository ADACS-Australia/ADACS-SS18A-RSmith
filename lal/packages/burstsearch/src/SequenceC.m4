dnl $Id$
define(SEQUENCETYPE,DATATYPE`Sequence')
void `XLALDestroy'SEQUENCETYPE (
	SEQUENCETYPE *sequence
)
{
	if(sequence)
		LALFree(sequence->data);
	LALFree(sequence);
}


void `LALDestroy'SEQUENCETYPE (
	LALStatus *status,
	SEQUENCETYPE *sequence
)
{
	INITSTATUS(status, "`LALDestroy'SEQUENCETYPE", SEQUENCEC);
	`XLALDestroy'SEQUENCETYPE (sequence);
	RETURN(status);
}


SEQUENCETYPE *`XLALCreate'SEQUENCETYPE (
	size_t length
)
{
	SEQUENCETYPE *new;
	DATATYPE *data;

	new = LALMalloc(sizeof(*new));
	data = LALMalloc(length * sizeof(*data));
	if(!new || !data) {
		LALFree(new);
		LALFree(data);
		return(NULL);
	}

	new->data = data;
	new->length = length;

	return(new);
}


void `LALCreate'SEQUENCETYPE (
	LALStatus *status,
	SEQUENCETYPE **output,
	size_t length
)
{
	INITSTATUS(status, "`LALCreate'SEQUENCETYPE", SEQUENCEC);
	ASSERT(output != NULL, status, LAL_NULL_ERR, LAL_NULL_MSG);
	*output = `XLALCreate'SEQUENCETYPE (length);
	ASSERT(*output != NULL, status, LAL_FAIL_ERR, LAL_FAIL_MSG);
	RETURN(status);
}


SEQUENCETYPE *`XLALCut'SEQUENCETYPE (
	SEQUENCETYPE *sequence,
	size_t first,
	size_t length
)
{
	SEQUENCETYPE *new;

	new = `XLALCreate'SEQUENCETYPE (length);
	if(!sequence || !sequence->data || !new) {
		`XLALDestroy'SEQUENCETYPE (new);
		return(NULL);
	}

	memcpy(new->data, sequence->data + first, length * sizeof(*new->data));

	return(new);
}


void `LALCut'SEQUENCETYPE (
	LALStatus *status,
	SEQUENCETYPE **output,
	SEQUENCETYPE *input,
	size_t first,
	size_t length
)
{
	INITSTATUS(status, "`LALCut'SEQUENCETYPE", SEQUENCEC);
	ASSERT(output != NULL, status, LAL_NULL_ERR, LAL_NULL_MSG);
	ASSERT(input != NULL, status, LAL_NULL_ERR, LAL_NULL_MSG);
	ASSERT(first + length <= input->length, status, LAL_RANGE_ERR, LAL_RANGE_MSG);
	*output = `XLALCut'SEQUENCETYPE (input, first, length);
	ASSERT(*output != NULL, status, LAL_FAIL_ERR, LAL_FAIL_MSG);
	RETURN(status);
}


void *`XLALShrink'SEQUENCETYPE (
	SEQUENCETYPE *sequence,
	size_t first,
	size_t length
)
{
	if(!sequence || !sequence->data)
		return(NULL);

	memmove(sequence->data, sequence->data + first, length * sizeof(*sequence->data));
	sequence->data = LALRealloc(sequence->data, length * sizeof(*sequence->data));
	sequence->length = length;

	return(sequence->data);
}


void `LALShrink'SEQUENCETYPE (
	LALStatus *status,
	SEQUENCETYPE *sequence,
	size_t first,
	size_t length
)
{
	INITSTATUS(status, "`LALShrink'SEQUENCETYPE", SEQUENCEC);
	ASSERT(sequence != NULL, status, LAL_NULL_ERR, LAL_NULL_MSG);
	ASSERT(first + length <= sequence->length, status, LAL_RANGE_ERR, LAL_RANGE_MSG);
	ASSERT(`XLALShrink'SEQUENCETYPE (sequence, first, length), status, LAL_FAIL_ERR, LAL_FAIL_MSG);
	RETURN(status);
}

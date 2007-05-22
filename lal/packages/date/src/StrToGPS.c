#include <ctype.h>
#include <errno.h>
#include <locale.h>
#include <stdlib.h>
#include <lal/Date.h>
#include <lal/LALDatatypes.h>

#include <lal/LALRCSID.h>
NRCSID (STRTOGPSC,"$Id$");


/*
 * Check for a base 10 or base 16 number.
 */

static int isbase10(const char *s, int radix)
{
	if(*s == radix)
		s++;
	if(isdigit(*s))
		return(1);
	return(0);
}

static int isbase16(const char *s, int radix)
{
	if(*s == '0') {
		s++;
		if(*s == 'X' || *s == 'x') {
			s++;
			if(*s == radix)
				s++;
			if(isxdigit(*s))
				return(1);
		}
	}
	return(0);
}


/*
 * Check that a string contains an exponent.
 */

static int isdecimalexp(const char *s)
{
	if(*s == 'E' || *s == 'e') {
		s++;
		if(*s == '+' || *s == '-')
			s++;
		if(isdigit(*s))
			return(1);
	}
	return(0);
}

static int isbinaryexp(const char *s)
{
	if(*s == 'P' || *s == 'p') {
		s++;
		if(*s == '+' || *s == '-')
			s++;
		if(isdigit(*s))
			return(1);
	}
	return(0);
}


/*
 * Parse an ASCII string into a LIGOTimeGPS structure.
 */

int XLALStrToGPS(LIGOTimeGPS *t, const char *nptr, char **endptr)
{
	const char *func = "XLALStrToGPS";
	union { char *s; const char *cs; } pconv; /* this is bad */
	int olderrno;
	int radix;
	char *digits;
	int len;
	int sign;
	int base;
	int radixpos;
	int exppart;

	/* save and clear C library errno so we can check for failures */
	olderrno = errno;
	errno = 0;

	/* retrieve the radix character */
	radix = localeconv()->decimal_point[0];

	/* this is bad ... there is a reason for warnings! */
	pconv.cs = nptr;

	/* consume leading white space */
	while(isspace(*(pconv.cs)))
		(pconv.cs)++;
	if(endptr)
		*endptr  = pconv.s;

	/* determine the sign */
	if(*(pconv.cs) == '-') {
		sign = -1;
		(pconv.cs)++;
	} else if(*(pconv.cs) == '+') {
		sign = +1;
		(pconv.cs)++;
	} else
		sign = +1;

	/* determine the base */
	if(isbase16((pconv.cs), radix)) {
		base = 16;
		(pconv.cs) += 2;
	} else if(isbase10((pconv.cs), radix)) {
		base = 10;
	} else {
		/* this isn't a recognized number */
		XLALGPSSet(t, 0, 0);
		return(0);
	}

	/* count the number of digits including the radix but not including
	 * the exponent. */
	radixpos = -1;
	switch(base) {
	case 10:
		for(len = 0; 1; len++) {
			if(isdigit((pconv.cs)[len]))
				continue;
			if((pconv.cs)[len] == radix && radixpos < 0) {
				radixpos = len;
				continue;
			}
			break;
		}
		break;
	
	case 16:
		for(len = 0; 1; len++) {
			if(isxdigit((pconv.cs)[len]))
				continue;
			if((pconv.cs)[len] == radix && radixpos < 0) {
				radixpos = len;
				continue;
			}
			break;
		}
		break;
	}

	/* copy the digits into a scratch space, removing the radix character
	 * if one was found */
	if(radixpos >= 0) {
		digits = malloc(len + 1);
		memcpy(digits, (pconv.cs), radixpos);
		memcpy(digits + radixpos, (pconv.cs) + radixpos + 1, len - radixpos - 1);
		digits[len - 1] = '\0';
		(pconv.cs) += len;
		len--;
	} else {
		digits = malloc(len + 2);
		memcpy(digits, (pconv.cs), len);
		digits[len] = '\0';
		radixpos = len;
		(pconv.cs) += len;
	}

	/* check for and parse an exponent, performing an adjustment of the
	 * radix position */
	exppart = 1;
	switch(base) {
	case 10:
		/* exponent is the number of powers of 10 */
		if(isdecimalexp((pconv.cs)))
			radixpos += strtol((pconv.cs) + 1, &pconv.s, 10);
		break;

	case 16:
		/* exponent is the number of powers of 2 */
		if(isbinaryexp((pconv.cs))) {
			exppart = strtol((pconv.cs) + 1, &pconv.s, 10);
			radixpos += exppart / 4;
			exppart %= 4;
			if(exppart < 0) {
				radixpos--;
				exppart += 4;
			}
			exppart = 1 << exppart;
		}
		break;
	}

	/* save end of converted characters */
	if(endptr)
		*endptr = pconv.s;

	/* insert the radix character, padding the scratch digits with zeroes
	 * if needed */
	if(radixpos < 2) {
		digits = realloc(digits, len + 2 + (2 - radixpos));
		memmove(digits + (2 - radixpos) + 1, digits, len + 1);
		memset(digits, '0', (2 - radixpos) + 1);
		if(radixpos == 1)
			digits[1] = digits[2];
		radixpos = 2;
	} else if(radixpos > len) {
		digits = realloc(digits, radixpos + 2);
		memset(digits + len, '0', radixpos - len);
		digits[radixpos + 1] = '\0';
	} else {
		memmove(digits + radixpos + 1, digits + radixpos, len - radixpos + 1);
	}
	digits[radixpos] = radix;

	/* parse the integer part */
	XLALINT8NSToGPS(t, sign * strtol(digits, NULL, base) * exppart * 1000000000ll);

	/* parse the fractional part */
	if(errno != ERANGE) {
		switch(base) {
		case 10:
			break;

		case 16:
			digits[radixpos - 2] = '0';
			digits[radixpos - 1] = 'x';
			radixpos -= 2;
			break;
		}
		XLALGPSAdd(t, sign * strtod(digits + radixpos, NULL) * exppart);
	}

	/* free the scratch space */
	free(digits);

	/* check for failures and restore errno if there weren't any */
	if(errno == ERANGE)
		XLAL_ERROR(func, XLAL_ERANGE);
	errno = olderrno;

	/* success */
	return(0);
}

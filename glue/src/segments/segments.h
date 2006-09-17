/*
 * $Id$
 *
 * Copyright (C) 2006  Kipp C. Cannon
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
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */

/*
 * ============================================================================
 *
 *                    Segments Module Component --- Header
 *
 * ============================================================================
 */

#ifndef __SEGMENTS_H__
#define __SEGMENTS_H__

#include <Python.h>

#define MODULE_NAME "glue.__segments"


/*
 * ============================================================================
 *
 *                               Infinity Class
 *
 * ============================================================================
 */


/*
 * Structure
 */

typedef struct {
	PyObject_HEAD
} segments_Infinity;


/*
 * Type
 */

extern PyTypeObject segments_Infinity_Type;


/*
 * Pre-allocated instances
 */

extern segments_Infinity *segments_PosInfinity;
extern segments_Infinity *segments_NegInfinity;


/*
 * ============================================================================
 *
 *                               Segment Class
 *
 * ============================================================================
 */


/*
 * Structure
 */

typedef PyTupleObject segments_Segment;


/*
 * Type
 */

extern PyTypeObject segments_Segment_Type;


#endif /* __SEGMENTS_H__ */

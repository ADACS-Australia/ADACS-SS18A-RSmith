/*
 * Copyright (C) 2015-2017  Leo Singer
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with with program; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston,
 * MA  02111-1307  USA
 *
 *
 *      Accelerated 1D and 2D cubic interpolation
 *      1. Constant boundary conditions
 *      2. Robust to invalid data: drops to linear or nearest-neighbor
 *         when the input data contains NaNs or infinities
 *      3. Bounds and invalid value checks are precomputed:
 *         minimal branch instructions needed in evaluate function
 *      5. Interpolating polynomial is precomputed:
 *         modest speedup relative to naive 1D interpolant,
 *         4x speedup over relative to naive 2D interpolant
 *         at the cost of 16x the memory footprint
 */


#ifndef CUBIC_INTERP_H
#define CUBIC_INTERP_H

/* exclude from SWIG interface and C++ code */
#if !defined(SWIG) && !defined(__cplusplus)

typedef struct {
    double f, t0, length;
    double a[][4];
} cubic_interp;

typedef struct {
    double fs, ft, s0, t0, slength, tlength;
    double a[][4][4];
} bicubic_interp;

cubic_interp *cubic_interp_init(
    const double *data, int n, double tmin, double dt);

void cubic_interp_free(cubic_interp *interp);

double cubic_interp_eval(const cubic_interp *interp, double t);

bicubic_interp *bicubic_interp_init(
    const double *data, int ns, int nt,
    double smin, double tmin, double ds, double dt);

void bicubic_interp_free(bicubic_interp *interp);

double bicubic_interp_eval(const bicubic_interp *interp, double s, double t);

#endif /* !defined(SWIG) && !defined(__cplusplus) */

#endif /* CUBIC_INTEPR_H */

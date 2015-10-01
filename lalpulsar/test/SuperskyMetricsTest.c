//
// Copyright (C) 2015 Karl Wette
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with with program; see the file COPYING. If not, write to the
// Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston,
// MA 02111-1307 USA
//

// Tests of the supersky metric code in SuperskyMetrics.[ch].

#include <math.h>
#include <gsl/gsl_blas.h>

#include <lal/SuperskyMetrics.h>
#include <lal/LALStdlib.h>
#include <lal/LALInitBarycenter.h>
#include <lal/MetricUtils.h>

#include "../src/GSLHelpers.h"

#define NUM_POINTS 10

#define REF_TIME { 900100100, 0 }
const LIGOTimeGPS ref_time = REF_TIME;

const PulsarDopplerParams phys_points[NUM_POINTS] = {
  { .refTime = REF_TIME, .Alpha = 0.00000000000000, .Delta =  0.000000000000000, .fkdot = {100.0000000000000,  0.00000000000000e-00} },
  { .refTime = REF_TIME, .Alpha = 4.88014010120016, .Delta = -0.954446475246007, .fkdot = { 99.9999983978492,  1.48957780038094e-09} },
  { .refTime = REF_TIME, .Alpha = 0.52587274931672, .Delta =  0.685297319257976, .fkdot = { 99.9999923150006, -1.41365319702693e-09} },
  { .refTime = REF_TIME, .Alpha = 3.53542175437611, .Delta = -1.502778038590950, .fkdot = {100.0000064863180, -1.28748375084384e-09} },
  { .refTime = REF_TIME, .Alpha = 1.36054903961191, .Delta =  0.241343663657163, .fkdot = { 99.9999901679571,  3.37107171004537e-10} },
  { .refTime = REF_TIME, .Alpha = 2.85470536965808, .Delta = 1.1575340928032900, .fkdot = {100.0000074463050,  2.46412240438217e-09} },
  { .refTime = REF_TIME, .Alpha = 1.82755817952460, .Delta =  0.667995269285982, .fkdot = { 99.9999897239871,  1.79900370692270e-10} },
  { .refTime = REF_TIME, .Alpha = 1.70734223243163, .Delta = -1.213787405673430, .fkdot = {100.0000026535270, -1.07122135891104e-09} },
  { .refTime = REF_TIME, .Alpha = 2.30597131157246, .Delta =  0.348657791621429, .fkdot = {100.0000133749770, -5.43309003215614e-10} },
  { .refTime = REF_TIME, .Alpha = 3.31129323970275, .Delta = -1.225892709583030, .fkdot = {100.0000062524320,  8.07713885739405e-10} }
};

#define NUM_SEGS 3

const double Tspan = 3 * 86400;
const double deltat[NUM_SEGS] = { -8 * 86400, 0, 8 * 86400 };

const double semi_ussky_metric_ref[5][5] = {
  { 1.875399697216015e+07,  6.562888395133028e+06,  2.848318664412220e+06,  2.034003043324815e+09, -5.025484950945562e+13},
  { 6.562888395133028e+06,  2.561771110376429e+06,  1.111728527754202e+06,  7.223614506915147e+08,  1.191189674660967e+14},
  { 2.848318664412220e+06,  1.111728527754202e+06,  4.824674307199425e+05,  3.135050561021789e+08,  5.166617541197261e+13},
  { 2.034003043324815e+09,  7.223614506915147e+08,  3.135050561021789e+08,  2.210286062098707e+11, -8.077428080473405e+02},
  {-5.025484950945562e+13,  1.191189674660967e+14,  5.166617541197261e+13, -8.077428080473405e+02,  7.064620283536166e+22},
};

const double coh_ussky_metric_refs[NUM_SEGS][5][5] = {
  {
    { 2.064714297922182e+07,  4.469413173721094e+06,  1.940468414778165e+06,  2.136255051814566e+09, -1.476682314938338e+15},
    { 4.469413173721094e+06,  9.683579217483700e+05,  4.204085621764289e+05,  4.624483663936913e+08, -3.192182021511352e+14},
    { 1.940468414778165e+06,  4.204085621764289e+05,  1.825307650051137e+05,  2.007795958320413e+08, -1.385893436710151e+14},
    { 2.136255051814566e+09,  4.624483663936913e+08,  2.007795958320413e+08,  2.210286062098726e+11, -1.527749726122714e+17},
    {-1.476682314938338e+15, -3.192182021511352e+14, -1.385893436710151e+14, -1.527749726122714e+17,  1.058455565252480e+23},
  }, {
    { 1.895617224814336e+07,  6.731809404038861e+06,  2.921605097922917e+06,  2.046904385910545e+09, -1.656990342031264e+11},
    { 6.731809404038861e+06,  2.391591549452773e+06,  1.037930347967158e+06,  7.269432686412601e+08,  4.092485067498952e+11},
    { 2.921605097922917e+06,  1.037930347967158e+06,  4.504649325079115e+05,  3.154935665162594e+08,  1.815805951797570e+11},
    { 2.046904385910545e+09,  7.269432686412601e+08,  3.154935665162594e+08,  2.210286062098680e+11, -5.119042942971644e-01},
    {-1.656990342031264e+11,  4.092485067498952e+11,  1.815805951797570e+11, -5.119042942971644e-01,  2.474954556318625e+20},
  }, {
    { 1.665867568911552e+07,  8.487442607639275e+06,  3.682882480535639e+06,  1.918849692249366e+09,  1.326083465443978e+15},
    { 8.487442607639275e+06,  4.325363859941072e+06,  1.876846673124623e+06,  9.776927170396193e+08,  6.761658560425941e+14},
    { 3.682882480535639e+06,  1.876846673124623e+06,  8.144065946492312e+05,  4.242420059582472e+08,  2.934062893117179e+14},
    { 1.918849692249366e+09,  9.776927170396193e+08,  4.242420059582472e+08,  2.210286062098711e+11,  1.527749726122546e+17},
    { 1.326083465443978e+15,  6.761658560425941e+14,  2.934062893117179e+14,  1.527749726122546e+17,  1.058455565252238e+23},
  }
};

const double semi_rssky_metric_ref[4][4] = {
  { 5.069230179517075e+02,  0.000000000000000e+00,  0.000000000000000e+00,  0.000000000000000e+00},
  { 0.000000000000000e+00,  6.547662357441416e+01,  0.000000000000000e+00,  0.000000000000000e+00},
  { 0.000000000000000e+00,  0.000000000000000e+00,  7.064620283536166e+22, -8.077428080473405e+02},
  { 0.000000000000000e+00,  0.000000000000000e+00, -8.077428080473405e+02,  2.210286062098707e+11},
};

const double coh_rssky_metric_refs[NUM_SEGS][4][4] = {
  {
    { 6.568636182144087e+01,  0.000000000000000e+00,  0.000000000000000e+00,  0.000000000000000e+00},
    { 0.000000000000000e+00,  6.237312016400978e+01,  0.000000000000000e+00,  0.000000000000000e+00},
    { 0.000000000000000e+00,  0.000000000000000e+00,  1.058455565252480e+23, -1.527749726122714e+17},
    { 0.000000000000000e+00,  0.000000000000000e+00, -1.527749726122714e+17,  2.210286062098726e+11},
  }, {
    { 6.568666765075774e+01,  0.000000000000000e+00,  0.000000000000000e+00,  0.000000000000000e+00},
    { 0.000000000000000e+00,  6.236546718525190e+01,  0.000000000000000e+00,  0.000000000000000e+00},
    { 0.000000000000000e+00,  0.000000000000000e+00,  2.474954556329869e+20, -1.208418861582654e+02},
    { 0.000000000000000e+00,  0.000000000000000e+00, -1.208418861582654e+02,  2.210286062098680e+11},
  }, {
    { 6.568680103016807e+01,  0.000000000000000e+00,  0.000000000000000e+00,  0.000000000000000e+00},
    { 0.000000000000000e+00,  6.235717686528133e+01,  0.000000000000000e+00,  0.000000000000000e+00},
    { 0.000000000000000e+00,  0.000000000000000e+00,  1.058455565252238e+23,  1.527749726122546e+17},
    { 0.000000000000000e+00,  0.000000000000000e+00,  1.527749726122546e+17,  2.210286062098711e+11},
  }
};

const double semi_rssky_transf_ref[5][3] = {
  { 9.363383835591375e-01,  3.276518345636219e-01,  1.261535048302427e-01},
  {-3.309081001489509e-01,  9.436487575620593e-01,  5.181853663774516e-03},
  {-1.173467542357819e-01, -4.659718509388588e-02,  9.919971983890140e-01},
  {-2.134760488419512e-11,  1.830302603631687e-09,  7.303907936812212e-10},
  { 9.866360627623324e-03,  4.620189300680953e-05,  1.748754269165573e-04},
};

const double coh_rssky_transf_refs[NUM_SEGS][5][3] = {
  {
    { 7.014188729589919e-01, -7.127492992955380e-01,  3.179083552258733e-05},
    { 7.127491812384910e-01,  7.014187811320953e-01,  5.460027715689418e-04},
    {-4.114617819526091e-04, -3.603177566767635e-04,  9.999998504351469e-01},
    {-1.518511649108543e-09,  9.119239045225907e-10,  7.652734977238319e-10},
    { 4.238436305149103e-03,  8.987129848963495e-03,  1.432613602463570e-03},
  }, {
    { 7.928332170641094e-01, -6.094386653165459e-01,  5.600859050599027e-05},
    { 6.094384042541772e-01,  7.928329562280915e-01,  8.572856860540078e-04},
    {-5.668685006887667e-04, -6.455507823947125e-04,  9.999996309620771e-01},
    {-1.538506600219701e-09,  9.036045301253465e-10,  7.329842343015285e-10},
    { 5.337970249571395e-03,  8.252674722491856e-03,  1.420014590206618e-03},
  }, {
    { 8.689897105711624e-01, -4.948301493223842e-01,  7.901287635748977e-05},
    { 4.948297433622081e-01,  8.689891816122729e-01,  1.152095883124570e-03},
    {-6.387531126429617e-04, -9.620615466963718e-04,  9.999993332157983e-01},
    {-1.560552363459718e-09,  9.006124427330534e-10,  6.868353365772801e-10},
    { 6.434079710711734e-03,  7.519415923170119e-03,  1.434855908485180e-03},
  }
};

const double semi_phys_mismatch[NUM_POINTS][NUM_POINTS] = {
  {0.000000e+00, 2.842696e+07, 1.714926e+05, 2.821377e+07, 3.877800e+06, 2.839587e+07, 1.360228e+07, 2.139293e+07, 3.333989e+07, 4.206510e+07},
  {2.842696e+07, 0.000000e+00, 2.419283e+07, 2.504598e+05, 1.251935e+07, 1.002015e+06, 3.282244e+06, 5.689312e+05, 3.277731e+05, 1.336220e+06},
  {1.714926e+05, 2.419283e+07, 0.000000e+00, 2.398770e+07, 2.552860e+06, 2.429737e+07, 1.080643e+07, 1.773417e+07, 2.877145e+07, 3.688169e+07},
  {2.821377e+07, 2.504598e+05, 2.398770e+07, 0.000000e+00, 1.312504e+07, 2.237378e+06, 4.014595e+06, 5.120383e+05, 9.922287e+05, 1.759271e+06},
  {3.877800e+06, 1.251935e+07, 2.552860e+06, 1.312504e+07, 0.000000e+00, 1.153347e+07, 3.162766e+06, 8.455535e+06, 1.533802e+07, 2.178058e+07},
  {2.839587e+07, 1.002015e+06, 2.429737e+07, 2.237378e+06, 1.153347e+07, 0.000000e+00, 2.696814e+06, 1.922870e+06, 6.601354e+05, 2.417185e+06},
  {1.360228e+07, 3.282244e+06, 1.080643e+07, 4.014595e+06, 3.162766e+06, 2.696814e+06, 0.000000e+00, 1.740316e+06, 4.592892e+06, 8.444680e+06},
  {2.139293e+07, 5.689312e+05, 1.773417e+07, 5.120383e+05, 8.455535e+06, 1.922870e+06, 1.740316e+06, 0.000000e+00, 1.695582e+06, 3.585868e+06},
  {3.333989e+07, 3.277731e+05, 2.877145e+07, 9.922287e+05, 1.533802e+07, 6.601354e+05, 4.592892e+06, 1.695582e+06, 0.000000e+00, 6.169347e+05},
  {4.206510e+07, 1.336220e+06, 3.688169e+07, 1.759271e+06, 2.178058e+07, 2.417185e+06, 8.444680e+06, 3.585868e+06, 6.169347e+05, 0.000000e+00},
};

const double coh_phys_mismatches[NUM_SEGS][NUM_POINTS][NUM_POINTS] = {
  {
    {0.000000e+00, 3.024189e+07, 1.543644e+05, 2.361445e+07, 7.297336e+06, 4.341212e+07, 2.026653e+07, 1.998370e+07, 4.088115e+07, 4.591125e+07},
    {3.024189e+07, 0.000000e+00, 2.607532e+07, 4.103494e+05, 7.831289e+06, 1.189821e+06, 9.965225e+05, 1.059082e+06, 8.007433e+05, 1.629541e+06},
    {1.543644e+05, 2.607532e+07, 0.000000e+00, 1.995049e+07, 5.329463e+06, 3.838988e+07, 1.688386e+07, 1.662551e+07, 3.601172e+07, 4.074165e+07},
    {2.361445e+07, 4.103494e+05, 1.995049e+07, 0.000000e+00, 4.662728e+06, 2.997510e+06, 1.322075e+05, 1.516108e+05, 2.356988e+06, 3.673699e+06},
    {7.297336e+06, 7.831289e+06, 5.329463e+06, 4.662728e+06, 0.000000e+00, 1.511270e+07, 3.242171e+06, 3.132814e+06, 1.363639e+07, 1.660431e+07},
    {4.341212e+07, 1.189821e+06, 3.838988e+07, 2.997510e+06, 1.511270e+07, 0.000000e+00, 4.355434e+06, 4.492228e+06, 3.929618e+04, 3.792766e+04},
    {2.026653e+07, 9.965225e+05, 1.688386e+07, 1.322075e+05, 3.242171e+06, 4.355434e+06, 0.000000e+00, 3.558143e+03, 3.580300e+06, 5.172549e+06},
    {1.998370e+07, 1.059082e+06, 1.662551e+07, 1.516108e+05, 3.132814e+06, 4.492228e+06, 3.558143e+03, 0.000000e+00, 3.701188e+06, 5.315678e+06},
    {4.088115e+07, 8.007433e+05, 3.601172e+07, 2.356988e+06, 1.363639e+07, 3.929618e+04, 3.580300e+06, 3.701188e+06, 0.000000e+00, 1.462400e+05},
    {4.591125e+07, 1.629541e+06, 4.074165e+07, 3.673699e+06, 1.660431e+07, 3.792766e+04, 5.172549e+06, 5.315678e+06, 1.462400e+05, 0.000000e+00},
  }, {
    {0.000000e+00, 2.876453e+07, 1.736967e+05, 2.845102e+07, 3.414811e+06, 2.743194e+07, 1.322922e+07, 2.165178e+07, 3.346665e+07, 4.253244e+07},
    {2.876453e+07, 0.000000e+00, 2.447187e+07, 1.779353e+03, 1.236248e+07, 1.946008e+04, 2.981507e+06, 5.045797e+05, 1.784609e+05, 1.342001e+06},
    {1.736967e+05, 2.447187e+07, 0.000000e+00, 2.418261e+07, 2.049651e+06, 2.324404e+07, 1.037386e+07, 1.795027e+07, 2.882252e+07, 3.727495e+07},
    {2.845102e+07, 1.779353e+03, 2.418261e+07, 0.000000e+00, 1.215974e+07, 1.722632e+04, 2.883870e+06, 4.636396e+05, 2.061967e+05, 1.412060e+06},
    {3.414811e+06, 1.236248e+07, 2.049651e+06, 1.215974e+07, 0.000000e+00, 1.149079e+07, 3.202415e+06, 7.874615e+06, 1.550436e+07, 2.184963e+07},
    {2.743194e+07, 1.946008e+04, 2.324404e+07, 1.722632e+04, 1.149079e+07, 0.000000e+00, 2.561189e+06, 3.466390e+05, 3.014610e+05, 1.652932e+06},
    {1.322922e+07, 2.981507e+06, 1.037386e+07, 2.883870e+06, 3.202415e+06, 2.561189e+06, 0.000000e+00, 1.035231e+06, 4.614106e+06, 8.322602e+06},
    {2.165178e+07, 5.045797e+05, 1.795027e+07, 4.636396e+05, 7.874615e+06, 3.466390e+05, 1.035231e+06, 0.000000e+00, 1.282484e+06, 3.491914e+06},
    {3.346665e+07, 1.784609e+05, 2.882252e+07, 2.061967e+05, 1.550436e+07, 3.014610e+05, 4.614106e+06, 1.282484e+06, 0.000000e+00, 5.430866e+05},
    {4.253244e+07, 1.342001e+06, 3.727495e+07, 1.412060e+06, 2.184963e+07, 1.652932e+06, 8.322602e+06, 3.491914e+06, 5.430866e+05, 0.000000e+00},
  }, {
    {0.000000e+00, 2.627447e+07, 1.867214e+05, 3.257584e+07, 9.213898e+05, 1.434368e+07, 7.311169e+06, 2.254331e+07, 2.567187e+07, 3.775160e+07},
    {2.627447e+07, 0.000000e+00, 2.203155e+07, 3.392506e+05, 1.736439e+07, 1.796882e+06, 5.868773e+06, 1.431319e+05, 4.129221e+03, 1.037116e+06},
    {1.867214e+05, 2.203155e+07, 0.000000e+00, 2.783016e+07, 2.794658e+05, 1.125817e+07, 5.161574e+06, 1.862685e+07, 2.148010e+07, 3.262861e+07},
    {3.257584e+07, 3.392506e+05, 2.783016e+07, 0.000000e+00, 2.255274e+07, 3.697478e+06, 9.027756e+06, 9.208643e+05, 4.135111e+05, 1.920540e+05},
    {9.213898e+05, 1.736439e+07, 2.794658e+05, 2.255274e+07, 0.000000e+00, 7.996908e+06, 3.043713e+06, 1.435923e+07, 1.687331e+07, 2.688786e+07},
    {1.434368e+07, 1.796882e+06, 1.125817e+07, 3.697478e+06, 7.996908e+06, 0.000000e+00, 1.173820e+06, 9.298057e+05, 1.639649e+06, 5.560758e+06},
    {7.311169e+06, 5.868773e+06, 5.161574e+06, 9.027756e+06, 3.043713e+06, 1.173820e+06, 0.000000e+00, 4.182195e+06, 5.584268e+06, 1.183893e+07},
    {2.254331e+07, 1.431319e+05, 1.862685e+07, 9.208643e+05, 1.435923e+07, 9.298057e+05, 4.182195e+06, 0.000000e+00, 1.030798e+05, 1.950011e+06},
    {2.567187e+07, 4.129221e+03, 2.148010e+07, 4.135111e+05, 1.687331e+07, 1.639649e+06, 5.584268e+06, 1.030798e+05, 0.000000e+00, 1.161485e+06},
    {3.775160e+07, 1.037116e+06, 3.262861e+07, 1.920540e+05, 2.688786e+07, 5.560758e+06, 1.183893e+07, 1.950011e+06, 1.161485e+06, 0.000000e+00},
  }
};

#define CHECK_RELERR(A, B, TOL) do { \
    const double lhs = fabs( (A) - (B) ); \
    const double tol = (TOL); \
    const double rhs = GSL_MAX( 1.0, fabs( (A) + (B) ) ); \
    XLALPrintInfo( #A"=%0.5e   "#B"=%0.5e   |"#A" - "#B"|=%0.5e   tol=%0.5e   |"#A" + "#B"|=%0.5e\n", A, B, lhs, tol, rhs ); \
    XLAL_CHECK( lhs <= tol * rhs, XLAL_ETOL, "|"#A" - "#B"| = %0.5e > %0.5e = %0.5e * |"#A" + "#B"|", lhs, tol * rhs, tol ); \
  } while(0)

static int CompareDoppler(const PulsarDopplerParams *a, const PulsarDopplerParams *b)
{
  XLAL_CHECK(XLALGPSCmp(&a->refTime, &b->refTime) == 0, XLAL_ETOL, "Reference time mismatch!");
  CHECK_RELERR(cos(a->Alpha), cos(b->Alpha), 1e-10);
  CHECK_RELERR(sin(a->Alpha), sin(b->Alpha), 1e-10);
  CHECK_RELERR(a->Delta, b->Delta, 1e-10);
  CHECK_RELERR(a->fkdot[0], b->fkdot[0], 1e-10);
  CHECK_RELERR(a->fkdot[1], b->fkdot[1], 1e-10);
  return XLAL_SUCCESS;
}

static int CheckSuperskyMetrics(
  const gsl_matrix *ussky_metric,
  const double ussky_metric_ref[5][5],
  const gsl_matrix *rssky_metric,
  const double rssky_metric_ref[4][4],
  const gsl_matrix *rssky_transf,
  const double rssky_transf_ref[5][3],
  const double phys_mismatch[NUM_POINTS][NUM_POINTS],
  const double ussky_phys_mismatch_tol,
  const double rssky_phys_mismatch_tol
  )
{

  // Check supersky metrics
  {
    gsl_matrix_const_view ussky_metric_ref_view = gsl_matrix_const_view_array((const double *)ussky_metric_ref, 5, 5);
    const double err = XLALCompareMetrics(ussky_metric, &ussky_metric_ref_view.matrix), err_tol = 1e-7;
    XLAL_CHECK(err <= err_tol, XLAL_ETOL, "'ussky_metric' check failed: err = %0.3e > %0.3e = err_tol", err, err_tol);
  }
  {
    gsl_matrix_const_view rssky_metric_ref_view = gsl_matrix_const_view_array((const double *)rssky_metric_ref, 4, 4);
    const double err = XLALCompareMetrics(rssky_metric, &rssky_metric_ref_view.matrix), err_tol = 1e-7;
    XLAL_CHECK(err <= err_tol, XLAL_ETOL, "'rssky_metric' check failed: err = %0.3e > %0.3e = err_tol", err, err_tol);
  }
  {
    double max_err = 0;
    for (size_t i = 0; i < 5; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        const double rssky_transf_ij = gsl_matrix_get(rssky_transf, i, j);
        const double rssky_transf_ref_ij = rssky_transf_ref[i][j];
        const double err_ij = fabs((rssky_transf_ij - rssky_transf_ref_ij) / rssky_transf_ref_ij);
        if (err_ij > max_err) {
          max_err = err_ij;
        }
      }
    }
    const double err_tol = 2e-4;
    XLAL_CHECK(max_err <= err_tol, XLAL_ETOL, "'rssky_transf' check failed: max(err) = %0.3e > %0.3e = err_tol", max_err, err_tol);
  }

  // Check round-trip conversions of each test point
  {
    gsl_vector *GAVEC(ussky_point, 5);
    for (size_t i = 0; i < NUM_POINTS; ++i) {
      PulsarDopplerParams XLAL_INIT_DECL(point);
      XLAL_CHECK(XLALConvertPhysicalToSupersky(SC_USSKY, ussky_point, &phys_points[i], NULL, &ref_time) == XLAL_SUCCESS, XLAL_EFUNC);
      XLAL_CHECK(XLALConvertSuperskyToPhysical(&point, SC_USSKY, ussky_point, NULL, &ref_time) == XLAL_SUCCESS, XLAL_EFUNC);
      XLAL_CHECK(CompareDoppler(&phys_points[i], &point) == EXIT_SUCCESS, XLAL_EFUNC);
    }
    GFVEC(ussky_point);
  }
  {
    gsl_vector *GAVEC(rssky_point, 4);
    for (size_t i = 0; i < NUM_POINTS; ++i) {
      PulsarDopplerParams XLAL_INIT_DECL(point);
      XLAL_CHECK(XLALConvertPhysicalToSupersky(SC_RSSKY, rssky_point, &phys_points[i], rssky_transf, &ref_time) == XLAL_SUCCESS, XLAL_EFUNC);
      XLAL_CHECK(XLALConvertSuperskyToPhysical(&point, SC_RSSKY, rssky_point, rssky_transf, &ref_time) == XLAL_SUCCESS, XLAL_EFUNC);
      XLAL_CHECK(CompareDoppler(&phys_points[i], &point) == EXIT_SUCCESS, XLAL_EFUNC);
    }
    GFVEC(rssky_point);
  }

  // Check mismatches between pairs of points
  {
    gsl_vector *GAVEC(ussky_point_i, 5);
    gsl_vector *GAVEC(ussky_point_j, 5);
    gsl_vector *GAVEC(temp, 5);
    for (size_t i = 0; i < NUM_POINTS; ++i) {
      XLAL_CHECK(XLALConvertPhysicalToSupersky(SC_USSKY, ussky_point_i, &phys_points[i], NULL, &ref_time) == XLAL_SUCCESS, XLAL_EFUNC);
      for (size_t j = 0; j < NUM_POINTS; ++j) {
        XLAL_CHECK(XLALConvertPhysicalToSupersky(SC_USSKY, ussky_point_j, &phys_points[j], NULL, &ref_time) == XLAL_SUCCESS, XLAL_EFUNC);
        gsl_vector_sub(ussky_point_j, ussky_point_i);
        gsl_blas_dgemv(CblasNoTrans, 1.0, ussky_metric, ussky_point_j, 0.0, temp);
        double mismatch = 0.0;
        gsl_blas_ddot(ussky_point_j, temp, &mismatch);
        CHECK_RELERR(mismatch, phys_mismatch[i][j], ussky_phys_mismatch_tol);
      }
    }
    GFVEC(ussky_point_i, ussky_point_j, temp);
  }
  {
    gsl_vector *GAVEC(rssky_point_i, 4);
    gsl_vector *GAVEC(rssky_point_j, 4);
    gsl_vector *GAVEC(temp, 4);
    for (size_t i = 0; i < NUM_POINTS; ++i) {
      XLAL_CHECK(XLALConvertPhysicalToSupersky(SC_RSSKY, rssky_point_i, &phys_points[i], rssky_transf, &ref_time) == XLAL_SUCCESS, XLAL_EFUNC);
      for (size_t j = 0; j < NUM_POINTS; ++j) {
        XLAL_CHECK(XLALConvertPhysicalToSupersky(SC_RSSKY, rssky_point_j, &phys_points[j], rssky_transf, &ref_time) == XLAL_SUCCESS, XLAL_EFUNC);
        gsl_vector_sub(rssky_point_j, rssky_point_i);
        gsl_blas_dgemv(CblasNoTrans, 1.0, rssky_metric, rssky_point_j, 0.0, temp);
        double mismatch = 0.0;
        gsl_blas_ddot(rssky_point_j, temp, &mismatch);
        CHECK_RELERR(mismatch, phys_mismatch[i][j], rssky_phys_mismatch_tol);
      }
    }
    GFVEC(rssky_point_i, rssky_point_j, temp);
  }

  return XLAL_SUCCESS;

}

int main(void)
{

  // Load ephemeris data
  EphemerisData *edat =  XLALInitBarycenter(TEST_DATA_DIR "earth00-19-DE405.dat.gz",
                                            TEST_DATA_DIR "sun00-19-DE405.dat.gz");
  XLAL_CHECK_MAIN(edat != NULL, XLAL_EFUNC);

  // Create segment list
  LALSegList segments;
  XLAL_CHECK_MAIN(XLALSegListInit(&segments) == XLAL_SUCCESS, XLAL_EFUNC);
  for (size_t n = 0; n < NUM_SEGS; ++n) {
    LALSeg segment;
    LIGOTimeGPS start_time = ref_time, end_time = ref_time;
    XLALGPSAdd(&start_time, deltat[n] - 0.5 * Tspan);
    XLALGPSAdd(&end_time, deltat[n] + 0.5 * Tspan);
    XLAL_CHECK_MAIN(XLALSegSet(&segment, &start_time, &end_time, 0) == XLAL_SUCCESS, XLAL_EFUNC);
    XLAL_CHECK_MAIN(XLALSegListAppend(&segments, &segment) == XLAL_SUCCESS, XLAL_EFUNC);
  }

  // Compute supersky metrics
  const MultiLALDetector detectors = { .length = 1, .sites = { lalCachedDetectors[LAL_LLO_4K_DETECTOR] } };
  SuperskyMetrics *metrics = XLALComputeSuperskyMetrics(1, &ref_time, &segments, 100.0, &detectors, NULL, DETMOTION_SPIN | DETMOTION_PTOLEORBIT, edat);
  XLAL_CHECK_MAIN(metrics != NULL, XLAL_EFUNC);

  // Check coherent metrics
  for (size_t n = 0; n < NUM_SEGS; ++n) {
    XLAL_CHECK_MAIN(CheckSuperskyMetrics(
                      metrics->ussky_metric_seg[n], coh_ussky_metric_refs[n],
                      metrics->rssky_metric_seg[n], coh_rssky_metric_refs[n],
                      metrics->rssky_transf_seg[n], coh_rssky_transf_refs[n],
                      coh_phys_mismatches[n], 1e-2, 1e-2
                      ) == XLAL_SUCCESS, XLAL_EFUNC);
  }

  // Check semicoherent metric
  XLAL_CHECK_MAIN(CheckSuperskyMetrics(
                    metrics->ussky_metric_avg, semi_ussky_metric_ref,
                    metrics->rssky_metric_avg, semi_rssky_metric_ref,
                    metrics->rssky_transf_avg, semi_rssky_transf_ref,
                    semi_phys_mismatch, 1e-2, 3e-2
                    ) == XLAL_SUCCESS, XLAL_EFUNC);

  // Cleanup
  XLALDestroyEphemerisData(edat);
  XLALSegListClear(&segments);
  XLALDestroySuperskyMetrics(metrics);
  LALCheckMemoryLeaks();

  return EXIT_SUCCESS;

}

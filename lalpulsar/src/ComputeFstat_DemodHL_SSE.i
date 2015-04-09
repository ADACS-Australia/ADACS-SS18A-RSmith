//
// Copyright (C) 2007--2010, 2012 Bernd Machenschalk, Reinhard Prix, Fekete Akos
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with with program; see the file COPYING. If not, write to the
// Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston,
// MA  02111-1307  USA
//

/** SSE version with precalculated divisors from Akos */
{
  {
    REAL4 s_alpha, c_alpha;   /* sin(2pi kappa_alpha) and (cos(2pi kappa_alpha)-1) */
    //COMPLEX8 XSums __attribute__ ((aligned (16))); /* sums of Xa.re and Xa.im for SSE */
    REAL4 kappa_s = kappa_star; /* single precision version of kappa_star */

    static REAL4 *scd = &(sincosLUTdiff[0]);
    static REAL4 *scb = &(sincosLUTbase[0]);
    static REAL4 M1 = -1.0f;
    static REAL8 sincos_adds = 402653184.0;
    REAL8 tmp; /* this could as well be an int64, it's just a 64Bit placeholder */
    REAL8 _lambda_alpha = lambda_alpha;
    REAL8 kstar8 = kappa_star; /* kstar8 should be optimized away if kappa_star is already REAL8 */
    /* vector constants */
    /* having these not aligned will crash the assembler code */
  #define ALIGNED_VECTOR(name) static REAL4 name[4] __attribute__ ((aligned (16)))
    ALIGNED_VECTOR(D2222) = {2.0f, 2.0f, 2.0f, 2.0f};
    ALIGNED_VECTOR(D1100) = {1.0f, 1.0f, 0.0f, 0.0f};
    ALIGNED_VECTOR(D3322) = {3.0f, 3.0f, 2.0f, 2.0f};
    ALIGNED_VECTOR(D5544) = {5.0f, 5.0f, 4.0f, 4.0f};
    ALIGNED_VECTOR(D7766) = {7.0f, 7.0f, 6.0f, 6.0f};
    ALIGNED_VECTOR(Daabb) = {-1.0f, -1.0f, -2.0f, -2.0f};
    ALIGNED_VECTOR(Dccdd) = {-3.0f, -3.0f, -4.0f, -4.0f};
    ALIGNED_VECTOR(Deeff) = {-5.0f, -5.0f, -6.0f, -6.0f};
    ALIGNED_VECTOR(Dgghh) = {-7.0f, -7.0f, -8.0f, -8.0f};

    /* hand-coded SSE version from Akos */

    __asm __volatile
      (
       "movaps %[D7766],%%xmm0    \n\t"
       "movaps %[D5544],%%xmm1  \n\t"
       "movups (%[Xa]),%%xmm2   \n\t"
       "movups 0x10(%[Xa]),%%xmm3 \n\t"
       "movss  %[kappa_s],%%xmm7  \n\t"
       "shufps $0x0,%%xmm7,%%xmm7 \n\t"
       SINCOS_P0(kappa_star)
       "addps  %%xmm7,%%xmm0      \n\t"
       "addps  %%xmm7,%%xmm1      \n\t"
       "rcpps  %%xmm0,%%xmm0      \n\t"
       "rcpps  %%xmm1,%%xmm1      \n\t"
       "mulps  %%xmm2,%%xmm0      \n\t"
       "mulps  %%xmm3,%%xmm1      \n\t"
       SINCOS_P1
       "addps  %%xmm1,%%xmm0      \n\t"
       "movaps %[D3322],%%xmm2    \n\t"
       "movaps %[Dccdd],%%xmm3    \n\t"
       "movups 0x20(%[Xa]),%%xmm4 \n\t"
       "movups 0x50(%[Xa]),%%xmm5 \n\t"
       SINCOS_P2
       "addps  %%xmm7,%%xmm2      \n\t"
       "addps  %%xmm7,%%xmm3      \n\t"
       "rcpps  %%xmm2,%%xmm2      \n\t"
       "rcpps  %%xmm3,%%xmm3      \n\t"
       "mulps  %%xmm4,%%xmm2      \n\t"
       "mulps  %%xmm5,%%xmm3      \n\t"
       SINCOS_P3
       "addps  %%xmm3,%%xmm2      \n\t"
       "movaps %[Deeff],%%xmm4    \n\t"
       "movaps %[Dgghh],%%xmm5    \n\t"
       "movups 0x60(%[Xa]),%%xmm1 \n\t"
       "movups 0x70(%[Xa]),%%xmm6 \n\t"
       SINCOS_P4
       "addps  %%xmm7,%%xmm4      \n\t"
       "addps  %%xmm7,%%xmm5      \n\t"
       "rcpps  %%xmm4,%%xmm4      \n\t"
       "rcpps  %%xmm5,%%xmm5      \n\t"
       "mulps  %%xmm1,%%xmm4      \n\t"
       "mulps  %%xmm6,%%xmm5      \n\t"
       SINCOS_P5(sin)
       "addps  %%xmm2,%%xmm0      \n\t"
       "addps  %%xmm5,%%xmm4      \n\t"
       "movaps %[D1100],%%xmm1    \n\t"
       "movaps %[Daabb],%%xmm2    \n\t"
       SINCOS_P6(cos)
       "addps  %%xmm7,%%xmm1      \n\t"
       "addps  %%xmm7,%%xmm2      \n\t"
       "rcpps  %%xmm1,%%xmm5      \n\t"
       "rcpps  %%xmm2,%%xmm6      \n\t"
       SINCOS_TRIM_P0A(_lambda_alpha)
       "addps  %%xmm4,%%xmm0      \n\t"
       "movaps %[D2222],%%xmm3    \n\t"
       "movaps %[D2222],%%xmm4    \n\t"
       "mulps  %%xmm5,%%xmm1      \n\t"
       "mulps  %%xmm6,%%xmm2      \n\t"
       SINCOS_TRIM_P0B(_lambda_alpha)
       "subps  %%xmm1,%%xmm3      \n\t"
       "subps  %%xmm2,%%xmm4      \n\t"
       "mulps  %%xmm3,%%xmm5      \n\t"
       "mulps  %%xmm4,%%xmm6      \n\t"
       "movups 0x30(%[Xa]),%%xmm1 \n\t"
       "movups 0x40(%[Xa]),%%xmm2 \n\t"
       SINCOS_P1
       "mulps  %%xmm5,%%xmm1      \n\t"
       "mulps  %%xmm6,%%xmm2      \n\t"
       "addps  %%xmm1,%%xmm0      \n\t"
       "addps  %%xmm2,%%xmm0      \n\t"
       SINCOS_P2
       "movhlps %%xmm0,%%xmm1     \n\t"
       "addps  %%xmm1,%%xmm0      \n\t"

  /*
  c_alpha-=1.0f;
    realXP = s_alpha * XSums.re - c_alpha * XSums.im;
    imagXP = c_alpha * XSums.re + s_alpha * XSums.im;
  */

       "movss %[M1],%%xmm5        \n\t"
       "movaps %%xmm0,%%xmm3      \n\t"
       "shufps $1,%%xmm3,%%xmm3   \n\t"
       SINCOS_P3
       "movss %[cos],%%xmm2       \n\t"
       "movss %[sin],%%xmm1       \n\t"
       "addss %%xmm5,%%xmm2       \n\t"
       "movss %%xmm2,%%xmm6       \n\t"
       SINCOS_P4
       "movss %%xmm1,%%xmm5       \n\t"
       "mulss %%xmm0,%%xmm1       \n\t"
       "mulss %%xmm0,%%xmm2       \n\t"
       SINCOS_P5(Qimag)
       "mulss %%xmm3,%%xmm5       \n\t"
       "mulss %%xmm3,%%xmm6       \n\t"
       "addss %%xmm5,%%xmm2       \n\t"
       "subss %%xmm6,%%xmm1       \n\t"
       SINCOS_P6(Qreal)
       "movss %%xmm2,%[XPimag]    \n\t"
       "MOVSS %%xmm1,%[XPreal]    \n\t"

       /* interface */
       :
       /* output  (here: to memory)*/
       [XPreal]     "=m" (realXP),  /*REAL4*/
       [XPimag]     "=m" (imagXP),  /*REAL4*/
       [Qreal]      "=m" (realQ),   /*REAL4 sincos*/
       [Qimag]      "=m" (imagQ),   /*REAL4 sincos*/
       [sin]        "=m" (s_alpha), /*REAL4 sincos*/
       [cos]        "=m" (c_alpha), /*REAL4 sincos*/
       [tmp]        "=m" (tmp)      /*REAL8 sincos*/

       :
       /* input */
       [Xa]            "r" (Xalpha_l),     /*(*REAL4)*/
       [kappa_s]       "m" (kappa_s),      /*REAL4*/
       [kappa_star]    "m" (kstar8),       /*REAL8 sincos*/
       [_lambda_alpha] "m" (_lambda_alpha),/*REAL8 sincos*/
       [scd]           "m" (scd),          /*(*REAL4) sincos*/
       [scb]           "m" (scb),          /*(*REAL4) sincos*/
       [sincos_adds]   "m" (sincos_adds),  /*REAL8 sincos*/
       [M1]            "m" (M1),           /*REAL4*/


       /* vector constants */
       [D2222]       "m"  (D2222[0]),
       [D1100]       "m"  (D1100[0]),
       [D3322]       "m"  (D3322[0]),
       [D5544]       "m"  (D5544[0]),
       [D7766]       "m"  (D7766[0]),
       [Daabb]       "m"  (Daabb[0]),
       [Dccdd]       "m"  (Dccdd[0]),
       [Deeff]       "m"  (Deeff[0]),
       [Dgghh]       "m"  (Dgghh[0])

       :
       /* clobbered registers */
       "xmm0","xmm1","xmm2","xmm3","xmm4","xmm5","xmm6","xmm7",
       SINCOS_REGISTERS
       );

  }
}

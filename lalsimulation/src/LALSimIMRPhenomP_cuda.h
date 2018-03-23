//==============================================================================
//
// This code was developed as part of the Astronomy Data and Computing Services
// (ADACS; https://adacs.org.au) 2018A Software Support program.
//
// Written by: Gregory B. Poole
// Date:       September May 2018
//
// It is distributed under the MIT (Expat) License (see https://opensource.org/):
//
// Copyright (c) 2017 Astronomy Data and Computing Services (ADACS)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
//==============================================================================

#ifndef _LALSIMIMRPHENOMP_CUDA_H
#define _LALSIMIMRPHENOMP_CUDA_H

void PhenomPCoreAllFrequencies_cuda(int L_fCut,
        int   *errcode);

#ifdef __cplusplus
#ifdef __NVCC__
#include <string>
#include <cuda_runtime.h>


#endif
#endif
#endif

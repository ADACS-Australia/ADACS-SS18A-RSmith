/*
 * Copyright (C) 2013-2016  Leo Singer
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
 */

#include "config.h"
#include <chealpix.h>
#include <gsl/gsl_errno.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/ufuncobject.h>
#include <lal/bayestar_distance.h>


static void pdf_loop(
    char **args, npy_intp *dimensions, npy_intp *steps, void *NPY_UNUSED(data))
{
    gsl_error_handler_t *old_handler = gsl_set_error_handler_off();
    const npy_intp n = dimensions[0];

    #pragma omp parallel for
    for (npy_intp i = 0; i < n; i ++)
    {
        *(double *) &args[4][i * steps[4]] = bayestar_distance_pdf(
        *(double *) &args[0][i * steps[0]],
        *(double *) &args[1][i * steps[1]],
        *(double *) &args[2][i * steps[2]],
        *(double *) &args[3][i * steps[3]]);
    }

    gsl_set_error_handler(old_handler);
}


static void cdf_loop(
    char **args, npy_intp *dimensions, npy_intp *steps, void *NPY_UNUSED(data))
{
    gsl_error_handler_t *old_handler = gsl_set_error_handler_off();
    const npy_intp n = dimensions[0];

    #pragma omp parallel for
    for (npy_intp i = 0; i < n; i ++)
    {
        *(double *) &args[4][i * steps[4]] = bayestar_distance_cdf(
        *(double *) &args[0][i * steps[0]],
        *(double *) &args[1][i * steps[1]],
        *(double *) &args[2][i * steps[2]],
        *(double *) &args[3][i * steps[3]]);
    }

    gsl_set_error_handler(old_handler);
}


static void ppf_loop(
    char **args, npy_intp *dimensions, npy_intp *steps, void *NPY_UNUSED(data))
{
    gsl_error_handler_t *old_handler = gsl_set_error_handler_off();
    const npy_intp n = dimensions[0];

    #pragma omp parallel for
    for (npy_intp i = 0; i < n; i ++)
    {
        *(double *) &args[4][i * steps[4]] = bayestar_distance_ppf(
        *(double *) &args[0][i * steps[0]],
        *(double *) &args[1][i * steps[1]],
        *(double *) &args[2][i * steps[2]],
        *(double *) &args[3][i * steps[3]]);
    }

    gsl_set_error_handler(old_handler);
}


static void moments_to_parameters_loop(
    char **args, npy_intp *dimensions, npy_intp *steps, void *NPY_UNUSED(data))
{
    gsl_error_handler_t *old_handler = gsl_set_error_handler_off();
    const npy_intp n = dimensions[0];

    #pragma omp parallel for
    for (npy_intp i = 0; i < n; i ++)
    {
        bayestar_distance_moments_to_parameters(
            *(double *) &args[0][i * steps[0]],
            *(double *) &args[1][i * steps[1]],
             (double *) &args[2][i * steps[2]],
             (double *) &args[3][i * steps[3]],
             (double *) &args[4][i * steps[4]]);
    }

    gsl_set_error_handler(old_handler);
}


static void parameters_to_moments_loop(
    char **args, npy_intp *dimensions, npy_intp *steps, void *NPY_UNUSED(data))
{
    gsl_error_handler_t *old_handler = gsl_set_error_handler_off();
    const npy_intp n = dimensions[0];

    #pragma omp parallel for
    for (npy_intp i = 0; i < n; i ++)
    {
        bayestar_distance_parameters_to_moments(
            *(double *) &args[0][i * steps[0]],
            *(double *) &args[1][i * steps[1]],
             (double *) &args[2][i * steps[2]],
             (double *) &args[3][i * steps[3]],
             (double *) &args[4][i * steps[4]]);
    }

    gsl_set_error_handler(old_handler);
}


static void volume_render_kernel_loop(
    char **args, npy_intp *dimensions, npy_intp *steps, void *NPY_UNUSED(data))
{
    gsl_error_handler_t *old_handler = gsl_set_error_handler_off();
    const npy_intp n = dimensions[0];
    const long nside = npix2nside(dimensions[2]);

    /* Assert that rotation matrix is 3x3 */
    assert(dimensions[1] == 3);

    /* FIXME: Check that array arguments are stored contiguously */

    #pragma omp parallel for
    for (npy_intp i = 0; i < n; i ++)
    {
        *(double *) &args[11][i * steps[11]] = bayestar_volume_render_kernel(
            *(double *)   &args[0][i * steps[0]],
            *(double *)   &args[1][i * steps[1]],
            *(double *)   &args[2][i * steps[2]],
            *(int *)      &args[3][i * steps[3]],
            *(int *)      &args[4][i * steps[4]],
             (double *)   &args[5][i * steps[5]],
            nside,
            *(npy_bool *) &args[6][i * steps[6]],
             (double *)   &args[7][i * steps[7]],
             (double *)   &args[8][i * steps[8]],
             (double *)   &args[9][i * steps[9]],
             (double *)   &args[10][i * steps[10]]);
    }

    gsl_set_error_handler(old_handler);
}


static void marginal_distribution_loop(
    char **args, npy_intp *dimensions, npy_intp *steps, void *NPY_UNUSED(data))
{
    gsl_error_handler_t *old_handler = gsl_set_error_handler_off();
    const npy_intp n = dimensions[0];
    const long npix = dimensions[1];

    /* Assert that array arguments are stored contiguously */
    assert(steps[6] == sizeof(double));

    #pragma omp parallel for
    for (npy_intp i = 0; i < n; i ++)
    {
        *(double *) &args[5][i * steps[5]] =
            bayestar_marginal_distance_distribution(
            *(double *) &args[0][i * steps[0]], npix,
             (double *) &args[1][i * steps[1]],
             (double *) &args[2][i * steps[2]],
             (double *) &args[3][i * steps[3]],
             (double *) &args[4][i * steps[4]]);
    }

    gsl_set_error_handler(old_handler);
}


static const PyUFuncGenericFunction
    pdf_loops[] = {pdf_loop},
    cdf_loops[] = {cdf_loop},
    ppf_loops[] = {ppf_loop},
    moments_to_parameters_loops[] = {moments_to_parameters_loop},
    parameters_to_moments_loops[] = {parameters_to_moments_loop},
    volume_render_kernel_loops[] = {volume_render_kernel_loop},
    marginal_distribution_loops[] = {marginal_distribution_loop};

static const char volume_render_kernel_ufunc_types[] = {
    NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_INT, NPY_INT, NPY_DOUBLE, NPY_BOOL,
    NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE};

static const char double_ufunc_types[] = {
    NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE};

static const void *no_ufunc_data[] = {NULL};

static const char modulename[] = "_distance";


#if PY_MAJOR_VERSION >= 3
static PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    modulename, NULL, -1, NULL
};
#endif


#if PY_MAJOR_VERSION < 3
PyMODINIT_FUNC init_distance(void); /* Silence -Wmissing-prototypes */
PyMODINIT_FUNC init_distance(void)
#else
PyMODINIT_FUNC PyInit__distance(void); /* Silence -Wmissing-prototypes */
PyMODINIT_FUNC PyInit__distance(void)
#endif
{
    PyObject *module;
    import_array();
    import_umath();

#if PY_MAJOR_VERSION < 3
    module = Py_InitModule(modulename, NULL);
#else
    module = PyModule_Create(&moduledef);
#endif

    PyModule_AddObject(
        module, "pdf", PyUFunc_FromFuncAndData(
            pdf_loops, no_ufunc_data,
            double_ufunc_types, 1, 4, 1, PyUFunc_None,
            "pdf", NULL, 0));

    PyModule_AddObject(
        module, "cdf", PyUFunc_FromFuncAndData(
            cdf_loops, no_ufunc_data,
            double_ufunc_types, 1, 4, 1, PyUFunc_None,
            "cdf", NULL, 0));

    PyModule_AddObject(
        module, "ppf", PyUFunc_FromFuncAndData(
            ppf_loops, no_ufunc_data,
            double_ufunc_types, 1, 4, 1, PyUFunc_None,
            "ppf", NULL, 0));

    PyModule_AddObject(
        module, "moments_to_parameters", PyUFunc_FromFuncAndData(
            moments_to_parameters_loops, no_ufunc_data,
            double_ufunc_types, 1, 2, 3, PyUFunc_None,
            "moments_to_parameters", NULL, 0));

    PyModule_AddObject(
        module, "parameters_to_moments", PyUFunc_FromFuncAndData(
            parameters_to_moments_loops, no_ufunc_data,
            double_ufunc_types, 1, 2, 3, PyUFunc_None,
            "parameters_to_moments", NULL, 0));

    PyModule_AddObject(
        module, "volume_render_kernel", PyUFunc_FromFuncAndDataAndSignature(
            volume_render_kernel_loops, no_ufunc_data,
            volume_render_kernel_ufunc_types, 1, 11, 1, PyUFunc_None,
            "volume_render_kernel", NULL, 0,
            "(),(),(),(),(),(i,i),(),(n),(n),(n),(n)->()"));

    PyModule_AddObject(
        module, "marginal_distribution", PyUFunc_FromFuncAndDataAndSignature(
            marginal_distribution_loops, no_ufunc_data,
            double_ufunc_types, 1, 5, 1, PyUFunc_None,
            "marginal_distribution", NULL, 0,
            "(),(n),(n),(n),(n)->()"));

#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}

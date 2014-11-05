/*
 *  LALInferenceEnsembleSampler.c:  Bayesian Followup, ensemble-sampling algorithm.
 *
 *  Copyright (C) 2014 Ben Farr
 *
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with with program; see the file COPYING. If not, write to the
 *  Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston,
 *  MA  02111-1307  USA
 */

#include <mpi.h>
#include <lal/LALInference.h>
#include "LALInferenceEnsembleSampler.h"
#include <lal/LALInferenceProposal.h>
#include <lal/LALInferenceClusteredKDE.h>

#include <LALAppsVCSInfo.h>

#define PROGRAM_NAME "LALInferenceEnsembleSampler.c"
#define CVS_ID_STRING "$Id$"
#define CVS_REVISION "$Revision$"
#define CVS_SOURCE "$Source$"
#define CVS_DATE "$Date$"
#define CVS_NAME_STRING "$Name$"

void ensemble_sampler(struct tagLALInferenceRunState *run_state) {
    INT4 mpi_rank, mpi_size;
    INT4 walker, nwalkers_per_thread, nsteps;
    INT4 skip, update_interval, verbose;
    INT4 *step;
    INT4 *naccepts;
    char **walker_output_names = NULL;

    /* Initialize LIGO status */
    LALStatus status;
    memset(&status, 0, sizeof(status));

    /* Determine number of MPI threads, and this thread's rank */
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    /* Parameters controlling output and ensemble update frequency */
    LALInferenceVariables *algorithm_params = run_state->algorithmParams;
    nsteps = LALInferenceGetINT4Variable(algorithm_params, "nsteps");
    skip = LALInferenceGetINT4Variable(algorithm_params, "skip");
    verbose = LALInferenceGetINT4Variable(algorithm_params, "verbose");

    step = (INT4*) LALInferenceGetVariable(algorithm_params, "step");

    nwalkers_per_thread =
        LALInferenceGetINT4Variable(algorithm_params, "nwalkers_per_thread");
    update_interval =
        LALInferenceGetINT4Variable(algorithm_params, "update_interval");

    /* Initialize walker acceptance rate tracking and outputs.
     * Files are closed to avoid hitting system I/O limits */
    naccepts = XLALCalloc(nwalkers_per_thread, sizeof(INT4));
    walker_output_names = XLALCalloc(nwalkers_per_thread, sizeof(char*));
    for (walker = 0; walker < nwalkers_per_thread; walker++) {
        naccepts[walker] = 0.;
        walker_output_names[walker] =
            init_ensemble_output(run_state, walker, mpi_rank*nwalkers_per_thread, verbose);
    }

    /* Setup clustered-KDE proposal from the current state of the ensemble */
    ensemble_update(run_state);

    /* Main sampling loop */
    while (*step < nsteps) {

        /* Update step counters */
        (*step)++;

        /* Update the proposal from the current state of the ensemble */
        if ((*step % update_interval) == 0)
            ensemble_update(run_state);

        /* Update all walkers on this MPI-thread */
        #pragma omp parallel for
        for (walker=0; walker<nwalkers_per_thread; walker++) {
            INT4 accepted;
            REAL8 acceptance_rate = 1./(*step);

            LALInferenceVariables proposed_params;
            proposed_params.head = NULL;
            proposed_params.dimension = 0;

            accepted = walker_step(run_state, run_state->modelArray[walker],
                                    run_state->currentParamArray[walker],
                                    &proposed_params,
                                    &(run_state->currentPriors[walker]),
                                    &(run_state->currentLikelihoods[walker]),
                                    &(run_state->currentPropDensityArray[walker]));

            /* Track acceptance rates */
            naccepts[walker] += accepted;
            acceptance_rate *= naccepts[walker];

            /* Output sample to file */
            if ((*step % skip) == 0) {
                print_ensemble_sample(run_state, walker_output_names, walker);
                if (verbose)
                    print_proposed_sample(run_state, &proposed_params,
                                            walker, accepted);
            }

            LALInferenceClearVariables(&proposed_params);
        }

        if (verbose)
            print_acceptance_rate(run_state, naccepts, *step);
    }

    /* Sampling complete, so clean up and return */
    for (walker=0; walker<nwalkers_per_thread; walker++) {
        run_state->currentPropDensityArray[walker] = -DBL_MAX;
        XLALFree(walker_output_names[walker]);
    }
    XLALFree(walker_output_names);

    return;
}

INT4 walker_step(LALInferenceRunState *run_state,
                    LALInferenceModel *model,
                    LALInferenceVariables *current_params,
                    LALInferenceVariables *proposed_params,
                    REAL8 *current_prior, REAL8 *current_likelihood,
                    REAL8 *current_prop_density) {
    REAL8 proposed_prior = -INFINITY;
    REAL8 proposed_likelihood = -INFINITY;
    REAL8 proposal_ratio, acceptance_probability;
    INT4 accepted = 0;

    /* Propose a new sample */
    LALInferenceClearVariables(proposed_params);

    /* Get the probability of proposing the reverse jump */
    REAL8 prop_density = *current_prop_density;
    proposal_ratio = LALInferenceStoredClusterKDEProposal(run_state,
                                                            current_params,
                                                            proposed_params,
                                                            &prop_density);

    /* Only bother calculating likelihood if within prior boundaries */
    proposed_prior = run_state->prior(run_state, proposed_params, model);
    if (proposed_prior > -DBL_MAX)
        proposed_likelihood =
            run_state->likelihood(proposed_params, run_state->data, model);

    /* Find jump acceptance probability */
    acceptance_probability = (proposed_prior + proposed_likelihood)
                            - (*current_prior + *current_likelihood)
                            + proposal_ratio;

    /* Accept the jump with the calculated probability */
    if (acceptance_probability > 0
            || (log(gsl_rng_uniform(run_state->GSLrandom)) < acceptance_probability)) {
        LALInferenceCopyVariables(proposed_params, current_params);
        *current_prior = proposed_prior;
        *current_likelihood = proposed_likelihood;
        *current_prop_density = prop_density;

        accepted = 1;
    }

    return accepted;
}


void ensemble_update(LALInferenceRunState *run_state) {
    INT4 nwalkers, nwalkers_per_thread, walker, ndim, cyclic_reflective;
    REAL8 *parameters, *samples, *param_array;

    LALInferenceVariables *algorithm_params = run_state->algorithmParams;
    nwalkers = LALInferenceGetINT4Variable(algorithm_params, "nwalkers");
    nwalkers_per_thread =
        LALInferenceGetINT4Variable(algorithm_params, "nwalkers_per_thread");
    cyclic_reflective =
        LALInferenceGetINT4Variable(algorithm_params, "cyclic_reflective");

    /* Prepare array to contain samples */
    ndim = LALInferenceGetVariableDimensionNonFixed(run_state->currentParamArray[0]);
    samples = XLALCalloc(nwalkers * ndim, sizeof(REAL8));

    /* Get this thread's walkers' locations */
    param_array = XLALCalloc(nwalkers_per_thread * ndim, sizeof(REAL8));
    for (walker = 0; walker < nwalkers_per_thread; walker++) {
        parameters = &(param_array[ndim*walker]);
        LALInferenceCopyVariablesToArray(run_state->currentParamArray[walker], parameters);
    }

    /* Send all walker locations to all MPI threads */
    MPI_Allgather(param_array, nwalkers_per_thread*ndim,
                    MPI_DOUBLE, samples, nwalkers_per_thread*ndim,
                    MPI_DOUBLE, MPI_COMM_WORLD);

    /* Update the KDE proposal */
    parallel_incremental_kmeans(run_state, samples, nwalkers, cyclic_reflective);

    /* Clean up */
    XLALFree(param_array);
    XLALFree(samples);
}


/* This is a temporary, messy solution for now.
TODO: When MPI is enables in lalinference, move this routine over and clean up */
void parallel_incremental_kmeans(LALInferenceRunState *run_state,
                                    REAL8 *samples,
                                    INT4 nwalkers,
                                    INT4 cyclic_reflective) {
    INT4 i, k, ndim;
    INT4 mpi_rank, mpi_size, best_rank;
    REAL8 bic = -INFINITY;
    REAL8 best_bic = -INFINITY;
    REAL8 *bics;

    LALInferenceKmeans *kmeans;
    LALInferenceKmeans *best_clustering = NULL;

    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    bics = XLALCalloc(mpi_size, sizeof(REAL8));

    ndim = LALInferenceGetVariableDimensionNonFixed(run_state->currentParamArray[0]);
    gsl_matrix_view mview = gsl_matrix_view_array(samples, nwalkers, ndim);

    /* Keep track of clustered parameter names */
    LALInferenceVariables *backward_params =
        XLALCalloc(1, sizeof(LALInferenceVariables));
    LALInferenceVariables *cluster_params =
        XLALCalloc(1, sizeof(LALInferenceVariables));
    LALInferenceVariableItem *item;
    for (item = run_state->currentParamArray[0]->head; item; item = item->next)
        if (LALInferenceCheckVariableNonFixed(run_state->currentParamArray[0], item->name))
            LALInferenceAddVariable(backward_params, item->name,
                                    item->value, item->type, item->vary);

    for (item = backward_params->head; item; item = item->next)
        LALInferenceAddVariable(cluster_params, item->name,
                                item->value, item->type, item->vary);

    /* Have each MPI thread handle a fixed-k clustering */
    k = mpi_rank + 1;
    while (1) {
        kmeans =
            LALInferenceKmeansRunBestOf(k, &mview.matrix, 8, run_state->GSLrandom);
        bic = -INFINITY;
        if (kmeans)
            bic = LALInferenceKmeansBIC(kmeans);
        if (bic > best_bic) {
            if (best_clustering)
                LALInferenceKmeansDestroy(best_clustering);
            best_clustering = kmeans;
            best_bic = bic;
        } else {
            LALInferenceKmeansDestroy(kmeans);
            break;
        }
    }

    MPI_Gather(&best_bic, 1, MPI_DOUBLE, bics, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (mpi_rank == 0) {
        best_bic = -INFINITY;
        for (i = 0; i < mpi_size; i++) {
           if (bics[i] > best_bic) {
               best_bic = bics[i];
               best_rank = i;
           }
        }
    }

    /* Send the winning k-size to everyone */
    k = best_clustering->k;
    MPI_Bcast(&best_rank, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&k, 1, MPI_INT, best_rank, MPI_COMM_WORLD);

    /* Create a kmeans instance with the winning k */
    if (mpi_rank != best_rank) {
        if (best_clustering)
            LALInferenceKmeansDestroy(best_clustering);
        best_clustering =
                LALInferenceKmeansRunBestOf(k, &mview.matrix, 8, run_state->GSLrandom);
    }

    /* Broadcast cluster assignments */
    MPI_Bcast(best_clustering->assignments, nwalkers,
                MPI_INT, best_rank, MPI_COMM_WORLD);

    /* Calculate centroids, the run to compute sizes and weights */
    LALInferenceKmeansUpdate(best_clustering);
    LALInferenceKmeansRun(best_clustering);

    LALInferenceClusteredKDE *proposal =
        XLALCalloc(1, sizeof(LALInferenceClusteredKDE));

    proposal->kmeans = best_clustering;

    LALInferenceInitClusteredKDEProposal(run_state, proposal,
                                            samples, nwalkers,
                                            cluster_params, "ClusteredKDEProposal",
                                            1.0, NULL, cyclic_reflective, 0);

    LALInferenceAddClusteredKDEProposalToSet(run_state, proposal);

    LALInferenceClearVariables(backward_params);
    XLALFree(backward_params);
    XLALFree(bics);
}


//-----------------------------------------
// file output routines:
//-----------------------------------------
char *init_ensemble_output(LALInferenceRunState *run_state,
                            INT4 walker,
                            INT4 walker_offset,
                            INT4 verbose) {
    ProcessParamsTable *ppt;
    INT4 randomseed;
    char *outfile_name = NULL;
    char *prop_name = NULL;
    FILE *walker_output = NULL;

    /* Randomseed used to prevent overwriting when peforming multiple analyses */
    randomseed = LALInferenceGetINT4Variable(run_state->algorithmParams, "random_seed");

    /* Decide on file name(s) and open */
    ppt = LALInferenceGetProcParamVal(run_state->commandLine, "--outfile");
    if (ppt) {
        outfile_name = (char*) XLALCalloc(strlen(ppt->value)+255, sizeof(char*));
        sprintf(outfile_name, "%s.%i", ppt->value, walker);
    } else {
        outfile_name = (char*) XLALCalloc(255, sizeof(char*));
        sprintf(outfile_name,
                "ensemble.output.%u.%i", randomseed, walker_offset+walker);
    }

    if (verbose) {
        prop_name = (char*) XLALCalloc(255, sizeof(char*));
        sprintf(prop_name, "ensemble.proposed.%u.%i",
                randomseed, walker_offset+walker);
    }

    walker_output = fopen(outfile_name, "w");
    if (walker_output == NULL){
        XLALErrorHandler = XLALExitErrorHandler;
        XLALPrintError("Output file error in %s, line %d. %s.\n",
                        __FILE__, __LINE__, strerror(errno));
        XLAL_ERROR_NULL(XLAL_EIO);
    }

    /* Print headers to file */
    print_ensemble_header(run_state, walker_output, walker);

    /* Close to avoid hitting system I/O limits when using many walkers */
    fclose(walker_output);

    /* Extra outputs when running verbosely */
    if (verbose) {
        FILE *prop_out = fopen(prop_name, "w");
        LALInferenceFprintParameterNonFixedHeaders(prop_out, run_state->currentParamArray[0]);
        fprintf(prop_out, "accepted\n");
        fclose(prop_out);
        XLALFree(prop_name);
    }

    return outfile_name;
}

void print_ensemble_sample(LALInferenceRunState *run_state,
                            char **walker_output_names,
                            INT4 walker) {
    REAL8 null_likelihood, timestamp, timestamp_epoch;
    REAL8 *current_priors, *current_likelihoods;
    INT4 step;
    INT4 benchmark;
    FILE *walker_output;

    step = LALInferenceGetINT4Variable(run_state->algorithmParams, "step");
    walker_output = fopen(walker_output_names[walker], "a");
    current_priors = run_state->currentPriors;
    current_likelihoods = run_state->currentLikelihoods;

    /* Print step number, log(posterior) */
    null_likelihood = LALInferenceGetREAL8Variable(run_state->proposalArgs, "nullLikelihood");
    fprintf(walker_output, "%d\t", step);
    fprintf(walker_output, "%f\t",
            (current_likelihoods[walker] - null_likelihood) + current_priors[walker]);

    /* Print the non-fixed parameter values */
    LALInferencePrintSampleNonFixed(walker_output, run_state->currentParamArray[walker]);

    /* Print log(prior) and log(likelihood)  */
    fprintf(walker_output, "%f\t", current_priors[walker]);
    fprintf(walker_output, "%f\t", current_likelihoods[walker] - null_likelihood);

    /* Keep track of wall time if benchmarking */
    benchmark = LALInferenceGetINT4Variable(run_state->algorithmParams, "benchmark");
    if (benchmark) {
        struct timeval tv;
        timestamp_epoch = LALInferenceGetREAL8Variable(run_state->algorithmParams,
                                                        "timestamp_epoch");

        gettimeofday(&tv, NULL);
        timestamp = tv.tv_sec + tv.tv_usec/1E6 - timestamp_epoch;
        fprintf(walker_output, "%f\t", timestamp);
    }
    fprintf(walker_output,"\n");

    /* Close file before returning */
    fclose(walker_output);
}


void print_proposed_sample(LALInferenceRunState *run_state,
                            LALInferenceVariables *proposed_params,
                            INT4 walker,
                            INT4 accepted) {
    INT4 mpi_rank;
    INT4 nwalkers_per_thread;
    INT4 randomseed;
    FILE *output = NULL;
    char *outname = (char*) XLALCalloc(255, sizeof(char*));

    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);    // This thread's index

    LALInferenceVariables *algorithm_params = run_state->algorithmParams;
    randomseed = LALInferenceGetINT4Variable(algorithm_params, "random_seed");
    nwalkers_per_thread =
        LALInferenceGetINT4Variable(algorithm_params, "nwalkers_per_thread");

    sprintf(outname, "ensemble.proposed.%u.%2.2d",
            randomseed, nwalkers_per_thread*mpi_rank+walker);

    output = fopen(outname, "a");
    LALInferencePrintSampleNonFixed(output, proposed_params);
    fprintf(output, "%i\n", accepted);

    fclose(output);
    XLALFree(outname);
}


void print_acceptance_rate(LALInferenceRunState *run_state,
                            INT4 *naccepts,
                            INT4 step) {
    INT4 mpi_rank, walker;
    INT4 nwalkers_per_thread;
    INT4 randomseed;
    FILE *output = NULL;
    char *outname = (char*) XLALCalloc(255, sizeof(char*));

    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);    // This thread's index

    LALInferenceVariables *algorithm_params = run_state->algorithmParams;
    randomseed = LALInferenceGetINT4Variable(algorithm_params, "random_seed");
    nwalkers_per_thread =
        LALInferenceGetINT4Variable(algorithm_params, "nwalkers_per_thread");

    sprintf(outname, "ensemble.rates.%u.%i", randomseed, mpi_rank);

    output = fopen(outname, "a");
    for (walker = 0; walker < nwalkers_per_thread; walker++)
        fprintf(output, "%f\t", naccepts[walker]/((REAL8) step));
    fprintf(output, "\n");

    fclose(output);
    XLALFree(outname);
}


void print_ensemble_header(LALInferenceRunState *run_state,
                            FILE *walker_output,
                            INT4 walker) {
    LALInferenceVariables **current_param_array;
    LALInferenceIFOData *ifo_data;
    REAL8TimeSeries *time_data;
    INT4 ndim, int_pn_order, waveform = 0;
    INT4 nifo, randomseed, benchmark;
    REAL8 null_likelihood, normed_logl, pn_order=-1.0;
    REAL8 network_snr, sampling_rate;
    REAL8 timestamp, f_ref = 0.0;
    struct timeval tv;
    char *cmd_str;

    /* Save the command line for repoducability */
    cmd_str = LALInferencePrintCommandLine(run_state->commandLine);

    randomseed = LALInferenceGetINT4Variable(run_state->algorithmParams, "random_seed");
    null_likelihood = LALInferenceGetREAL8Variable(run_state->proposalArgs, "nullLikelihood");
    current_param_array = run_state->currentParamArray;
    ndim = LALInferenceGetVariableDimensionNonFixed(current_param_array[walker]);

    /* Reference frequency for evolving parameters */
    if (LALInferenceCheckVariable(current_param_array[walker], "fRef"))
        f_ref = LALInferenceGetREAL8Variable(current_param_array[walker], "fRef");

    /* Count number of detectors */
    ifo_data = run_state->data;
    nifo = 0;
    while (ifo_data){
      nifo++;
      ifo_data = ifo_data->next;
    }

    /* Integer (from an enum) identfying the waveform family used */
    if (LALInferenceCheckVariable(current_param_array[walker], "LAL_APPROXIMANT"))
        waveform = LALInferenceGetINT4Variable(current_param_array[walker], "LAL_APPROXIMANT");

    /* Determine post-Newtonian (pN) order (half of the integer stored in currentParams) */
    if (LALInferenceCheckVariable(current_param_array[walker], "LAL_PNORDER")) {
        int_pn_order = LALInferenceGetINT4Variable(current_param_array[walker], "LAL_PNORDER");
        pn_order = int_pn_order/2.0;
    }

    /* Calculated the network signal-to-noise ratio if an injection was done */
    ifo_data = run_state->data;
    network_snr = 0.0;
    while (ifo_data) {
        network_snr += ifo_data->SNR * ifo_data->SNR;
        ifo_data = ifo_data->next;
    }
    network_snr = sqrt(network_snr);

    /* Keep track of time if benchmarking */
    benchmark = LALInferenceGetINT4Variable(run_state->algorithmParams, "benchmark");

    /* Write the header information to file */
    fprintf(walker_output,
            "  LALInference version:%s,%s,%s,%s,%s\n",
            lalAppsVCSId, lalAppsVCSDate, lalAppsVCSBranch,
            lalAppsVCSAuthor, lalAppsVCSStatus);

    fprintf(walker_output, "  %s\n", cmd_str);

    fprintf(walker_output, "%6s\t%20s\t%6s\t%12s\t%9s\t%9s\t%8s\t%8s\n",
        "seed", "null_likelihood", "ndet", "network_snr", "waveform", "pn_order", "ndim", "f_ref");

    fprintf(walker_output, "%u\t%20.10lf\t%6d\t%14.6f\t%9i\t%9.1f\t%8i\t%12.1f\n",
        randomseed, null_likelihood, nifo, network_snr, waveform, pn_order, ndim, f_ref);

    /* Use time step in time-domain data to determine sampling rate */
    fprintf(walker_output, "\n%16s\t%16s\t%10s\t%10s\t%10s\t%10s\t%20s\n",
        "Detector", "SNR", "f_low", "f_high", "start_time", "segment_length", "sampling_rate");
    ifo_data=run_state->data;
    while(ifo_data){
        time_data = ifo_data->timeData;
        sampling_rate = 1.0/time_data->deltaT;
        fprintf(walker_output,
                "%16s\t%16.8lf\t%10.2lf\t%10.2lf\t%15.7lf\t%12d\t%10.2lf\n",
                ifo_data->detector->frDetector.name,
                ifo_data->SNR, ifo_data->fLow, ifo_data->fHigh,
                XLALGPSGetREAL8(&(ifo_data->epoch)), time_data->data->length, sampling_rate);
        ifo_data=ifo_data->next;
    }
    fprintf(walker_output, "\n\n\n");

    /* These are the actual column headers for the samples to be output */
    fprintf(walker_output, "cycle\tlogpost\t");

    LALInferenceFprintParameterNonFixedHeaders(walker_output, current_param_array[walker]);

    fprintf(walker_output, "logprior\tlogl\t");

    if (benchmark)
        fprintf(walker_output, "timestamp\t");
    fprintf(walker_output,"\n");

    /* Print starting values as 0th iteration */
    normed_logl = run_state->currentLikelihoods[walker]-null_likelihood;
    fprintf(walker_output, "%d\t%f\t", 0,
            run_state->currentPriors[walker] + normed_logl);

    LALInferencePrintSampleNonFixed(walker_output, current_param_array[walker]);

    fprintf(walker_output, "%f\t%f\t",
            run_state->currentPriors[walker],
            run_state->currentLikelihoods[walker] - null_likelihood);

    if (benchmark) {
        gettimeofday(&tv, NULL);
        timestamp = tv.tv_sec + tv.tv_usec/1E6;
        LALInferenceAddVariable(run_state->algorithmParams, "timestamp_epoch", &timestamp,
                                LALINFERENCE_REAL8_t, LALINFERENCE_PARAM_FIXED);
        fprintf(walker_output, "%f\t", 0.0);
    }
    fprintf(walker_output,"\n");

    XLALFree(cmd_str);
}

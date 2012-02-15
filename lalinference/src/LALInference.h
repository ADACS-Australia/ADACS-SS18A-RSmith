/*
 *
 *  LALInference:             Bayesian Followup        
 *  include/LALInference.h:   main header file
 *
 *  Copyright (C) 2009 Ilya Mandel, Vivien Raymond, Christian Roever, Marc van der Sluys and John Veitch
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

/**
 * \file LALInference.h
 * \brief Main header file for LALInference common routines and structures
 * \ingroup LALInference
 * 
 * LALInference is a Bayesian analysis toolkit for use with LAL. It contains
 * common requirements for Bayesian codes such as Likelihood functions, data
 * handling routines, MCMC and Nested Sampling algorithms and a template generation
 * interface to the LALInspiral package.
 * 
 * This file contains the basic structures for the algorithm state, inteferometer
 * data, manipulation of variables and type declarations for the standard function types.
 * 
 * 
 */



#ifndef LALInference_h
#define LALInference_h

# include <math.h>
# include <stdio.h>
# include <stdlib.h>

#define VARNAME_MAX 128
#define VARVALSTRINGSIZE_MAX 128

# include <lal/LALStdlib.h>
# include <lal/LALConstants.h>
# include <lal/SimulateCoherentGW.h>
# include <lal/GeneratePPNInspiral.h>
# include <lal/LIGOMetadataTables.h>
# include <lal/LALDatatypes.h>
# include <lal/FindChirp.h>
# include <lal/Window.h>
#include <lal/LALString.h>

#include <lal/SFTutils.h>
#include <lal/SFTfileIO.h>
#include <lal/LALDetectors.h>
#include <lal/LALBarycenter.h>
#include <lal/LALInitBarycenter.h>
#include <lal/BinaryPulsarTiming.h>


#include <gsl/gsl_linalg.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_min.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <sys/time.h>

//...other includes

struct tagLALInferenceRunState;
struct tagLALInferenceIFOData;

/*Data storage type definitions*/

/** An enumerated type for denoting the type of a variable. Several LAL
 * types are supported as well as others.
*/
typedef enum {
  LALINFERENCE_INT4_t, 		
  LALINFERENCE_INT8_t,
  LALINFERENCE_UINT4_t,
  LALINFERENCE_REAL4_t, 
  LALINFERENCE_REAL8_t, 
  LALINFERENCE_COMPLEX8_t, 
  LALINFERENCE_COMPLEX16_t, 
  LALINFERENCE_gslMatrix_t,
  LALINFERENCE_REAL8Vector_t,
  LALINFERENCE_UINT4Vector_t,
  LALINFERENCE_string_t,
  LALINFERENCE_void_ptr_t
} LALInferenceVariableType;

/** An enumerated type for denoting time or frequency domain
*/
typedef enum {
  LALINFERENCE_DOMAIN_TIME, 
  LALINFERENCE_DOMAIN_FREQUENCY
} LALInferenceDomain;

/** An enumerated type for denoting the topolology of a parameter.
 * This information is used by the sampling routines when deciding
 * what to vary in a proposal, etc.
*/
typedef enum {
	LALINFERENCE_PARAM_LINEAR, /** A parameter that simply has a maximum and a minimum */
	LALINFERENCE_PARAM_CIRCULAR, /** A parameter that is cyclic, such as an angle between 0 and 2pi */
	LALINFERENCE_PARAM_FIXED,    /** A parameter that never changes, functions should respect this */
	LALINFERENCE_PARAM_OUTPUT    /** A parameter changed by an inner code and passed out */
} LALInferenceParamVaryType;

/** An enumerated type for denoting a type of taper
*/
typedef enum
{
	LALINFERENCE_TAPER_NONE,
	LALINFERENCE_TAPER_START,
	LALINFERENCE_TAPER_END,
	LALINFERENCE_TAPER_STARTEND,
	LALINFERENCE_TAPER_NUM_OPTS,
	LALINFERENCE_RING,
	LALINFERENCE_SMOOTH
}  LALInferenceApplyTaper;

extern size_t LALInferenceTypeSize[];

/** The LALInferenceVariableItem list node structure
 * This should only be accessed using the accessor functions below
 * Implementation may change to hash table so please use only the
 * accessor functions below.
*/
typedef struct
tagVariableItem
{
  char                    name[VARNAME_MAX];
  void                    *value;
  LALInferenceVariableType		type;
  LALInferenceParamVaryType		vary;
  struct tagVariableItem		*next;
} LALInferenceVariableItem;

/** The LALInferenceVariables structure to contain a set of parameters
 * Implemented as a linked list of LALInferenceVariableItems.
 * Should only be accessed using the accessor functions below
 */
typedef struct
tagLALInferenceVariables
{
  LALInferenceVariableItem	*head;
  INT4 				dimension;
} LALInferenceVariables;

/** Returns an array of header strings (terminated by NULL) from a common-format output file */
char **LALInferenceGetHeaderLine(FILE *inp);

/** Converts between internally used parameter names and those external (e.g. in SimInspiralTable?) */
const char *LALInferenceTranslateInternalToExternalParamName(const char *inName);

/** Print the parameters which do not vary to a file as a tab-separated ASCII line
 * \param out [in] pointer to output file
 * \param params [in] LALInferenceVaraibles structure to print
 */
INT4 LALInferenceFprintParameterNonFixedHeaders(FILE *out, LALInferenceVariables *params);

/** Prints a variable item to a string (must be pre-allocated!) */
void LALInferencePrintVariableItem(char *out, LALInferenceVariableItem *ptr);

/** Return a pointer to the memory the variable is stored in specified by \param name
 * User must cast this pointer to the expected type before dereferencing
 * it to get the value of the variable.
 */
void *LALInferenceGetVariable(const LALInferenceVariables * vars, const char * name);

/** Get number of dimensions in this variable */
INT4 LALInferenceGetVariableDimension(LALInferenceVariables *vars);

/** Get number of dimensions which are not fixed to a certain value */
INT4 LALInferenceGetVariableDimensionNonFixed(LALInferenceVariables *vars);

/** Get the LALInferenceVariableType of the \param idx -th item in the \param vars
 * Indexing starts at 1
 */
LALInferenceVariableType LALInferenceGetVariableTypeByIndex(LALInferenceVariables *vars, int idx);

/** Get the LALInferenceVariableType of the parameter named \param name in \param vars */
LALInferenceVariableType LALInferenceGetVariableType(const LALInferenceVariables *vars, const char *name);

/** Get the LALInferenceParamVaryType of the parameter named \param name in \param vars
 * see the declaration of LALInferenceParamVaryType for possibilities
 */
LALInferenceParamVaryType LALInferenceGetVariableVaryType(LALInferenceVariables *vars, const char *name);

/** Get the name of the idx-th variable
 * Indexing starts at 1
 */
char *LALInferenceGetVariableName(LALInferenceVariables *vars, int idx);

/** Set a variable named \param name in \param vars with a value.
 * Pass a void * in \param value to the value you wish to set,
 * i.e. LALInferenceSetVariable(vars, "mu", (void *)&mu);
 */
void LALInferenceSetVariable(LALInferenceVariables * vars, const char * name, void * value);

/** Add a variable named \param name to \param vars with initial value referenced by \param value
 * \param type is a LALInferenceVariableType (enumerated above) 
 * \param vary is a LALInferenceParamVaryType (enumerated above)
 * If the variable already exists it will be over-written UNLESS IT HAS A CONFLICTING TYPE
 */
void LALInferenceAddVariable(LALInferenceVariables * vars, const char * name, void * value, 
	LALInferenceVariableType type, LALInferenceParamVaryType vary);

/** Remove \param name from \param vars.
 * Frees the memory for the \param name structure and its contents
 */
void LALInferenceRemoveVariable(LALInferenceVariables *vars,const char *name);

/** Checks for \param name being present in \param vars.
 *  returns 1(==true) or 0
 */
int  LALInferenceCheckVariable(LALInferenceVariables *vars,const char *name);

/** Checks for \param name being present in \param vars and having type LINEAR or CIRCULAR.
 * returns 1 or 0
 */
int LALInferenceCheckVariableNonFixed(LALInferenceVariables *vars, const char *name);

/** Delete the variables in this structure.
 *  Does not free the LALInferenceVariables itself
 *  \param vars will have its dimension set to 0 */
void LALInferenceDestroyVariables(LALInferenceVariables *vars);

/** Deep copy the variables from one to another LALInferenceVariables structure */
void LALInferenceCopyVariables(LALInferenceVariables *origin, LALInferenceVariables *target);

/** Print variables to stdout */
void LALInferencePrintVariables(LALInferenceVariables *var);

/** Check for equality in two variables */
int LALInferenceCompareVariables(LALInferenceVariables *var1, LALInferenceVariables *var2);


//Wrapper for template computation 
//(relies on LAL libraries for implementation) <- could be a #DEFINE ?
//typedef void (LALTemplateFunction) (LALInferenceVariables *currentParams, struct tagLALInferenceIFOData *data); //Parameter Set is modelParams of LALInferenceIFOData
/** Type declaration for template function, which operates on
 a LALInferenceIFOData structure \param *data */
typedef void (LALInferenceTemplateFunction) (struct tagLALInferenceIFOData *data);


/** Jump proposal distribution
 * Computes \param proposedParams based on \param currentParams 
 * and additional variables stored as proposalArgs inside \param runState,
 * which could include correlation matrix, etc.,
 * as well as forward and reverse proposal probability.
 * A jump proposal distribution function could call other jump proposal
 * distribution functions with various probabilities to allow for multiple
 * jump proposal distributions
 */
typedef void (LALInferenceProposalFunction) (struct tagLALInferenceRunState *runState,
	LALInferenceVariables *proposedParams);

/** Jump proposal statistics
 * Stores the weight given for a proposal function, the number of times
 * it has been proposed, and the number of times it has been accepted
 */
typedef struct
tagLALInferenceProposalStatistics
{
  UINT4   weight;     // Weight of proposal function in cycle
  UINT4   proposed;   // Number of times proposal has been called
  UINT4   accepted;   // Number of times a proposal from this function has been accepted
} LALInferenceProposalStatistics;

/** Type declaration for prior function which returns p(\param params)
  * Can depend on \param runState ->priorArgs
  */
typedef REAL8 (LALInferencePriorFunction) (struct tagLALInferenceRunState *runState,
	LALInferenceVariables *params);

//Likelihood calculator 
//Should take care to perform expensive evaluation of h+ and hx 
//only once if possible, unless necessary because different IFOs 
//have different data lengths or sampling rates 
/** Type declaration for likelihood function
 * Computes p(\param data | \param currentParams, \param template)
 * \param template is a LALInferenceTemplateFunction defined below
 */
typedef REAL8 (LALInferenceLikelihoodFunction) (LALInferenceVariables *currentParams,
        struct tagLALInferenceIFOData * data, LALInferenceTemplateFunction *template);

/** Perform one step of an algorithm, replaces \param runState ->currentParams */
typedef void (LALInferenceEvolveOneStepFunction) (struct tagLALInferenceRunState *runState);

/** Type declaration for an algorithm function which is called by the driver code
 * The user must initialise runState before use. The Algorithm manipulates
 * \param runState to do its work
 */
typedef void (LALInferenceAlgorithm) (struct tagLALInferenceRunState *runState);

/** Type declaration for output logging function, can be user-declared */
typedef void (LALInferenceLogFunction) (struct tagLALInferenceRunState *runState, LALInferenceVariables *vars);

/** Structure containing inference run state
 * This includes pointers to the function types required to run
 * the algorithm, and data structures as required */
typedef struct 
tagLALInferenceRunState
{
  ProcessParamsTable        *commandLine; /** A ProcessParamsTable with command line arguments */
  LALInferenceAlgorithm              *algorithm; /** The algorithm function */
  LALInferenceEvolveOneStepFunction  *evolve; /** The algorithm's single iteration function */
  LALInferencePriorFunction          *prior; /** The prior for the parameters */
  LALInferenceLikelihoodFunction     *likelihood; /** The likelihood function */
  LALInferenceProposalFunction       *proposal; /** The proposal function */
  LALInferenceTemplateFunction       *template; /** The template generation function */
  LALInferenceLogFunction	     *logsample; /** Log sample, i.e. to disk */
  struct tagLALInferenceIFOData      *data; /** The data from the interferometers */
  LALInferenceVariables              *currentParams, /** The current parameters */
    *priorArgs,                                      /** Any special arguments for the prior function */
    *proposalArgs,                                   /** Any special arguments for the proposal function */
    *proposalStats,                                  /** Set of structs containing statistics for each proposal*/
    *algorithmParams;                                /** Parameters which control the running of the algorithm*/
  LALInferenceVariables				**livePoints; /** Array of live points for Nested Sampling */
  LALInferenceVariables **differentialPoints;        /** Array of points for differential evolution */
  size_t differentialPointsLength;                   /** Length of the current differential points stored in 
                                                         differentialPoints.  This should be removed can be given 
                                                         as an algorithmParams entry */
  size_t differentialPointsSize;                     /** Size of the differentialPoints memory block 
                                                         (must be >= length of differential points).  
                                                         Can also be removed. */
  REAL8			currentLikelihood;  /** This should be removed, can be given as an algorithmParams or proposalParams entry */
  REAL8                 currentPrior;       /** This should be removed, can be given as an algorithmParams entry */
  gsl_rng               *GSLrandom;         /** A pointer to a GSL random number generator */
} LALInferenceRunState;


#define DETNAMELEN 256

/** Structure to contain IFO data.
 *  Some fields may be left empty if not needed
*/
typedef struct
tagLALInferenceIFOData
{
  char                       name[DETNAMELEN]; /** Detector name */
  REAL8TimeSeries           *timeData,         /** A time series from the detector */
                            *timeModelhPlus, *timeModelhCross, /** Time series model buffers */
                            *whiteTimeData, *windowedTimeData; /** white is not really white, but over-white. */
  /* Stores the log(L) for the model in presence of data.  These were
     added to allow for individual-detector log(L) output.  The
     convention is that loglikelihood always stores the log(L) for the
     model in freqModel... or timeModel....  When a jump is accepted,
     that value is copied into acceptedloglikelihood, which is the
     quantity that is actually output in the output files. */
  REAL8                      nullloglikelihood, loglikelihood, acceptedloglikelihood; 
  REAL8                      fPlus, fCross; /** Detector responses */
  REAL8                      timeshift;     /** What is this? */
  COMPLEX16FrequencySeries  *freqData,      /** Buffer for frequency domain data */
                            *freqModelhPlus, *freqModelhCross, /** Buffers for frequency domain models */
                            *whiteFreqData; /* Over-white. */
  COMPLEX16TimeSeries       *compTimeData, *compModelData; /** Complex time series data buffers */
  LIGOTimeGPSVector         *dataTimes;                    /** Vector of time stamps for time domain data */
  LALInferenceVariables              *modelParams;         /** Parameters used when filling the buffers - template functions should copy to here */
  LALInferenceVariables		    *dataParams; /* Optional data parameters */
  LALInferenceDomain                 modelDomain;         /** Domain of model */
  REAL8FrequencySeries      *oneSidedNoisePowerSpectrum;  /** one-sided Noise Power Spectrum */
  REAL8TimeSeries           *timeDomainNoiseWeights; /** Roughly, InvFFT(1/Noise PSD). */
  REAL8Window               *window;                 /** A window */
  REAL8FFTPlan              *timeToFreqFFTPlan, *freqToTimeFFTPlan; /** Pre-calculated FFT plans for forward and reverse FFTs */
  REAL8                     fLow, fHigh;	/** integration limits for overlap integral in F-domain */
  LALDetector               *detector;          /** LALDetector structure for where this data came from */
  BarycenterInput           *bary;              /** Barycenter information */
  EphemerisData             *ephem;             /** Ephemeris data */
  LIGOTimeGPS		    epoch;              /** The epoch of this observation (the time of the first sample) */
  REAL8                     SNR;                /** IF INJECTION ONLY, E(SNR) of the injection in the detector.*/
  REAL8                     STDOF;              /** Degrees of freedom for IFO to be used in Student-T Likelihood. */

  struct tagLALInferenceIFOData      *next;     /** A pointer to the next set of data for linked list */
} LALInferenceIFOData;

/** Returns the element of the process params table with "name" */
ProcessParamsTable *LALInferenceGetProcParamVal(ProcessParamsTable *procparams,const char *name);

/** parses a character string (passed as one of the options) and decomposes   
 it into individual parameter character strings. \param input is of the form
   input   :  "[one,two,three]"
 and the resulting \param output is
   strings :  {"one", "two", "three"}   
 length of parameter names is for now limited to 512 characters. 
 (should 'theoretically' (untested) be able to digest white space as well.
 Irrelevant for command line options, though.)                             */
void LALInferenceParseCharacterOptionString(char *input, char **strings[], UINT4 *n);

/** Return a ProcessParamsTable from the command line arguments */
ProcessParamsTable *LALInferenceParseCommandLine(int argc, char *argv[]);

/** Output the command line to \param str based on the ProcessParamsTable */
void LALInferencePrintCommandLine(ProcessParamsTable *procparams, char *str);

/** Execute FFT for data in \param data */
void LALInferenceExecuteFT(LALInferenceIFOData *IFOdata);
/** Execute Inverse FFT for data in \param data */
void LALInferenceExecuteInvFT(LALInferenceIFOData *IFOdata);

/** Return the list node for "name" - do not rely on this */
LALInferenceVariableItem *LALInferenceGetItem(const LALInferenceVariables *vars,const char *name);

/** Return the list node for the idx-th item - do not rely on this
  Indexing starts at 1
  */
LALInferenceVariableItem *LALInferenceGetItemNr(LALInferenceVariables *vars, int idx);

/** Output the sample to file *fp, in ASCII format */
void LALInferencePrintSample(FILE *fp,LALInferenceVariables *sample);

/** Output only non-fixed parameters */
void LALInferencePrintSampleNonFixed(FILE *fp,LALInferenceVariables *sample);

/** Output proposal statistics header to file *fp */
int LALInferencePrintProposalStatsHeader(FILE *fp,LALInferenceVariables *propStats);

/** Output proposal statistics to file *fp */
void LALInferencePrintProposalStats(FILE *fp,LALInferenceVariables *propStats);

/** Reads one line from the given file and stores the values there into
   the variable structure, using the given header array to name the
   columns.  Returns 0 on success. */
void LALInferenceProcessParamLine(FILE *inp, char **headers, LALInferenceVariables *vars);

/** Sorts the variable structure by name */
void LALInferenceSortVariablesByName(LALInferenceVariables *vars);

/** Append the sample to a file. file pointer is stored in state->algorithmParams as a
 * LALInferenceVariable called "outfile", as a void ptr.
 * Caller is responsible for opening and closing file.
 * Variables are alphabetically sorted before being written
 */
void LALInferenceLogSampleToFile(LALInferenceRunState *state, LALInferenceVariables *vars);

/** Append the sample to an array which can be later processed by the user.
 * Array is stored as a C array in a LALInferenceVariable in state->algorithmParams
 * called "outputarray". Number of items in the array is stored as "N_outputarray".
 * Will create the array and store it in this way if it does not exist.
 * DOES NOT FREE ARRAY, user must clean up after use.
 * Also outputs sample to disk if possible using LALInferenceLogSampleToFile()*/
void LALInferenceLogSampleToArray(LALInferenceRunState *state, LALInferenceVariables *vars);

/** Convert from Mc, eta space to m1, m2 space (note m1 > m2).*/
void LALInferenceMcEta2Masses(double mc, double eta, double *m1, double *m2);

/** Convert from Mc, q space to m1, m2 space (q = m2/m1, with m1 > m2). */
void LALInferenceMcQ2Masses(double mc, double q, double *m1, double *m2);

/** Convert from q to eta (q = m2/m1, with m1 > m2). */
void LALInferenceQ2Eta(double q, double *eta);

/** A kD tree cell contains some points (npts), a bounding box
    enclosing the cell (lowerLeft to upperRight), a bounding box
    tightly enclosing all the points currently in the cell
    (pointsLowerLeft to pointsUpperRight), and two sub-cells, left and
    right, which split the bounding box in half along one coordinate
    dimension (in N dimensions, the coordinate dimension that splits
    is given by the level of the cell in the tree mod N). */
typedef struct tagLALInferenceKDCell {
  size_t npts; /** Stores the number of tree points that lie in the cell. */
  REAL8 *lowerLeft; /** Lower left (i.e. coordinate minimum) bound;
                         length is ndim from LALInferenceKDTree. */
  REAL8 *upperRight; /** Upper right (i.e. coordinate maximum) bound. */
  REAL8 *pointsLowerLeft; /** Lower left for the contained points. */
  REAL8 *pointsUpperRight; /** Upper right for contained points. */
  struct tagLALInferenceKDCell *left; /** Left (i.e. lower-coordinate)
                                          sub-tree, may be NULL if
                                          empty.*/
  struct tagLALInferenceKDCell *right; /** Right
                                           (i.e. upper-coordinate)
                                           sub-tree, may be NULL if
                                           empty. */
} LALInferenceKDCell;

/** The kD trees used in LALInference are not quite the standard kD
    trees.  Our kD trees split the domain exactly in half along
    successive dimensions (at level i, in N dimensions, the dimension
    that is split is i%N), producing left- and right-sub-cells until
    each cell contains either zero or one point.  The advantage of
    this structure over the standard kD tree that splits the domain
    along the median coordinate of the points at each level is that
    this structure can be updated incrementally by the
    LALInferenceKDAddPoint() function. 

    To produce a kD tree from a set of points, add them one-at-a-time
    using LALInferenceKDAddPoint() and starting with an empty tree
    produced by LALInferenceKDEmpty().*/
typedef struct {
  size_t npts; /** The number of points. */
  size_t ndim; /** Each point is ndim long. */
  REAL8 **pts; /** Array of points. */
  LALInferenceKDCell *topCell; /** The top-level cell in the tree. */
} LALInferenceKDTree;

/** Delete a kD-tree.  Also deletes all contained cells, and points. */
void LALInferenceKDTreeDelete(LALInferenceKDTree *tree);

/** Constructs a fresh, empty kD tree.  The top-level cell will get
    the given bounds, which should enclose every point added by
    LALInferenceKDAddPoint(). */
LALInferenceKDTree *LALInferenceKDEmpty(REAL8 *lowerLeft, REAL8 *upperRight, size_t ndim);

/** Adds a point to the kD-tree, returns 0 on successful exit. */
int LALInferenceKDAddPoint(LALInferenceKDTree *tree, REAL8 *pt);

/** Returns the first cell that contains the given point that also
    contains fewer than Npts points, if possible.  If no cell
    containing the given point has fewer than Npts points, then
    returns the cell containing the fewest number of points and the
    given point.  Non-positive Npts will give the fewest-point cell in
    the tree containing the given point.  Returns NULL on error. */
LALInferenceKDCell *LALInferenceKDFindCell(LALInferenceKDTree *tree, REAL8 *pt, size_t Npts);

/** Returns the volume of the given cell, which is part of the given
    tree. */
double LALInferenceKDLogCellVolume(LALInferenceKDTree *tree, LALInferenceKDCell *cell);

/** Returns the volume of a box that tightly encloses the points in
    the given cell, which is part of the given tree. */
double LALInferenceKDLogPointsVolume(LALInferenceKDTree *tree, LALInferenceKDCell *cell);

/** Fills in the given REAL8 array with the parameter values from
    params; the ordering of the variables is taken from the order of
    the non-fixed variables in template.  It is an error if pt does
    not point to enough storage to store all the non-fixed parameters
    from template and params. */
void LALInferenceKDVariablesToREAL8(LALInferenceVariables *params, REAL8 *pt, LALInferenceVariables *template);

/** Fills in the non-fixed variables in params from the given REAL8
    array.  The ordering of variables is given by the order of the
    non-fixed variables in template. */
void LALInferenceKDREAL8ToVariables(LALInferenceVariables *params, REAL8 *pt, LALInferenceVariables *template);

#endif

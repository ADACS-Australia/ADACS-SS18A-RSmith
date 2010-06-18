/* 
 *  LALInference.c:  Bayesian Followup functions
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

#include <stdio.h>
#include <stdlib.h>
#include "LALInference.h"
#include <lal/DetResponse.h>
#include <lal/TimeDelay.h>
#include <lal/TimeSeries.h>
#include <lal/FrequencySeries.h>
#include <lal/Units.h>
#include <lal/TimeFreqFFT.h>
#include <lal/VectorOps.h>
#include <lal/Date.h>

size_t typeSize[] = {sizeof(INT4), 
                     sizeof(INT8), 
                     sizeof(REAL4), 
                     sizeof(REAL8), 
                     sizeof(COMPLEX8), 
                     sizeof(COMPLEX16), 
                     sizeof(gsl_matrix *)};


void die(char *message)
{
  fprintf(stderr, message);
  exit(1);
}



/* ============ Accessor functions for the Variable structure: ========== */



LALVariableItem *getItem(LALVariables *vars,const char *name)
/* (this function is only to be used internally) */
/* Returns pointer to item for given item name.  */
{
  LALVariableItem *this = vars->head;
  while (this != NULL) { 
    if (!strcmp(this->name,name)) break;
    else this = this->next;
  }
  return(this);
}


LALVariableItem *getItemNr(LALVariables *vars, int index)
/* (this function is only to be used internally)  */
/* Returns pointer to item for given item number. */
{
  int i=1;
  if (index < i) die(" Error in getItemNr(): requesting zero or negative index entry.\n");
  LALVariableItem *this=vars->head;
  while (this != NULL) { 
    if (i == index) break;
    else {
      this = this->next;
      ++i;
    }
  }
  return(this);
}


void *getVariable(LALVariables * vars,const char * name)
/* Return the value of variable name from the vars structure by walking the list */
{
  LALVariableItem *item;
  item=getItem(vars,name);
  if(!item) {
    fprintf(stderr, " ERROR in getVariable(): entry \"%s\" not found.\n", name);
    exit(1);
  }
  return(item->value);
}


INT4 getVariableDimension(LALVariables *vars)
{
  return(vars->dimension);
}


VariableType getVariableType(LALVariables *vars, int index)
/* Returns type of the i-th entry, */
/* where  1 <= index <= dimension. */
{
  LALVariableItem *item;
  if ((index < 1) | (index > vars->dimension)){
    fprintf(stderr, " ERROR in getVariableName(...,index=%d): index needs to be 1 <= index <= dimension = %d.\n", 
            index, vars->dimension);
    exit(1);
  }
  item = getItemNr(vars, index);
  return(item->type);
}


char *getVariableName(LALVariables *vars, int index)
/* Returns (pointer to) the name of the i-th entry, */
/* where  1 <= index <= dimension.                  */
{
  LALVariableItem *item;
  if ((index < 1) | (index > vars->dimension)){
    fprintf(stderr, " ERROR in getVariableName(...,index=%d): index needs to be 1 <= index <= dimension = %d.\n", 
            index, vars->dimension);
    exit(1);
  }
  item = getItemNr(vars, index);
  return(item->name);
}


void setVariable(LALVariables * vars, const char * name, void *value)
/* Set the value of variable name in the vars structure to value */
{
  LALVariableItem *item;
  item=getItem(vars,name);
  if(!item) {
    fprintf(stderr, " ERROR in setVariable(): entry \"%s\" not found.\n", name);
    exit(1);
  }
  memcpy(item->value,value,typeSize[item->type]);
  return;
}



void addVariable(LALVariables * vars, const char * name, void *value, VariableType type)
/* Add the variable name with type type and value value to vars */
{
  /* Check the name doesn't already exist */
  /* *** If variable already exists, should we just set it?*/
  if(checkVariable(vars,name)) {fprintf(stderr," ERROR in addVariable(): Cannot re-add \"%s\".\n",name); exit(1);}

  LALVariableItem *new=malloc(sizeof(LALVariableItem));
  memset(new,0,sizeof(LALVariableItem));
  new->value = (void *)malloc(typeSize[type]);
  if(new==NULL||new->value==NULL) die(" ERROR in addVariable(): unable to allocate memory for list item.\n");
  memcpy(new->name,name,VARNAME_MAX);
  new->type = type;
  memcpy(new->value,value,typeSize[type]);
  new->next = vars->head;
  vars->head = new;
  vars->dimension++;
  return;
}



void removeVariable(LALVariables *vars,const char *name)
{
  LALVariableItem *this=vars->head;
  LALVariableItem *parent=NULL;
  while(this){
    if(!strcmp(this->name,name)) break;
    else {parent=this; this=this->next;}
  }
  if(!this) {fprintf(stderr," WARNING in removeVariable(): entry \"%s\" not found.\n",name); return;}
  if(!parent) vars->head=this->next;
  else parent->next=this->next;
  free(this->value);
  free(this);
  vars->dimension--;
  return;
}



int checkVariable(LALVariables *vars,const char *name)
/* Check for existance of name */
{
  if(getItem(vars,name)) return 1;
  else return 0;
}



void destroyVariables(LALVariables *vars)
/* Free the entire structure */
{
  LALVariableItem *this,*next;
  if(!vars) return;
  this=vars->head;
  if(this) next=this->next;
  while(this){
    free(this->value);
    free(this);
    this=next;
    if(this) next=this->next;
  }
  vars->head=NULL;
  vars->dimension=0;
  return;
}



void copyVariables(LALVariables *origin, LALVariables *target)
/*  copy contents of "origin" over to "target"  */
{
  LALVariableItem *ptr;
  /* first dispose contents of "target" (if any): */
  destroyVariables(target);
  /* then copy over elements of "origin": */
  ptr = origin->head;
  while (ptr != NULL) {
    addVariable(target, ptr->name, ptr->value, ptr->type);
    ptr = ptr->next;
  }
  return;
}



void printVariables(LALVariables *var)
/* output contents of a 'LALVariables' structure       */
/* (by now only prints names and types, but no values) */
{
  LALVariableItem *ptr = var->head;
  fprintf(stdout, "LALVariables:\n");
  if (ptr==NULL) fprintf(stdout, "  <empty>\n");
  else {
    /* loop over entries: */
    while (ptr != NULL) {
      /* print name: */
      fprintf(stdout, "  \"%s\"", ptr->name); 
      /* print type: */
      fprintf(stdout, "  (type #%d, ", ((int) ptr->type));
      switch (ptr->type) {
        case INT4_t:
          fprintf(stdout, "'INT4'");
          break;
        case INT8_t:
          fprintf(stdout, "'INT8'");
          break;
        case REAL4_t:
          fprintf(stdout, "'REAL4'");
          break;
        case REAL8_t:
          fprintf(stdout, "'REAL8'");
          break;
        case COMPLEX8_t:
          fprintf(stdout, "'COMPLEX8'");
          break;
        case COMPLEX16_t:
          fprintf(stdout, "'COMPLEX16'");
          break;
        case gslMatrix_t:
          fprintf(stdout, "'gslMatrix'");
          break;
        default:
          fprintf(stdout, "<unknown type>");
      }
      fprintf(stdout, ")  ");
      /* print value: */
      switch (ptr->type) {
        case INT4_t:
          fprintf(stdout, "%d", *(INT4 *) ptr->value);
          break;
        case INT8_t:
          fprintf(stdout, "%lld", *(INT8 *) ptr->value);
          break;
        case REAL4_t:
          fprintf(stdout, "%e", *(REAL4 *) ptr->value);
          break;
        case REAL8_t:
          fprintf(stdout, "%e", *(REAL8 *) ptr->value);
          break;
        case COMPLEX8_t:
          fprintf(stdout, "%e + i*%e", 
                  (REAL4) ((COMPLEX8 *) ptr->value)->re, (REAL4) ((COMPLEX8 *) ptr->value)->im);
          break;
        case COMPLEX16_t:
          fprintf(stdout, "%e + i*%e", 
                  (REAL8) ((COMPLEX16 *) ptr->value)->re, (REAL8) ((COMPLEX16 *) ptr->value)->im);
          break;
        case gslMatrix_t:
          fprintf(stdout, "<can't print matrix>");          
          break;
        default:
          fprintf(stdout, "<can't print>");          
      }
      fprintf(stdout, "\n");
      ptr = ptr->next;
    }  
  }
  return;
}



int compareVariables(LALVariables *var1, LALVariables *var2)
/*  Compare contents of "var1" and "var2".                       */
/*  Returns zero for equal entries, and one if difference found. */
/*  Make sure to only call this function when all entries are    */
/*  actually comparable. For example, "gslMatrix" type entries   */
/*  cannot (yet?) be checked for equality.                       */
{
  int result = 0;
  LALVariableItem *ptr1 = var1->head;
  LALVariableItem *ptr2 = NULL;
  if (var1->dimension != var2->dimension) result = 1;  // differing dimension
  while ((ptr1 != NULL) & (result == 0)) {
    ptr2 = getItem(var2, ptr1->name);
    if (ptr2 != NULL) {  // corrsesponding entry exists; now compare type, then value:
      if (ptr2->type == ptr1->type) {  // entry type identical
        switch (ptr1->type) {  // do value comparison depending on type:
          case INT4_t: 
            result = ((*(INT4 *) ptr2->value) != (*(INT4 *) ptr1->value));
            break;
          case INT8_t: 
            result = ((*(INT8 *) ptr2->value) != (*(INT8 *) ptr1->value));
            break;
          case REAL4_t: 
            result = ((*(REAL4 *) ptr2->value) != (*(REAL4 *) ptr1->value));
            break;
          case REAL8_t:
            result = ((*(REAL8 *) ptr2->value) != (*(REAL8 *) ptr1->value));
            break;
          case COMPLEX8_t: 
            result = (((REAL4) ((COMPLEX8 *) ptr2->value)->re != (REAL4) ((COMPLEX8 *) ptr1->value)->re)
                      || ((REAL4) ((COMPLEX8 *) ptr2->value)->im != (REAL4) ((COMPLEX8 *) ptr1->value)->im));
            break;
          case COMPLEX16_t: 
            result = (((REAL8) ((COMPLEX16 *) ptr2->value)->re != (REAL8) ((COMPLEX16 *) ptr1->value)->re)
                      || ((REAL8) ((COMPLEX16 *) ptr2->value)->im != (REAL8) ((COMPLEX16 *) ptr1->value)->im));
            break;
          case gslMatrix_t: 
            fprintf(stderr, " WARNING: compareVariables() cannot yet compare \"gslMatrix\" type entries.\n");
            fprintf(stderr, "          (entry: \"%s\").\n", ptr1->name);
            fprintf(stderr, "          For now entries are by default assumed different.\n");
            result = 1;
            break;
          default:
            fprintf(stderr, " ERROR: encountered unknown LALVariables type in compareVariables()\n");
            fprintf(stderr, "        (entry: \"%s\").\n", ptr1->name);
            exit(1);
        }
      }
      else result = 1;  // same name but differing type
    }
    else result = 1;  // entry of given name doesn't exist in var2
    ptr1 = ptr1->next;
  }
  return(result);
}



/* ============ Command line parsing functions etc.: ========== */



ProcessParamsTable *getProcParamVal(ProcessParamsTable *procparams,const char *name)
/* Returns pointer to element "name" of the ProcessParamsTable, */
/* if present, and NULL otherwise.                              */
{
  ProcessParamsTable *this=procparams;
  while (this!=NULL) { 
    if (!strcmp(this->param, name)) break;
    else this=this->next;
  }
  return(this);
}



void parseCharacterOptionString(char *input, char **strings[], int *n)
/* parses a character string (passed as one of the options) and decomposes   */
/* it into individual parameter character strings. Input is of the form      */
/*   input   :  "[one,two,three]"                                            */
/* and the resulting output is                                               */
/*   strings :  {"one", "two", "three"}                                      */
/* length of parameter names is for now limited to 512 characters.           */
/* (should 'theoretically' (untested) be able to digest white space as well. */
/* Irrelevant for command line options, though.)                             */
{
  int i,j,k,l;
  /* perform a very basic well-formedness-check and count number of parameters: */
  i=0; j=0;
  *n = 0;
  while (input[i] != '\0') {
    if ((j==0) & (input[i]=='[')) j=1;
    if ((j==1) & (input[i]==',')) ++*n;
    if ((j==1) & (input[i]==']')) {++*n; j=2;}
    ++i;
  }
  if (j!=2) fprintf(stderr, " ERROR: argument vector \"%s\" not well-formed!\n", input);
  /* now allocate memory for results: */
  *strings  = (char**)  malloc(sizeof(char*) * (*n));
  for (i=0; i<(*n); ++i) (*strings)[i] = (char*) malloc(sizeof(char)*512);
  i=0; j=0; 
  k=0; /* string counter    */
  l=0; /* character counter */
  while ((input[i] != '\0') & (j<3)) {
    /* state transitions: */
    if ((j==0) & ((input[i]!='[') & (input[i]!=' '))) j=1;
    if (((j==1)|(j==2)) & (input[i]==',')) {(*strings)[k][l]='\0'; j=2; ++k; l=0;}
    if ((j==1) & (input[i]==' ')) j=2;
    if ((j==1) & (input[i]==']')) {(*strings)[k][l]='\0'; j=3;}
    if ((j==2) & (input[i]==']')) {(*strings)[k][l]='\0'; j=3;}
    if ((j==2) & ((input[i]!=']') & (input[i]!=',') & (input[i]!=' '))) j=1;
    /* actual copying: */
    if (j==1) {
      if (l>=511) {
        fprintf(stderr, " WARNING: character argument too long!\n");
        fprintf(stderr, " \"%s\"\n",(*strings)[k]);
      }
      else {
        (*strings)[k][l] = input[i];
        ++l;
      }
    }
    ++i;
  } 
}



ProcessParamsTable *parseCommandLine(int argc, char *argv[])
/* parse command line and set up & fill in 'ProcessParamsTable' linked list.          */
/* If no command line arguments are supplied, the 'ProcessParamsTable' still contains */
/* one empty entry.                                                                   */
{
  int i, state=1;
  int dbldash;
  ProcessParamsTable *head, *ptr=NULL;
  /* always (even for argc==1, i.e. no arguments) put one element in list: */
  head = (ProcessParamsTable*) calloc(1, sizeof(ProcessParamsTable));
  strcpy(head->program, argv[0]);
  ptr = head;
  i=1;
  while ((i<argc) & (state<=3)) {
    /* check for a double-dash at beginning of argument #i: */
    dbldash = ((argv[i][0]=='-') && (argv[i][1]=='-'));
    /* react depending on current state: */
    if (state==1){ /* ('state 1' means handling very 1st argument) */
      if (dbldash) {
        strcpy(head->param, argv[i]);
        strcpy(ptr->type, "string");
        state = 2;
      }
      else { /* (very 1st argument needs to start with "--...") */
        fprintf(stderr, " WARNING: orphaned first command line argument \"%s\" in parseCommandLine().\n", argv[i]);
        state = 4;
      }
    } 
    else if (state==2) { /* ('state 2' means last entry was a parameter starting with "--") */
      if (dbldash) {
        ptr->next = (ProcessParamsTable*) calloc(1, sizeof(ProcessParamsTable));
        ptr = ptr->next;
        strcpy(ptr->program, argv[0]);
        strcpy(ptr->param, argv[i]);
        strcpy(ptr->type, "string");
      }
      else {
        state = 3;
        strcpy(ptr->value, argv[i]);          
      }
    }
    else if (state==3) { /* ('state 3' means last entry was a value) */
      if (dbldash) {
        ptr->next = (ProcessParamsTable*) calloc(1, sizeof(ProcessParamsTable));
        ptr = ptr->next;
        strcpy(ptr->program, argv[0]);
        strcpy(ptr->param, argv[i]);
        strcpy(ptr->type, "string");
        state = 2;
      }
      else {
        fprintf(stderr, " WARNING: orphaned command line argument \"%s\" in parseCommandLine().\n", argv[i]);
        state = 4;
      }     
    }
    ++i;
  }
  if (state==4) die(" ERROR in parseCommandLine(): failed parsing command line options.\n");
  return(head);
}



/* ============ Likelihood computations: ========== */



REAL8 UndecomposedFreqDomainLogLikelihood(LALVariables *currentParams, LALIFOData * data, 
                              LALTemplateFunction *template)
/***************************************************************/
/* (log-) likelihood function.                                 */
/* Returns the non-normalised logarithmic likelihood.          */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* Required (`currentParams') parameters are:                  */
/*   - "rightascension"  (REAL8, radian, 0 <= RA <= 2pi)       */
/*   - "declination"     (REAL8, radian, -pi/2 <= dec <=pi/2)  */
/*   - "polarisation"    (REAL8, radian, 0 <= psi <= ?)        */
/*   - "distance"        (REAL8, Mpc, >0)                      */
/*   - "time"            (REAL8, GPS sec.)                     */
/***************************************************************/
{
  double Fplus, Fcross;
  double FplusScaled, FcrossScaled;
  double diffRe, diffIm, diffSquared;
  double dataReal, dataImag;
  REAL8 loglikeli;
  REAL8 plainTemplateReal, plainTemplateImag;
  REAL8 templateReal, templateImag;
  int i, lower, upper;
  LALIFOData *dataPtr;
  double ra, dec, psi, distMpc, gmst;
  double GPSdouble;
  LIGOTimeGPS GPSlal;
  double chisquared;
  double timedelay;  /* time delay b/w iterferometer & geocenter w.r.t. sky location */
  double timeshift;  /* time shift (not necessarily same as above)                   */
  double deltaT, TwoDeltaToverN, deltaF, twopit, f, re, im;
  double timeTmp;
  int different;
  LALStatus status;
  memset(&status,0,sizeof(status));
  LALVariables intrinsicParams;

  /* determine source's sky location & orientation parameters: */
  ra        = *(REAL8*) getVariable(currentParams, "rightascension"); /* radian      */
  dec       = *(REAL8*) getVariable(currentParams, "declination");    /* radian      */
  psi       = *(REAL8*) getVariable(currentParams, "polarisation");   /* radian      */
  GPSdouble = *(REAL8*) getVariable(currentParams, "time");           /* GPS seconds */
  distMpc   = *(REAL8*) getVariable(currentParams, "distance");       /* Mpc         */

  /* figure out GMST: */
  XLALINT8NSToGPS(&GPSlal, floor(1e9 * GPSdouble + 0.5));
  //UandA.units    = MST_RAD;
  //UandA.accuracy = LALLEAPSEC_LOOSE;
  //LALGPStoGMST1(&status, &gmst, &GPSlal, &UandA);
  gmst=XLALGreenwichMeanSiderealTime(&GPSlal);
  intrinsicParams.head      = NULL;
  intrinsicParams.dimension = 0;
  copyVariables(currentParams, &intrinsicParams);
  removeVariable(&intrinsicParams, "rightascension");
  removeVariable(&intrinsicParams, "declination");
  removeVariable(&intrinsicParams, "polarisation");
  removeVariable(&intrinsicParams, "time");
  removeVariable(&intrinsicParams, "distance");
  // TODO: add pointer to template function here.
  // (otherwise same parameters but different template will lead to no re-computation!!)

  chisquared = 0.0;
  /* loop over data (different interferometers): */
  dataPtr = data;

  while (dataPtr != NULL) {
    /* The parameters the Likelihood function can handle by itself   */
    /* (and which shouldn't affect the template function) are        */
    /* sky location (ra, dec), polarisation and signal arrival time. */
    /* Note that the template function shifts the waveform to so that*/
	/* t_c corresponds to the "time" parameter in                    */
	/* IFOdata->modelParams (set, e.g., from the trigger value).     */
    
    /* Compare parameter values with parameter values corresponding  */
    /* to currently stored template; ignore "time" variable:         */
    if (checkVariable(dataPtr->modelParams, "time")) {
      timeTmp = *(REAL8 *) getVariable(dataPtr->modelParams, "time");
      removeVariable(dataPtr->modelParams, "time");
    }
    else timeTmp = GPSdouble;
    different = compareVariables(dataPtr->modelParams, &intrinsicParams);
    /* "different" now may also mean that "dataPtr->modelParams" */
    /* wasn't allocated yet (as in the very 1st iteration).      */

    if (different) { /* template needs to be re-computed: */
      copyVariables(&intrinsicParams, dataPtr->modelParams);
      addVariable(dataPtr->modelParams, "time", &timeTmp, REAL8_t);
      template(dataPtr);
      if (dataPtr->modelDomain == timeDomain)
        executeFT(dataPtr);
      /* note that the dataPtr->modelParams "time" element may have changed here!! */
      /* (during "template()" computation)                                      */
    }
    else { /* no re-computation necessary. Return back "time" value, do nothing else: */
      addVariable(dataPtr->modelParams, "time", &timeTmp, REAL8_t);
    }

    /*-- Template is now in dataPtr->freqModelhPlus and dataPtr->freqModelhCross. --*/
    /*-- (Either freshly computed or inherited.)                            --*/

    /* determine beam pattern response (F_plus and F_cross) for given Ifo: */
    XLALComputeDetAMResponse(&Fplus, &Fcross,
                             dataPtr->detector->response,
			     ra, dec, psi, gmst);
    /* signal arrival time (relative to geocenter); */
    timedelay = XLALTimeDelayFromEarthCenter(dataPtr->detector->location,
                                             ra, dec, &GPSlal);
    /* (negative timedelay means signal arrives earlier at Ifo than at geocenter, etc.) */

    /* amount by which to time-shift template (not necessarily same as above "timedelay"): */
    timeshift =  (GPSdouble - (*(REAL8*) getVariable(dataPtr->modelParams, "time"))) + timedelay;
    twopit    = LAL_TWOPI * timeshift;

    /* include distance (overall amplitude) effect in Fplus/Fcross: */
    FplusScaled  = Fplus  / distMpc;
    FcrossScaled = Fcross / distMpc;

 //FILE *testout=fopen("test_likeliLAL.txt","w");
 //fprintf(testout, "f PSD dataRe dataIm signalRe signalIm\n");
    /* determine frequency range & loop over frequency bins: */
    deltaT = dataPtr->timeData->deltaT;
    deltaF = 1.0 / (((double)dataPtr->timeData->data->length) * deltaT);
    // printf("deltaF %g, Nt %d, deltaT %g\n", deltaF, dataPtr->timeData->data->length, dataPtr->timeData->deltaT);
    lower = ceil(dataPtr->fLow / deltaF);
    upper = floor(dataPtr->fHigh / deltaF);
    TwoDeltaToverN = 2.0 * deltaT / ((double) dataPtr->timeData->data->length);
    for (i=lower; i<=upper; ++i){
      /* derive template (involving location/orientation parameters) from given plus/cross waveforms: */
      plainTemplateReal = FplusScaled * dataPtr->freqModelhPlus->data->data[i].re  
                          +  FcrossScaled * dataPtr->freqModelhCross->data->data[i].re;
      plainTemplateImag = FplusScaled * dataPtr->freqModelhPlus->data->data[i].im  
                          +  FcrossScaled * dataPtr->freqModelhCross->data->data[i].im;

      /* do time-shifting...             */
      /* (also un-do 1/deltaT scaling): */
      f = ((double) i) * deltaF;
      /* real & imag parts of  exp(-2*pi*i*f*deltaT): */
      re = cos(twopit * f);
      im = - sin(twopit * f);
      templateReal = (plainTemplateReal*re - plainTemplateImag*im) / deltaT;
      templateImag = (plainTemplateReal*im + plainTemplateImag*re) / deltaT;
      dataReal     = dataPtr->freqData->data->data[i].re / deltaT;
      dataImag     = dataPtr->freqData->data->data[i].im / deltaT;
      /* compute squared difference & 'chi-squared': */
      diffRe       = dataReal - templateReal;         // Difference in real parts...
      diffIm       = dataImag - templateImag;         // ...and imaginary parts, and...
      diffSquared  = diffRe*diffRe + diffIm*diffIm ;  // ...squared difference of the 2 complex figures.
      chisquared  += ((TwoDeltaToverN * diffSquared) / dataPtr->oneSidedNoisePowerSpectrum->data->data[i]);
 //fprintf(testout, "%e %e %e %e %e %e\n",
 //        f, dataPtr->oneSidedNoisePowerSpectrum->data->data[i], 
 //        dataPtr->freqData->data->data[i].re, dataPtr->freqData->data->data[i].im,
 //        templateReal, templateImag);
    }
    dataPtr = dataPtr->next;
 //fclose(testout);
  }
  loglikeli = -1.0 * chisquared; // note (again): the log-likelihood is unnormalised!
  destroyVariables(&intrinsicParams);
  return(loglikeli);
}

REAL8 FreqDomainLogLikelihood(LALVariables *currentParams, LALIFOData * data, 
                              LALTemplateFunction *template)
/***************************************************************/
/* (log-) likelihood function.                                 */
/* Returns the non-normalised logarithmic likelihood.          */
/* Slightly slower but cleaner than							   */
/* UndecomposedFreqDomainLogLikelihood().          `		   */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* Required (`currentParams') parameters are:                  */
/*   - "rightascension"  (REAL8, radian, 0 <= RA <= 2pi)       */
/*   - "declination"     (REAL8, radian, -pi/2 <= dec <=pi/2)  */
/*   - "polarisation"    (REAL8, radian, 0 <= psi <= ?)        */
/*   - "distance"        (REAL8, Mpc, >0)                      */
/*   - "time"            (REAL8, GPS sec.)                     */
/***************************************************************/
{
  REAL8 loglikeli, totalChiSquared=0.0;
  LALIFOData *ifoPtr=data;
  COMPLEX16Vector *freqModelResponse=NULL;

  /* loop over data (different interferometers): */
  while (ifoPtr != NULL) {
	if(freqModelResponse==NULL)
		freqModelResponse= XLALCreateCOMPLEX16Vector(ifoPtr->freqData->data->length);
	else
		freqModelResponse= XLALResizeCOMPLEX16Vector(freqModelResponse, ifoPtr->freqData->data->length);
	/*compute the response*/
	ComputeFreqDomainResponse(currentParams, ifoPtr, template, freqModelResponse);
	/*if(residual==NULL)
		residual=XLALCreateCOMPLEX16Vector(ifoPtr->freqData->data->length);
	else
		residual=XLALResizeCOMPLEX16Vector(residual, ifoPtr->freqData->data->length);
	
	COMPLEX16VectorSubtract(residual, ifoPtr->freqData->data, freqModelResponse);
	totalChiSquared+=ComputeFrequencyDomainOverlap(ifoPtr, residual, residual); 
	*/

	totalChiSquared+=ComputeFrequencyDomainOverlap(ifoPtr, ifoPtr->freqData->data, ifoPtr->freqData->data)
		-2.0*ComputeFrequencyDomainOverlap(ifoPtr, ifoPtr->freqData->data, freqModelResponse)
		+ComputeFrequencyDomainOverlap(ifoPtr, freqModelResponse, freqModelResponse);

    ifoPtr = ifoPtr->next;
  }
  loglikeli = -0.5 * totalChiSquared; // note (again): the log-likelihood is unnormalised!
  XLALDestroyCOMPLEX16Vector(freqModelResponse);
  return(loglikeli);
}

void ComputeFreqDomainResponse(LALVariables *currentParams, LALIFOData * dataPtr, 
                              LALTemplateFunction *template, COMPLEX16Vector *freqWaveform)
/***************************************************************/
/* Frequency-domain single-IFO response computation.           */
/* Computes response for a given template.                     */
/* Will re-compute template only if necessary                  */
/* (i.e., if previous, as stored in data->freqModelhCross,     */
/* was based on different parameters or template function).    */
/* Carries out timeshifting for a given detector               */
/* and projection onto this detector.                          */
/* Result stored in freqResponse, assumed to be correctly      */
/* initialized												   */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* Required (`currentParams') parameters are:                  */
/*   - "rightascension"  (REAL8, radian, 0 <= RA <= 2pi)       */
/*   - "declination"     (REAL8, radian, -pi/2 <= dec <=pi/2)  */
/*   - "polarisation"    (REAL8, radian, 0 <= psi <= ?)        */
/*   - "distance"        (REAL8, Mpc, >0)                      */
/*   - "time"            (REAL8, GPS sec.)                     */
/***************************************************************/							  
{
	double ra, dec, psi, distMpc, gmst;
	
	double GPSdouble;
	double timeTmp;
	LIGOTimeGPS GPSlal;
	double timedelay;  /* time delay b/w iterferometer & geocenter w.r.t. sky location */
	double timeshift;  /* time shift (not necessarily same as above)                   */
	double deltaT, deltaF, twopit, f, re, im;

	int different;
	LALVariables intrinsicParams;
	LALStatus status;
	memset(&status,0,sizeof(status));

	double Fplus, Fcross;
	double FplusScaled, FcrossScaled;
	REAL8 plainTemplateReal, plainTemplateImag;
	int i;

	/* determine source's sky location & orientation parameters: */
	ra        = *(REAL8*) getVariable(currentParams, "rightascension"); /* radian      */
	dec       = *(REAL8*) getVariable(currentParams, "declination");    /* radian      */
	psi       = *(REAL8*) getVariable(currentParams, "polarisation");   /* radian      */
	GPSdouble = *(REAL8*) getVariable(currentParams, "time");           /* GPS seconds */
	distMpc   = *(REAL8*) getVariable(currentParams, "distance");       /* Mpc         */
		
	/* figure out GMST: */
	XLALINT8NSToGPS(&GPSlal, floor(1e9 * GPSdouble + 0.5));
	//UandA.units    = MST_RAD;
	//UandA.accuracy = LALLEAPSEC_LOOSE;
	//LALGPStoGMST1(&status, &gmst, &GPSlal, &UandA);
	gmst=XLALGreenwichMeanSiderealTime(&GPSlal);
	intrinsicParams.head      = NULL;
	intrinsicParams.dimension = 0;
	copyVariables(currentParams, &intrinsicParams);
	removeVariable(&intrinsicParams, "rightascension");
	removeVariable(&intrinsicParams, "declination");
	removeVariable(&intrinsicParams, "polarisation");
	removeVariable(&intrinsicParams, "time");
	removeVariable(&intrinsicParams, "distance");
	// TODO: add pointer to template function here.
	// (otherwise same parameters but different template will lead to no re-computation!!)
      
	/* The parameters the response function can handle by itself     */
    /* (and which shouldn't affect the template function) are        */
    /* sky location (ra, dec), polarisation and signal arrival time. */
    /* Note that the template function shifts the waveform to so that*/
	/* t_c corresponds to the "time" parameter in                    */
	/* IFOdata->modelParams (set, e.g., from the trigger value).     */
    
    /* Compare parameter values with parameter values corresponding  */
    /* to currently stored template; ignore "time" variable:         */
    if (checkVariable(dataPtr->modelParams, "time")) {
      timeTmp = *(REAL8 *) getVariable(dataPtr->modelParams, "time");
      removeVariable(dataPtr->modelParams, "time");
    }
    else timeTmp = GPSdouble;
    different = compareVariables(dataPtr->modelParams, &intrinsicParams);
    /* "different" now may also mean that "dataPtr->modelParams" */
    /* wasn't allocated yet (as in the very 1st iteration).      */

    if (different) { /* template needs to be re-computed: */
      copyVariables(&intrinsicParams, dataPtr->modelParams);
      addVariable(dataPtr->modelParams, "time", &timeTmp, REAL8_t);
      template(dataPtr);
      if (dataPtr->modelDomain == timeDomain)
        executeFT(dataPtr);
      /* note that the dataPtr->modelParams "time" element may have changed here!! */
      /* (during "template()" computation)                                      */
    }
    else { /* no re-computation necessary. Return back "time" value, do nothing else: */
      addVariable(dataPtr->modelParams, "time", &timeTmp, REAL8_t);
    }

    /*-- Template is now in dataPtr->freqModelhPlus and dataPtr->freqModelhCross. --*/
    /*-- (Either freshly computed or inherited.)                            --*/

    /* determine beam pattern response (F_plus and F_cross) for given Ifo: */
    XLALComputeDetAMResponse(&Fplus, &Fcross, dataPtr->detector->response,
			     ra, dec, psi, gmst);
		 
    /* signal arrival time (relative to geocenter); */
    timedelay = XLALTimeDelayFromEarthCenter(dataPtr->detector->location,
                                             ra, dec, &GPSlal);
    /* (negative timedelay means signal arrives earlier at Ifo than at geocenter, etc.) */

    /* amount by which to time-shift template (not necessarily same as above "timedelay"): */
    timeshift =  (GPSdouble - (*(REAL8*) getVariable(dataPtr->modelParams, "time"))) + timedelay;
    twopit    = LAL_TWOPI * timeshift;


    /* include distance (overall amplitude) effect in Fplus/Fcross: */
    FplusScaled  = Fplus  / distMpc;
    FcrossScaled = Fcross / distMpc;


	if(freqWaveform->length!=dataPtr->freqModelhPlus->data->length){
		printf("fW%d data%d\n", freqWaveform->length, dataPtr->freqModelhPlus->data->length);
		printf("Error!  Frequency data vector must be same length as original data!\n");
		exit(1);
	}
	
	deltaT = dataPtr->timeData->deltaT;
    deltaF = 1.0 / (((double)dataPtr->timeData->data->length) * deltaT);

FILE* file=fopen("TempSignal.dat", "w");	
	for(i=0; i<freqWaveform->length; i++){
		/* derive template (involving location/orientation parameters) from given plus/cross waveforms: */
		plainTemplateReal = FplusScaled * dataPtr->freqModelhPlus->data->data[i].re  
                          +  FcrossScaled * dataPtr->freqModelhCross->data->data[i].re;
		plainTemplateImag = FplusScaled * dataPtr->freqModelhPlus->data->data[i].im  
                          +  FcrossScaled * dataPtr->freqModelhCross->data->data[i].im;

		/* do time-shifting...             */
		/* (also un-do 1/deltaT scaling): */
		f = ((double) i) * deltaF;
		/* real & imag parts of  exp(-2*pi*i*f*deltaT): */
		re = cos(twopit * f);
		im = - sin(twopit * f);

		freqWaveform->data[i].re= (plainTemplateReal*re - plainTemplateImag*im);
		freqWaveform->data[i].im= (plainTemplateReal*im + plainTemplateImag*re);		
fprintf(file, "%lg %lg \t %lg\n", f, freqWaveform->data[i].re, freqWaveform->data[i].im);
	}
fclose(file);
	destroyVariables(&intrinsicParams);
}

	
							  						  
REAL8 ComputeFrequencyDomainOverlap(LALIFOData * dataPtr,
	//gsl_vector * freqData1, gsl_vector * freqData2
	COMPLEX16Vector * freqData1, COMPLEX16Vector * freqData2)
{
    int lower, upper, i;
	double deltaT, deltaF;

	double overlap=0.0;
	
	/* determine frequency range & loop over frequency bins: */
    deltaT = dataPtr->timeData->deltaT;
    deltaF = 1.0 / (((double)dataPtr->timeData->data->length) * deltaT);
    // printf("deltaF %g, Nt %d, deltaT %g\n", deltaF, dataPtr->timeData->data->length, dataPtr->timeData->deltaT);
    lower = ceil(dataPtr->fLow / deltaF);
    upper = floor(dataPtr->fHigh / deltaF);
	
	//for(i=1; i<=1; i++){
//fprintf(stdout, "freqData1->data[1].re %lg, freqData1->data[1].im %lg, noise[1] %lg\n", 
//freqData1->data[1].re, freqData1->data[1].im, dataPtr->oneSidedNoisePowerSpectrum->data->data[1]);
    for (i=lower; i<=upper; ++i){  	  	  
      /* compute squared difference & 'chi-squared': */
      //diffRe       = data1re - data2re;         // Difference in real parts...
      //diffIm       = data1im - data2im;         // ...and imaginary parts, and...
      //diffSquared  = diffRe*diffRe + diffIm*diffIm ;  // ...squared difference of the 2 complex figures.
	  overlap  += ((4.0*deltaF*(freqData1->data[i].re*freqData2->data[i].re+freqData1->data[i].im*freqData2->data[i].im)) 
		/ dataPtr->oneSidedNoisePowerSpectrum->data->data[i]);
	}
//fprintf(stdout, "Overlap %lg, lower %d upper %d\n", overlap, lower, upper);
	return overlap;
}


REAL8 NullLogLikelihood(LALIFOData *data)
/*Idential to FreqDomainNullLogLikelihood                        */
{
  REAL8 loglikeli, totalChiSquared=0.0;
  LALIFOData *ifoPtr=data;

  /* loop over data (different interferometers): */
  while (ifoPtr != NULL) {
	totalChiSquared+=ComputeFrequencyDomainOverlap(ifoPtr, ifoPtr->freqData->data, ifoPtr->freqData->data);
    ifoPtr = ifoPtr->next;
  }
  loglikeli = -0.5 * totalChiSquared; // note (again): the log-likelihood is unnormalised!
  return(loglikeli);
}

REAL8 FreqDomainNullLogLikelihood(LALIFOData *data)
/* calls the `FreqDomainLogLikelihood()' function in conjunction   */
/* with the `templateNullFreqdomain()' template in order to return */
/* the "Null likelihood" without having to bother specifying       */
/* parameters or template while ensuring computations are exactly  */
/* the same as in usual likelihood calculations.                   */
{
  LALVariables dummyParams;
  double dummyValue;
  double loglikeli;
  /* set some (basically arbitrary) dummy values for intrinsic parameters */
  /* (these shouldn't make a difference, but need to be present):         */
  dummyParams.head      = NULL;
  dummyParams.dimension = 0;
  dummyValue = 0.5;
  addVariable(&dummyParams, "rightascension", &dummyValue, REAL8_t);
  addVariable(&dummyParams, "declination",    &dummyValue, REAL8_t);
  addVariable(&dummyParams, "polarisation",   &dummyValue, REAL8_t);
  addVariable(&dummyParams, "distance",       &dummyValue, REAL8_t);
  dummyValue = XLALGPSGetREAL8(&data->timeData->epoch) 
               + (((double) data->timeData->data->length) / 2.0) * data->timeData->deltaT;
  addVariable(&dummyParams, "time",           &dummyValue, REAL8_t);
  loglikeli = FreqDomainLogLikelihood(&dummyParams, data, &templateNullFreqdomain);
  destroyVariables(&dummyParams);
  return(loglikeli);
}


void dumptemplateFreqDomain(LALVariables *currentParams, LALIFOData * data, 
                            LALTemplateFunction *template, char *filename)
/* de-bugging function writing (frequency-domain) template to a CSV file */
/* File contains real & imaginary parts of plus & cross components.      */
/* Template amplitude is scaled to 1Mpc distance.                        */
{
  FILE *outfile=NULL; 
  LALIFOData *dataPtr;
  double deltaT, deltaF, f;
  int i;

  copyVariables(currentParams, data->modelParams);
  dataPtr = data;
  while (dataPtr != NULL) { /* this loop actually does nothing (yet) here. */
    template(data);
    if (data->modelDomain == timeDomain)
      executeFT(data);

    outfile = fopen(filename, "w");
    /*fprintf(outfile, "f PSD dataRe dataIm signalPlusRe signalPlusIm signalCrossRe signalCrossIm\n");*/
    fprintf(outfile, "\"f\",\"PSD\",\"signalPlusRe\",\"signalPlusIm\",\"signalCrossRe\",\"signalCrossIm\"\n");
    deltaT = dataPtr->timeData->deltaT;
    deltaF = 1.0 / (((double)dataPtr->timeData->data->length) * deltaT);
    for (i=0; i<data->freqModelhPlus->data->length; ++i){
      f = ((double) i) * deltaF;
      fprintf(outfile, "%f,%e,%e,%e,%e,%e\n",
              f, data->oneSidedNoisePowerSpectrum->data->data[i],
              /*data->freqData->data->data[i].re, data->freqData->data->data[i].im,*/
              data->freqModelhPlus->data->data[i].re,
              data->freqModelhPlus->data->data[i].im,
              data->freqModelhCross->data->data[i].re,
              data->freqModelhCross->data->data[i].im);
    }
    fclose(outfile);
    dataPtr = NULL;
  }
  fprintf(stdout, " wrote (frequency-domain) template to CSV file \"%s\".\n", filename);
}


void dumptemplateTimeDomain(LALVariables *currentParams, LALIFOData * data, 
                            LALTemplateFunction *template, char *filename)
/* de-bugging function writing (frequency-domain) template to a CSV file */
/* File contains real & imaginary parts of plus & cross components.      */
/* Template amplitude is scaled to 1Mpc distance.                        */
{
  FILE *outfile=NULL; 
  LALIFOData *dataPtr;
  double deltaT, deltaF, t, epoch;
  int i;

  copyVariables(currentParams, data->modelParams);
  dataPtr = data;
  while (dataPtr != NULL) { /* this loop actually does nothing (yet) here. */
    template(data);
    if (data->modelDomain == frequencyDomain)
      executeInvFT(data);

    outfile = fopen(filename, "w");
    fprintf(outfile, "\"t\",\"signalPlus\",\"signalCross\"\n");
    deltaT = dataPtr->timeData->deltaT;
    deltaF = 1.0 / (((double)dataPtr->timeData->data->length) * deltaT);
    epoch = XLALGPSGetREAL8(&data->timeData->epoch);
    for (i=0; i<data->timeModelhPlus->data->length; ++i){
      t =  epoch + ((double) i) * deltaT;
      fprintf(outfile, "%f,%e,%e\n",
              t,
              data->timeModelhPlus->data->data[i],
              data->timeModelhCross->data->data[i]);
    }
    fclose(outfile);
    dataPtr = NULL;
  }
  fprintf(stdout, " wrote (time-domain) template to CSV file \"%s\".\n", filename);
}



void executeFT(LALIFOData *IFOdata)
/* Execute (forward, time-to-freq) Fourier transform.         */
/* Contents of IFOdata->timeModelh... are windowed and FT'ed, */
/* results go into IFOdata->freqModelh...                     */
/*  CHECK: keep or drop normalisation step here ?!?  */
{
  int i;
  double norm;
  for(;IFOdata;IFOdata=IFOdata->next){
    /* h+ */
    if(!IFOdata->freqModelhPlus)
      IFOdata->freqModelhPlus=(COMPLEX16FrequencySeries *)XLALCreateCOMPLEX16FrequencySeries("freqData",&(IFOdata->timeData->epoch),0.0,IFOdata->freqData->deltaF,&lalDimensionlessUnit,IFOdata->freqData->data->length);
    XLALDDVectorMultiply(IFOdata->timeModelhPlus->data,IFOdata->timeModelhPlus->data,IFOdata->window->data);
    XLALREAL8TimeFreqFFT(IFOdata->freqModelhPlus,IFOdata->timeModelhPlus,IFOdata->timeToFreqFFTPlan);
    /* hx */
    if(!IFOdata->freqModelhCross)
      IFOdata->freqModelhCross=(COMPLEX16FrequencySeries *)XLALCreateCOMPLEX16FrequencySeries("freqData",&(IFOdata->timeData->epoch),0.0,IFOdata->freqData->deltaF,&lalDimensionlessUnit,IFOdata->freqData->data->length);
    XLALDDVectorMultiply(IFOdata->timeModelhCross->data,IFOdata->timeModelhCross->data,IFOdata->window->data);
    XLALREAL8TimeFreqFFT(IFOdata->freqModelhCross,IFOdata->timeModelhCross,IFOdata->timeToFreqFFTPlan);
      norm=sqrt(IFOdata->window->sumofsquares/IFOdata->window->data->length);
	  for(i=0;i<IFOdata->freqModelhPlus->data->length;i++){
		  IFOdata->freqModelhPlus->data->data[i].re*=norm;
		  IFOdata->freqModelhPlus->data->data[i].im*=norm;
		  IFOdata->freqModelhCross->data->data[i].re*=norm;
		  IFOdata->freqModelhCross->data->data[i].im*=norm;
		  }
  }
}



void executeInvFT(LALIFOData *IFOdata)
/* Execute inverse (freq-to-time) Fourier transform. */
/* Results go into 'IFOdata->timeModelh...'          */
{
  while (IFOdata != NULL) {
    if (IFOdata->freqToTimeFFTPlan==NULL) die(" ERROR in executeInvFT(): encountered unallocated 'freqToTimeFFTPlan'.\n");
    /*  h+ :  */
    if (IFOdata->timeModelhPlus==NULL) die(" ERROR in executeInvFT(): encountered unallocated 'timeModelhPlus'.\n");
    if (IFOdata->freqModelhPlus==NULL) die(" ERROR in executeInvFT(): encountered unallocated 'freqModelhPlus'.\n");
    XLALREAL8FreqTimeFFT(IFOdata->timeModelhPlus, IFOdata->freqModelhPlus, IFOdata->freqToTimeFFTPlan);
    /*  hx :  */
    if (IFOdata->timeModelhCross==NULL) die(" ERROR in executeInvFT(): encountered unallocated 'timeModelhCross'.\n");
    if (IFOdata->freqModelhCross==NULL) die(" ERROR in executeInvFT(): encountered unallocated 'freqModelhCross'.\n");
    XLALREAL8FreqTimeFFT(IFOdata->timeModelhCross, IFOdata->freqModelhCross, IFOdata->freqToTimeFFTPlan);
    IFOdata=IFOdata->next;
  }
}



LALInferenceRunState *initialize(ProcessParamsTable *commandLine)
/* calls the "ReadData()" function to gather data & PSD from files, */
/* and initializes other variables accordingly.                     */
{
  LALInferenceRunState *irs=NULL;
  LALIFOData *ifoPtr, *ifoListStart;
  ProcessParamsTable *ppt=NULL;
  unsigned long int randomseed;
  struct timeval tv;
  FILE *devrandom;

  irs = calloc(1, sizeof(LALInferenceRunState));
  /* read data from files: */
  fprintf(stdout, " readData(): started.\n");
  irs->data = readData(commandLine);
  /* (this will already initialise each LALIFOData's following elements:  */
  /*     fLow, fHigh, detector, timeToFreqFFTPlan, freqToTimeFFTPlan,     */
  /*     window, oneSidedNoisePowerSpectrum, timeDate, freqData         ) */
  fprintf(stdout, " readData(): finished.\n");
  if (irs->data != NULL) {
    fprintf(stdout, " initialize(): successfully read data.\n");

    fprintf(stdout, " injectSignal(): started.\n");
    injectSignal(irs->data,commandLine);
    fprintf(stdout, " injectSignal(): finished.\n");

    ifoPtr = irs->data;
	ifoListStart = irs->data;
    while (ifoPtr != NULL) {
		/*If two IFOs have the same sampling rate, they should have the same timeModelh*,
			freqModelh*, and modelParams variables to avoid excess computation 
			in model waveform generation in the future*/
		LALIFOData * ifoPtrCompare=ifoListStart;
		int foundIFOwithSameSampleRate=0;
		while(ifoPtrCompare != NULL && ifoPtrCompare!=ifoPtr) {
			if(ifoPtrCompare->timeData->deltaT == ifoPtr->timeData->deltaT){
				ifoPtr->timeModelhPlus=ifoPtrCompare->timeModelhPlus;
				ifoPtr->freqModelhPlus=ifoPtrCompare->freqModelhPlus;
				ifoPtr->timeModelhCross=ifoPtrCompare->timeModelhCross;				
				ifoPtr->freqModelhCross=ifoPtrCompare->freqModelhCross;				
				ifoPtr->modelParams=ifoPtrCompare->modelParams;	
				foundIFOwithSameSampleRate=1;	
				break;
			}
		}
		if(!foundIFOwithSameSampleRate){
				ifoPtr->timeModelhPlus  = XLALCreateREAL8TimeSeries("timeModelhPlus",
                                                          &(ifoPtr->timeData->epoch),
                                                          0.0,
                                                          ifoPtr->timeData->deltaT,
                                                          &lalDimensionlessUnit,
                                                          ifoPtr->timeData->data->length);
				ifoPtr->timeModelhCross = XLALCreateREAL8TimeSeries("timeModelhCross",
                                                          &(ifoPtr->timeData->epoch),
                                                          0.0,
                                                          ifoPtr->timeData->deltaT,
                                                          &lalDimensionlessUnit,
                                                          ifoPtr->timeData->data->length);
				ifoPtr->freqModelhPlus = XLALCreateCOMPLEX16FrequencySeries("freqModelhPlus",
                                                                  &(ifoPtr->freqData->epoch),
                                                                  0.0,
                                                                  ifoPtr->freqData->deltaF,
                                                                  &lalDimensionlessUnit,
                                                                  ifoPtr->freqData->data->length);
				ifoPtr->freqModelhCross = XLALCreateCOMPLEX16FrequencySeries("freqModelhCross",
                                                                   &(ifoPtr->freqData->epoch),
                                                                   0.0,
                                                                   ifoPtr->freqData->deltaF,
                                                                   &lalDimensionlessUnit,
                                                                   ifoPtr->freqData->data->length);
				ifoPtr->modelParams = calloc(1, sizeof(LALVariables));
		}
		ifoPtr = ifoPtr->next;
    }
    irs->currentLikelihood=NullLogLikelihood(irs->data);
	printf("Injection Null Log Likelihood: %g\n", irs->currentLikelihood);
  }
  else
    fprintf(stdout, " initialize(): no data read.\n");

  /* set up GSL random number generator: */
  gsl_rng_env_setup();
  irs->GSLrandom = gsl_rng_alloc(gsl_rng_mt19937);
  /* (try to) get random seed from command line: */
  ppt = getProcParamVal(commandLine, "--randomseed");
  if (ppt != NULL)
    randomseed = atoi(ppt->value);
  else { /* otherwise generate "random" random seed: */
    if ((devrandom = fopen("/dev/random","r")) == NULL) {
      gettimeofday(&tv, 0);
      randomseed = tv.tv_sec + tv.tv_usec;
    } 
    else {
      fread(&randomseed, sizeof(randomseed), 1, devrandom);
      fclose(devrandom);
    }
  }
  fprintf(stdout, " initialize(): random seed: %lu\n", randomseed);
  gsl_rng_set(irs->GSLrandom, randomseed);

  return(irs);
}


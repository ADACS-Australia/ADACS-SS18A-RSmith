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

#define LAL_USE_OLD_COMPLEX_STRUCTS
#include <stdio.h>
#include <stdlib.h>
#include <lal/LALInference.h>
#include <lal/Units.h>
#include <lal/FrequencySeries.h>
#include <lal/TimeFreqFFT.h>
#include <lal/VectorOps.h>
#include <lal/Date.h>
#include <lal/XLALError.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_eigen.h>

#ifdef __GNUC__
#define UNUSED __attribute__ ((unused))
#else
#define UNUSED
#endif

size_t LALInferenceTypeSize[] = {sizeof(INT4),
                                 sizeof(INT8),
                                 sizeof(UINT4),
                                 sizeof(REAL4),
                                 sizeof(REAL8),
                                 sizeof(COMPLEX8),
                                 sizeof(COMPLEX16),
                                 sizeof(gsl_matrix *),
                                 sizeof(REAL8Vector *),
                                 sizeof(UINT4Vector *),
                                 sizeof(CHAR *),
                                 sizeof(void *)
};


/* ============ Accessor functions for the Variable structure: ========== */

static char *colNameToParamName(const char *colName);


LALInferenceVariableItem *LALInferenceGetItem(const LALInferenceVariables *vars,const char *name)
/* (this function is only to be used internally) */
/* Returns pointer to item for given item name.  */
{
  if(vars==NULL) return NULL;
  LALInferenceVariableItem *this = vars->head;
  while (this != NULL) {
    if (!strcmp(this->name,name)) break;
    else this = this->next;
  }
  return(this);
}


LALInferenceVariableItem *LALInferenceGetItemNr(LALInferenceVariables *vars, int idx)
/* (this function is only to be used internally)  */
/* Returns pointer to item for given item number. */
{
  int i=1;
  if (idx < i) {
    XLAL_ERROR_NULL(XLAL_EINVAL, "Requesting zero or negative idx entry.");
  }
  LALInferenceVariableItem *this=vars->head;
  while (this != NULL) {
    if (i == idx) break;
    else {
      this = this->next;
      ++i;
    }
  }
  return(this);
}

LALInferenceParamVaryType LALInferenceGetVariableVaryType(LALInferenceVariables *vars, const char *name)
{
  return (LALInferenceGetItem(vars,name)->vary);
}


void *LALInferenceGetVariable(const LALInferenceVariables * vars,const char * name)
/* Return the value of variable name from the vars structure by walking the list */
{
  LALInferenceVariableItem *item;
  item=LALInferenceGetItem(vars,name);
  if(!item) {
    XLAL_ERROR_NULL(XLAL_EFAILED, "Entry \"%s\" not found.", name);
  }
  return(item->value);
}


INT4 LALInferenceGetVariableDimension(LALInferenceVariables *vars)
{
  return(vars->dimension);
}


INT4 LALInferenceGetVariableDimensionNonFixed(LALInferenceVariables *vars)
{
  INT4 count=0;
  LALInferenceVariableItem *ptr = vars->head;
  if (ptr==NULL) return count;
  else {
    /* loop over entries: */
    while (ptr != NULL) {
      /* print name: */
      if (ptr->vary != LALINFERENCE_PARAM_FIXED) ++count;
      ptr = ptr->next;
    }
  }
  return count;
}

LALInferenceVariableType LALInferenceGetVariableType(const LALInferenceVariables *vars, const char *name)
{
  return LALInferenceGetItem(vars,name)->type;
}

INT4 LALInferenceGetVariableTypeByIndex(LALInferenceVariables *vars, int idx)
/* Returns type of the i-th entry, */
/* where  1 <= idx <= dimension. */
{
  LALInferenceVariableItem *item;
  if ((idx < 1) || (idx > vars->dimension)){
    XLAL_ERROR(XLAL_EINVAL, "idx = %d, but needs to be 1 <= idx <= dimension = %d.", idx, vars->dimension);
  }
  item = LALInferenceGetItemNr(vars, idx);
  return(item->type);
}


char *LALInferenceGetVariableName(LALInferenceVariables *vars, int idx)
/* Returns (pointer to) the name of the i-th entry, */
/* where  1 <= idx <= dimension.                  */
{
  LALInferenceVariableItem *item;
  if ((idx < 1) || (idx > vars->dimension)){
    XLAL_ERROR_NULL(XLAL_EINVAL, "idx = %d, but needs to be 1 <= idx <= dimension = %d.", idx, vars->dimension);
  }
  item = LALInferenceGetItemNr(vars, idx);
  return(item->name);
}


void LALInferenceSetVariable(LALInferenceVariables * vars, const char * name, void *value)
/* Set the value of variable name in the vars structure to value */
{
  LALInferenceVariableItem *item;
  item=LALInferenceGetItem(vars,name);
  if(!item) {
    XLAL_ERROR_VOID(XLAL_EINVAL, "Entry \"%s\" not found.", name);
  }
  if (item->vary==LALINFERENCE_PARAM_FIXED) return;
  memcpy(item->value,value,LALInferenceTypeSize[item->type]);
  return;
}



void LALInferenceAddVariable(LALInferenceVariables * vars, const char * name, void *value, LALInferenceVariableType type, LALInferenceParamVaryType vary)
/* Add the variable name with type type and value value to vars */
/* If variable already exists, it will over-write the current value if type compatible*/
{
  LALInferenceVariableItem *old=NULL;
  /* Check the name doesn't already exist */
  if(LALInferenceCheckVariable(vars,name)) {
    old=LALInferenceGetItem(vars,name);
    if(old->type != type)
    {
      XLAL_ERROR_VOID(XLAL_EINVAL, "Cannot re-add \"%s\" as previous definition has wrong type.", name);
    }
    LALInferenceSetVariable(vars,name,value);
    return;
  }

  if(!value) {
    XLAL_ERROR_VOID(XLAL_EFAULT, "Unable to access value through null pointer; trying to add \"%s\".", name);
  }

  LALInferenceVariableItem *new=XLALMalloc(sizeof(LALInferenceVariableItem));

  memset(new,0,sizeof(LALInferenceVariableItem));
  if(new) {
    new->value = (void *)XLALMalloc(LALInferenceTypeSize[type]);
  }
  if(new==NULL||new->value==NULL) {
    XLAL_ERROR_VOID(XLAL_ENOMEM, "Unable to allocate memory for list item.");
  }
  memcpy(new->name,name,VARNAME_MAX);
  new->type = type;
  new->vary = vary;
  memcpy(new->value,value,LALInferenceTypeSize[type]);
  new->next = vars->head;
  vars->head = new;
  vars->dimension++;
  return;
}



void LALInferenceRemoveVariable(LALInferenceVariables *vars,const char *name)
{
  LALInferenceVariableItem *this;
  if(!vars)
    XLAL_ERROR_VOID(XLAL_EFAULT);
  this=vars->head;
  LALInferenceVariableItem *parent=NULL;
  while(this){
    if(!strcmp(this->name,name)) break;
    else {parent=this; this=this->next;}
  }
  if(!this){
    XLAL_PRINT_WARNING("Entry \"%s\" not found.", name);
    return;
  }
  if(!parent) vars->head=this->next;
  else parent->next=this->next;
  XLALFree(this->value);
  this->value=NULL;
  XLALFree(this);
  this=NULL;
  vars->dimension--;
  return;
}

int LALInferenceCheckVariableNonFixed(LALInferenceVariables *vars, const char *name)
/* Checks for a writeable variable */
{
  LALInferenceParamVaryType type;
  if(!LALInferenceCheckVariable(vars,name)) return 0;
  type=LALInferenceGetVariableVaryType(vars,name);
  if(type==LALINFERENCE_PARAM_CIRCULAR||type==LALINFERENCE_PARAM_LINEAR) return 1;
  else return 0;

}

int LALInferenceCheckVariable(LALInferenceVariables *vars,const char *name)
/* Check for existance of name */
{
  if(LALInferenceGetItem(vars,name)) return 1;
  else return 0;
}

void LALInferenceDestroyVariables(LALInferenceVariables *vars)
/* Free the entire structure */
{
  LALInferenceVariableItem *this,*next;
  if(!vars) return;
  this=vars->head;
  if(this) next=this->next;
  while(this){
    XLALFree(this->value);
    XLALFree(this);
    this=next;
    if(this) next=this->next;
  }
  vars->head=NULL;
  vars->dimension=0;
  return;
}

void LALInferenceCopyVariables(LALInferenceVariables *origin, LALInferenceVariables *target)
/*  copy contents of "origin" over to "target"  */
{
  int dims = 0, i = 0;

  /* Check that the source and origin differ */
  if(origin==target) return;

  LALInferenceVariableItem *ptr;
  if(!origin)
  {
    XLAL_ERROR_VOID(XLAL_EFAULT, "Unable to access origin pointer.");
  }

  /* Make sure the structure is initialised */
  if(!target) XLAL_ERROR_VOID(XLAL_EFAULT, "Unable to copy to uninitialised LALInferenceVariables structure.");
  /* first dispose contents of "target" (if any): */
  LALInferenceDestroyVariables(target);

  /* get the number of elements in origin */
  dims = LALInferenceGetVariableDimension( origin );

  if ( !dims ){
    XLAL_ERROR_VOID(XLAL_EFAULT, "Origin variables has zero dimensions!");
  }

  /* then copy over elements of "origin" - due to how elements are added by
     LALInferenceAddVariable this has to be done in reverse order to preserve
     the ordering of "origin"  */
  for ( i = dims; i > 0; i-- ){
    ptr = LALInferenceGetItemNr(origin, i);

    if(!ptr)
    {
      XLAL_ERROR_VOID(XLAL_EFAULT, "Bad LALInferenceVariable structure found while trying to copy.");
    }
    else
    {
      if(!ptr->value || !ptr->name){
        XLAL_ERROR_VOID(XLAL_EFAULT, "Badly formed LALInferenceVariableItem structure!");
      }

      LALInferenceAddVariable(target, ptr->name, ptr->value, ptr->type,
                              ptr->vary);
    }
  }

  return;
}

/** Prints a variable item to a string (must be pre-allocated!) */
void LALInferencePrintVariableItem(char *out, LALInferenceVariableItem *ptr)
{
  if(ptr==NULL) {
    XLAL_ERROR_VOID(XLAL_EFAULT, "Null LALInferenceVariableItem pointer.");
  }
  if(out==NULL) {
    XLAL_ERROR_VOID(XLAL_EFAULT, "Null output string pointer.");
  }
  switch (ptr->type) {
        case LALINFERENCE_INT4_t:
          sprintf(out, "%d", *(INT4 *) ptr->value);
          break;
        case LALINFERENCE_INT8_t:
          sprintf(out, "%" LAL_INT8_FORMAT, *(INT8 *) ptr->value);
          break;
        case LALINFERENCE_UINT4_t:
          sprintf(out, "%ud", *(UINT4 *) ptr->value);
          break;
        case LALINFERENCE_REAL4_t:
          sprintf(out, "%.15lf", *(REAL4 *) ptr->value);
          break;
        case LALINFERENCE_REAL8_t:
          sprintf(out, "%.15lf", *(REAL8 *) ptr->value);
          break;
        case LALINFERENCE_COMPLEX8_t:
          sprintf(out, "%e + i*%e",
                 (REAL4) ((COMPLEX8 *) ptr->value)->re, (REAL4) ((COMPLEX8 *) ptr->value)->im);
          break;
        case LALINFERENCE_COMPLEX16_t:
          sprintf(out, "%e + i*%e",
                 (REAL8) ((COMPLEX16 *) ptr->value)->re, (REAL8) ((COMPLEX16 *) ptr->value)->im);
          break;
        case LALINFERENCE_gslMatrix_t:
          sprintf(out, "<can't print matrix>");
          break;
        default:
          sprintf(out, "<can't print>");
      }
  return;
}

void LALInferencePrintVariables(LALInferenceVariables *var)
/** output contents of a 'LALInferenceVariables' structure * /
/ * (by now only prints names and types, but no values) */
{
  LALInferenceVariableItem *ptr = var->head;
  fprintf(stdout, "LALInferenceVariables:\n");
  if (ptr==NULL) fprintf(stdout, "  <empty>\n");
  else {
    /* loop over entries: */
    while (ptr != NULL) {
      /* print name: */
      fprintf(stdout, "  \"%s\"", ptr->name);
      /* print type: */
      fprintf(stdout, "  (type #%d, ", ((int) ptr->type));
      switch (ptr->type) {
        case LALINFERENCE_INT4_t:
          fprintf(stdout, "'INT4'");
          break;
        case LALINFERENCE_INT8_t:
          fprintf(stdout, "'INT8'");
          break;
        case LALINFERENCE_UINT4_t:
          fprintf(stdout, "'UINT4'");
          break;
        case LALINFERENCE_REAL4_t:
          fprintf(stdout, "'REAL4'");
          break;
        case LALINFERENCE_REAL8_t:
          fprintf(stdout, "'REAL8'");
          break;
        case LALINFERENCE_COMPLEX8_t:
          fprintf(stdout, "'COMPLEX8'");
          break;
        case LALINFERENCE_COMPLEX16_t:
          fprintf(stdout, "'COMPLEX16'");
          break;
        case LALINFERENCE_gslMatrix_t:
          fprintf(stdout, "'gslMatrix'");
          break;
        default:
          fprintf(stdout, "<unknown type>");
      }
      fprintf(stdout, ")  ");
      /* print value: */
      switch (ptr->type) {
        case LALINFERENCE_INT4_t:
          fprintf(stdout, "%d", *(INT4 *) ptr->value);
          break;
        case LALINFERENCE_INT8_t:
          fprintf(stdout, "%" LAL_INT8_FORMAT, *(INT8 *) ptr->value);
          break;
        case LALINFERENCE_UINT4_t:
          fprintf(stdout, "%ud", *(UINT4 *) ptr->value);
          break;
        case LALINFERENCE_REAL4_t:
          fprintf(stdout, "%.15lf", *(REAL4 *) ptr->value);
          break;
        case LALINFERENCE_REAL8_t:
          fprintf(stdout, "%.15lf", *(REAL8 *) ptr->value);
          break;
        case LALINFERENCE_COMPLEX8_t:
          fprintf(stdout, "%e + i*%e",
                 (REAL4) ((COMPLEX8 *) ptr->value)->re, (REAL4) ((COMPLEX8 *) ptr->value)->im);
          break;
        case LALINFERENCE_COMPLEX16_t:
          fprintf(stdout, "%e + i*%e",
                 (REAL8) ((COMPLEX16 *) ptr->value)->re, (REAL8) ((COMPLEX16 *) ptr->value)->im);
          break;
        case LALINFERENCE_gslMatrix_t:
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

void LALInferencePrintSample(FILE *fp,LALInferenceVariables *sample){
  if(sample==NULL) return;
  LALInferenceVariableItem *ptr=sample->head;
  if(fp==NULL) return;
  while(ptr!=NULL) {
    switch (ptr->type) {
      case LALINFERENCE_INT4_t:
        fprintf(fp, "%d", *(INT4 *) ptr->value);
        break;
      case LALINFERENCE_INT8_t:
        fprintf(fp, "%"LAL_INT8_FORMAT , *(INT8 *) ptr->value);
        break;
      case LALINFERENCE_UINT4_t:
        fprintf(fp, "%ud", *(UINT4 *) ptr->value);
        break;
      case LALINFERENCE_REAL4_t:
        fprintf(fp, "%9.12e", *(REAL4 *) ptr->value);
        break;
      case LALINFERENCE_REAL8_t:
        fprintf(fp, "%9.12le", *(REAL8 *) ptr->value);
        break;
      case LALINFERENCE_COMPLEX8_t:
        fprintf(fp, "%e + i*%e",
            (REAL4) ((COMPLEX8 *) ptr->value)->re, (REAL4) ((COMPLEX8 *) ptr->value)->im);
        break;
      case LALINFERENCE_COMPLEX16_t:
        fprintf(fp, "%e + i*%e",
            (REAL8) ((COMPLEX16 *) ptr->value)->re, (REAL8) ((COMPLEX16 *) ptr->value)->im);
        break;
      case LALINFERENCE_string_t:
        fprintf(fp, "%s", *((CHAR **)ptr->value));
        break;
      case LALINFERENCE_gslMatrix_t:
        fprintf(stdout, "<can't print matrix>");
        break;
      default:
        fprintf(stdout, "<can't print>");
      }

  fprintf(fp,"\t");
  ptr=ptr->next;
  }
  return;
}

void LALInferencePrintSampleNonFixed(FILE *fp,LALInferenceVariables *sample){

	if(sample==NULL) return;
	LALInferenceVariableItem *ptr=sample->head;
	if(fp==NULL) return;
	while(ptr!=NULL) {
		if (ptr->vary != LALINFERENCE_PARAM_FIXED) {
			switch (ptr->type) {
				case LALINFERENCE_INT4_t:
					fprintf(fp, "%d", *(INT4 *) ptr->value);
					break;
				case LALINFERENCE_INT8_t:
					fprintf(fp, "%"LAL_INT8_FORMAT, *(INT8 *) ptr->value);
					break;
				case LALINFERENCE_UINT4_t:
					fprintf(fp, "%ud", *(UINT4 *) ptr->value);
					break;
				case LALINFERENCE_REAL4_t:
					fprintf(fp, "%11.7f", *(REAL4 *) ptr->value);
					break;
				case LALINFERENCE_REAL8_t:
					fprintf(fp, "%11.7f", *(REAL8 *) ptr->value);
					break;
				case LALINFERENCE_COMPLEX8_t:
					fprintf(fp, "%e + i*%e",
							(REAL4) ((COMPLEX8 *) ptr->value)->re, (REAL4) ((COMPLEX8 *) ptr->value)->im);
					break;
				case LALINFERENCE_COMPLEX16_t:
					fprintf(fp, "%e + i*%e",
							(REAL8) ((COMPLEX16 *) ptr->value)->re, (REAL8) ((COMPLEX16 *) ptr->value)->im);
					break;
				case LALINFERENCE_gslMatrix_t:
					fprintf(stdout, "<can't print matrix>");
					break;
				default:
					fprintf(stdout, "<can't print>");
			}
		fprintf(fp,"\t");
		}
		ptr=ptr->next;
	}
	return;
}

int LALInferencePrintProposalStatsHeader(FILE *fp,LALInferenceVariables *propStats) {
  LALInferenceVariableItem *head = propStats->head;
  while (head != NULL) {
    fprintf(fp, "%s\t", head->name);
    head = head->next;
  }
  fprintf(fp, "\n");
  return 0;
}

void LALInferencePrintProposalStats(FILE *fp,LALInferenceVariables *propStats){
  REAL4 accepted = 0;
  REAL4 proposed = 0;
  REAL4 acceptanceRate = 0;

  if(propStats==NULL || fp==NULL) return;
  LALInferenceVariableItem *ptr=propStats->head;
  while(ptr!=NULL) {
    accepted = (REAL4) (*(LALInferenceProposalStatistics *) ptr->value).accepted;
    proposed = (REAL4) (*(LALInferenceProposalStatistics *) ptr->value).proposed;
    acceptanceRate = accepted/proposed;
    fprintf(fp, "%9.5f\t", acceptanceRate);
    ptr=ptr->next;
  }
  fprintf(fp, "\n");
  return;
}

const char *LALInferenceTranslateInternalToExternalParamName(const char *inName) {
  if (!strcmp(inName, "a_spin1")) {
    return "a1";
  } else if (!strcmp(inName, "a_spin2")) {
    return "a2";
  } else if (!strcmp(inName, "phi_spin1")) {
    return "phi1";
  } else if (!strcmp(inName, "phi_spin2")) {
    return "phi2";
  } else if (!strcmp(inName, "theta_spin1")) {
    return "theta1";
  } else if (!strcmp(inName, "theta_spin2")) {
    return "theta2";
  } else if (!strcmp(inName, "chirpmass")) {
    return "mc";
  } else if (!strcmp(inName, "massratio")) {
    return "eta";
  } else if (!strcmp(inName, "asym_massratio")) {
    return "q";
  } else if (!strcmp(inName, "rightascension")) {
    return "ra";
  } else if (!strcmp(inName, "declination")) {
    return "dec";
  } else if (!strcmp(inName, "phase")) {
    return "phi_orb";
  } else if (!strcmp(inName, "polarisation")) {
    return "psi";
  } else if (!strcmp(inName, "inclination")) {
    return "iota";
  } else if (!strcmp(inName, "distance")) {
    return "dist";
  } else {
    return inName;
  }
}

int LALInferenceFprintParameterNonFixedHeaders(FILE *out, LALInferenceVariables *params) {
  LALInferenceVariableItem *head = params->head;

  while (head != NULL) {
    if (head->vary != LALINFERENCE_PARAM_FIXED) {
      fprintf(out, "%s\t", LALInferenceTranslateInternalToExternalParamName(head->name));
    }
    head = head->next;
  }

  return 0;
}

int LALInferenceCompareVariables(LALInferenceVariables *var1, LALInferenceVariables *var2)
/*  Compare contents of "var1" and "var2".                       */
/*  Returns zero for equal entries, and one if difference found. */
/*  Make sure to only call this function when all entries are    */
/*  actually comparable. For example, "gslMatrix" type entries   */
/*  cannot (yet?) be checked for equality.                       */
{
  int result = 0;
  LALInferenceVariableItem *ptr1 = var1->head;
  LALInferenceVariableItem *ptr2 = NULL;
  if (var1->dimension != var2->dimension) result = 1;  // differing dimension
  while ((ptr1 != NULL) & (result == 0)) {
    ptr2 = LALInferenceGetItem(var2, ptr1->name);
    if (ptr2 != NULL) {  // corrsesponding entry exists; now compare type, then value:
      if (ptr2->type == ptr1->type) {  // entry type identical
        switch (ptr1->type) {  // do value comparison depending on type:
          case LALINFERENCE_INT4_t:
            result = ((*(INT4 *) ptr2->value) != (*(INT4 *) ptr1->value));
            break;
          case LALINFERENCE_INT8_t:
            result = ((*(INT8 *) ptr2->value) != (*(INT8 *) ptr1->value));
            break;
          case LALINFERENCE_UINT4_t:
            result = ((*(UINT4 *) ptr2->value) != (*(UINT4 *) ptr1->value));
            break;
          case LALINFERENCE_REAL4_t:
            result = ((*(REAL4 *) ptr2->value) != (*(REAL4 *) ptr1->value));
            break;
          case LALINFERENCE_REAL8_t:
            result = ((*(REAL8 *) ptr2->value) != (*(REAL8 *) ptr1->value));
            break;
          case LALINFERENCE_COMPLEX8_t:
            result = (((REAL4) ((COMPLEX8 *) ptr2->value)->re != (REAL4) ((COMPLEX8 *) ptr1->value)->re)
                      || ((REAL4) ((COMPLEX8 *) ptr2->value)->im != (REAL4) ((COMPLEX8 *) ptr1->value)->im));
            break;
          case LALINFERENCE_COMPLEX16_t:
            result = (((REAL8) ((COMPLEX16 *) ptr2->value)->re != (REAL8) ((COMPLEX16 *) ptr1->value)->re)
                      || ((REAL8) ((COMPLEX16 *) ptr2->value)->im != (REAL8) ((COMPLEX16 *) ptr1->value)->im));
            break;
          case LALINFERENCE_gslMatrix_t:
            XLAL_PRINT_WARNING("Cannot yet compare \"gslMatrix\" entries (entry: \"%s\"). For now, entries are by default assumed different.", ptr1->name);
            result = 1;
            break;
          default:
            XLAL_ERROR(XLAL_EFAILED, "Encountered unknown LALInferenceVariables type (entry: \"%s\").", ptr1->name);
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



ProcessParamsTable *LALInferenceGetProcParamVal(ProcessParamsTable *procparams,const char *name)
/* Returns pointer to element "name" of the ProcessParamsTable, */
/* if present, and NULL otherwise.                              */
{
  ProcessParamsTable *this=procparams;

  if (this==NULL) {
    fprintf(stderr, " Warning:  ProcessParamsTable is a NULL pointer\n");
    exit(1);
  }

  while (this!=NULL) {
    if (!strcmp(this->param, name)) break;
    else this=this->next;
  }

  return(this);
}



void LALInferenceParseCharacterOptionString(char *input, char **strings[], UINT4 *n)
/* parses a character string (passed as one of the options) and decomposes   */
/* it into individual parameter character strings. Input is of the form      */
/*   input   :  "[one,two,three]"                                            */
/* and the resulting output is                                               */
/*   strings :  {"one", "two", "three"}                                      */
/* length of parameter names is for now limited to 512 characters.           */
/* (should 'theoretically' (untested) be able to digest white space as well. */
/* Irrelevant for command line options, though.)                             */
{
  UINT4 i,j,k,l;
  /* perform a very basic well-formedness-check and count number of parameters: */
  i=0; j=0;
  *n = 0;
  while (input[i] != '\0') {
    if ((j==0) & (input[i]=='[')) j=1;
    if ((j==1) & (input[i]==',')) ++*n;
    if ((j==1) & (input[i]==']')) {++*n; j=2;}
    ++i;
  }
  if (j!=2) XLAL_ERROR_VOID(XLAL_EINVAL, "Argument vector \"%s\" is not well-formed!", input);
  /* now allocate memory for results: */
  *strings  = (char**)  XLALMalloc(sizeof(char*) * (*n));
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
        XLAL_PRINT_WARNING("Character \"%s\" argument too long!", (*strings)[k]);
      }
      else {
        (*strings)[k][l] = input[i];
        ++l;
      }
    }
    ++i;
  }
}



ProcessParamsTable *LALInferenceParseCommandLine(int argc, char *argv[])
/* parse command line and set up & fill in 'ProcessParamsTable' linked list.          */
/* If no command line arguments are supplied, the 'ProcessParamsTable' still contains */
/* one empty entry.                                                                   */
{
  int i, state=1;
  int dbldash;
  ProcessParamsTable *head, *ptr=NULL;
  /* always (even for argc==1, i.e. no arguments) put one element in list: */
  head = (ProcessParamsTable*) calloc(1, sizeof(ProcessParamsTable));
  XLALStringCopy(head->program, argv[0], sizeof(CHAR)*LIGOMETA_PROGRAM_MAX);
  ptr = head;
  i=1;
  while ((i<argc) & (state<=3)) {
    /* check for a double-dash at beginning of argument #i: */
    dbldash = ((argv[i][0]=='-') && (argv[i][1]=='-'));
    /* react depending on current state: */
    if (state==1){ /* ('state 1' means handling very 1st argument) */
      if (dbldash) {
        XLALStringCopy(head->param, argv[i], sizeof(CHAR)*LIGOMETA_PARAM_MAX);
        XLALStringCopy(ptr->type, "string", sizeof(CHAR)*LIGOMETA_TYPE_MAX);
        state = 2;
      }
      else { /* (very 1st argument needs to start with "--...") */
        // TODO: Perhaps this should be a warning?
        XLAL_ERROR_NULL(XLAL_EINVAL, "Orphaned first command line argument: \"%s\".", argv[i]);
        state = 4;
      }
    }
    else if (state==2) { /* ('state 2' means last entry was a parameter starting with "--") */
      if (dbldash) {
        ptr->next = (ProcessParamsTable*) calloc(1, sizeof(ProcessParamsTable));
        ptr = ptr->next;
        XLALStringCopy(ptr->program, argv[0],
sizeof(CHAR)*LIGOMETA_PROGRAM_MAX);
        XLALStringCopy(ptr->param, argv[i], sizeof(CHAR)*LIGOMETA_PARAM_MAX);
        XLALStringCopy(ptr->type, "string", sizeof(CHAR)*LIGOMETA_TYPE_MAX);
      }
      else {
        state = 3;
        XLALStringCopy(ptr->value, argv[i], sizeof(CHAR)*LIGOMETA_VALUE_MAX);
      }
    }
    else if (state==3) { /* ('state 3' means last entry was a value) */
      if (dbldash) {
        ptr->next = (ProcessParamsTable*) calloc(1, sizeof(ProcessParamsTable));
        ptr = ptr->next;
        XLALStringCopy(ptr->program, argv[0],
                       sizeof(CHAR)*LIGOMETA_PROGRAM_MAX);
        XLALStringCopy(ptr->param, argv[i], sizeof(CHAR)*LIGOMETA_PARAM_MAX);
        XLALStringCopy(ptr->type, "string", sizeof(CHAR)*LIGOMETA_TYPE_MAX);
        state = 2;
      }
      else {
        // TODO: Perhaps this should be a warning?
        XLAL_ERROR_NULL(XLAL_EINVAL, "Orphaned first command line argument: \"%s\".", argv[i]);
        state = 4;
      }
    }
    ++i;
  }

  return head;
}


char* LALInferencePrintCommandLine(ProcessParamsTable *procparams)
{
  ProcessParamsTable *this=procparams;
  INT8 len=14; //number of characters of the "Command line: " string.
  while (this!=NULL) {
    len+=strlen(this->param);
    len+=strlen(this->value);
    len+=2;
    this=this->next;
  }// Now we know how long the buffer has to be.
  char * str = (char*) calloc(len+1,sizeof(char));
  if (str==NULL) {
    XLALPrintError("Calloc error, str is NULL (in %s, line %d)\n",__FILE__, __LINE__);
		XLAL_ERROR_NULL(XLAL_ENOMEM);
  }
  
  this=procparams;
  strcpy (str,"Command line: ");
  //strcat (str,this->program);
  while (this!=NULL) {
    strcat (str," ");
    strcat (str,this->param);
    strcat (str," ");
    strcat (str,this->value);
    this=this->next;
  }
  return str;
}



void LALInferenceExecuteFT(LALInferenceIFOData *IFOdata)
/* Execute (forward, time-to-freq) Fourier transform.         */
/* Contents of IFOdata->timeModelh... are windowed and FT'ed, */
/* results go into IFOdata->freqModelh...                     */
/*  CHECK: keep or drop normalisation step here ?!?  */
{
  UINT4 i;
  double norm;
  int errnum; 
  
  if (IFOdata==NULL) {
   		fprintf(stderr," ERROR: IFOdata is a null pointer at LALInferenceExecuteFT, exiting!.\n");
   		XLAL_ERROR_VOID(XLAL_EFAULT);
  }

  else if(!IFOdata->timeData && IFOdata->timeData){				
		XLALPrintError("timeData is NULL at LALInferenceExecuteFT, exiting!");
	 	XLAL_ERROR_VOID(XLAL_EFAULT);	
  }

  else if(!IFOdata->freqData && IFOdata->timeData){
		XLALPrintError("freqData is NULL at LALInferenceExecuteFT, exiting!");
		XLAL_ERROR_VOID(XLAL_EFAULT);	
  }

  else if(!IFOdata->freqData && !IFOdata->timeData){
		XLALPrintError("timeData and freqData are NULL at LALInferenceExecuteFT, exiting!");
		XLAL_ERROR_VOID(XLAL_EFAULT);
  }
 
  else if(!IFOdata->freqData->data->length){
		XLALPrintError("Frequency series length is not set, exiting!");
		XLAL_ERROR_VOID(XLAL_EFAULT);
  }	
	
 
  for(;IFOdata;IFOdata=IFOdata->next){
    /* h+ */
  if(!IFOdata->freqModelhPlus){     
	
        XLAL_TRY(IFOdata->freqModelhPlus=(COMPLEX16FrequencySeries *)XLALCreateCOMPLEX16FrequencySeries("freqData",&(IFOdata->timeData->epoch),0.0,IFOdata->freqData->deltaF,&lalDimensionlessUnit,IFOdata->freqData->data->length),errnum);

		if (errnum){
		XLALPrintError("Could not create COMPLEX16FrequencySeries in LALInferenceExecuteFT");
		XLAL_ERROR_VOID(errnum);
		}
  }
	if (!IFOdata->window || !IFOdata->window->data){
		XLALPrintError("IFOdata->window is NULL at LALInferenceExecuteFT: Exiting!");
		XLAL_ERROR_VOID(XLAL_EFAULT);
		}

	XLAL_TRY(XLALDDVectorMultiply(IFOdata->timeModelhPlus->data,IFOdata->timeModelhPlus->data,IFOdata->window->data),errnum);

		if (errnum){
			XLALPrintError("Could not window time-series in LALInferenceExecuteFT");
			XLAL_ERROR_VOID(errnum);
			}
   		
	if (!IFOdata->timeToFreqFFTPlan){
		XLALPrintError("IFOdata->timeToFreqFFTPlan is NULL at LALInferenceExecuteFT: Exiting!");
		XLAL_ERROR_VOID(XLAL_EFAULT);
		}

	XLAL_TRY(XLALREAL8TimeFreqFFT(IFOdata->freqModelhPlus,IFOdata->timeModelhPlus,IFOdata->timeToFreqFFTPlan),errnum);
	
		if (errnum){
			XLALPrintError("Could not h_plus FFT time-series");
			XLAL_ERROR_VOID(errnum);
			}
    			    
 
    /* hx */
  if(!IFOdata->freqModelhCross){ 

	XLAL_TRY(IFOdata->freqModelhCross=(COMPLEX16FrequencySeries *)XLALCreateCOMPLEX16FrequencySeries("freqData",&(IFOdata->timeData->epoch),0.0,IFOdata->freqData->deltaF,&lalDimensionlessUnit,IFOdata->freqData->data->length),errnum);
	
		if (errnum){	
			XLALPrintError("Could not create COMPLEX16FrequencySeries in LALInferenceExecuteFT");
		 	XLAL_ERROR_VOID(errnum);		
			}
  }
	XLAL_TRY(XLALDDVectorMultiply(IFOdata->timeModelhCross->data,IFOdata->timeModelhCross->data,IFOdata->window->data),errnum);

		if (errnum){
			XLALPrintError("Could not window time-series in LALInferenceExecuteFT");
			XLAL_ERROR_VOID(errnum);
			}
		 
	XLAL_TRY(XLALREAL8TimeFreqFFT(IFOdata->freqModelhCross,IFOdata->timeModelhCross,IFOdata->timeToFreqFFTPlan),errnum);
	
		if (errnum){
			XLALPrintError("Could not FFT h_cross time-series");
			XLAL_ERROR_VOID(errnum);
			}   


    norm=sqrt(IFOdata->window->sumofsquares/IFOdata->window->data->length);
    
     for(i=0;i<IFOdata->freqModelhPlus->data->length;i++){
      IFOdata->freqModelhPlus->data->data[i].re*=norm;
      IFOdata->freqModelhPlus->data->data[i].im*=norm;
      IFOdata->freqModelhCross->data->data[i].re*=norm;
      IFOdata->freqModelhCross->data->data[i].im*=norm;
  }
 }
}

void LALInferenceExecuteInvFT(LALInferenceIFOData *IFOdata)
/* Execute inverse (freq-to-time) Fourier transform. */
/* Results go into 'IFOdata->timeModelh...'          */
{
  while (IFOdata != NULL) {
    if (IFOdata->freqToTimeFFTPlan==NULL) {
      XLAL_ERROR_VOID(XLAL_EFAULT, "Encountered unallocated \"freqToTimeFFTPlan\".");
    }

    /*  h+ :  */
    if (IFOdata->timeModelhPlus==NULL) {
      XLAL_ERROR_VOID(XLAL_EFAULT, "Encountered unallocated \"timeModelhPlus\".");
    }
    if (IFOdata->freqModelhPlus==NULL) {
      XLAL_ERROR_VOID(XLAL_EFAULT, "Encountered unallocated \"freqModelhPlus\".");
    }

    XLALREAL8FreqTimeFFT(IFOdata->timeModelhPlus, IFOdata->freqModelhPlus, IFOdata->freqToTimeFFTPlan);

    /*  hx :  */
    if (IFOdata->timeModelhCross==NULL) {
      XLAL_ERROR_VOID(XLAL_EFAULT, "Encountered unallocated \"timeModelhCross\".");
    }
    if (IFOdata->freqModelhCross==NULL) {
      XLAL_ERROR_VOID(XLAL_EFAULT, "Encountered unallocated \"freqModelhCross\".");
    }

    XLALREAL8FreqTimeFFT(IFOdata->timeModelhCross, IFOdata->freqModelhCross, IFOdata->freqToTimeFFTPlan);

    IFOdata=IFOdata->next;
  }
}

void LALInferenceProcessParamLine(FILE *inp, char **headers, LALInferenceVariables *vars) {
  size_t i;

  for (i = 0; headers[i] != NULL; i++) {
    double param;
    int nread;

    nread = fscanf(inp, " %lg ", &param);
    if (nread != 1) {
      XLAL_ERROR_VOID(XLAL_EFAILED, "Could not read the value of the %zu parameter in the row.", i);
    }

    LALInferenceAddVariable(vars, headers[i], &param, LALINFERENCE_REAL8_t, LALINFERENCE_PARAM_FIXED);
  }
}

/* This function has a Memory Leak!  You cannot free the allocated
   header buffer (of length MAXSIZE).  Don't call it too many times!
   (It's only expected to be called once to initialize the
   differential evolution array, so this should be OK. */
char **LALInferenceGetHeaderLine(FILE *inp) {
  const size_t MAXSIZE=1024;
  const char *delimiters = " \n\t";
  char *header = malloc(MAXSIZE*sizeof(char));
  char **colNames = NULL;  /* Will be filled in with the column names,
                              terminated by NULL. */
  size_t colNamesLen=0, colNamesMaxLen=0;
  char *colName = NULL;

  if (!fgets(header, MAXSIZE, inp)) {
    /* Some error.... */
    XLAL_ERROR_NULL(XLAL_EFAILED, "Error reading header line from file.");
  } else if (strlen(header) >= MAXSIZE-1) {
    /* Probably ran out of space before reading the entire line. */
    XLAL_ERROR_NULL(XLAL_EFAILED, "Header line too long (more than %zu chars).", MAXSIZE - 1);
  }

  /* Sure hope we read the whole line. */
  colNamesMaxLen=2;
  colNames=(char **)malloc(2*sizeof(char *));

  if (!colNames) {
    XLAL_ERROR_NULL(XLAL_ENOMEM, "Failed to allocate memory for colNames.");
  }

  colName=strtok(header, delimiters);
  strcpy(colNames[0],colNameToParamName(colName));
  //colNames[0] = colNameToParamName(colName); /* switched to strcpy() to avoid warning: assignment discards qualifiers from pointer target type */
  colNamesLen=1;
  do {
    colName=strtok(NULL, delimiters);

    strcpy(colNames[colNamesLen],colNameToParamName(colName));
    colNamesLen++;

    /* Expand if necessary. */
    if (colNamesLen >= colNamesMaxLen) {
      colNamesMaxLen *= 2;
      colNames=realloc(colNames, colNamesMaxLen*sizeof(char *));
      if (!colNames) {
        XLAL_ERROR_NULL(XLAL_ENOMEM, "Failed to reallocate memory for colNames.");
      }
    }

  } while (colName != NULL);

  /* Trim down to size. */
  colNames=realloc(colNames, colNamesLen*sizeof(char *));

  return colNames;
}

char *colNameToParamName(const char *colName) {
  char *retstr=NULL;
  if (colName == NULL) {
    return NULL;
  }
  else if (!strcmp(colName, "dist")) {
    retstr=XLALStringDuplicate("distance");
  }

  else if (!strcmp(colName, "ra")) {
    retstr=XLALStringDuplicate("rightascension");
  }

  else if (!strcmp(colName, "iota")) {
    retstr=XLALStringDuplicate("inclination");
  }

  else if (!strcmp(colName, "psi")) {
    retstr=XLALStringDuplicate("polarisation");
  }

  else if (!strcmp(colName, "mc")) {
    retstr=XLALStringDuplicate("chirpmass");
  }

  else if (!strcmp(colName, "phi_orb")) {
    retstr=XLALStringDuplicate("phase");
  }

  else if (!strcmp(colName, "eta")) {
    retstr=XLALStringDuplicate("massratio");
  }

  else if (!strcmp(colName, "q")) {
    retstr=XLALStringDuplicate("asym_massratio");
  }

  else if (!strcmp(colName, "dec")) {
    retstr=XLALStringDuplicate("declination");
  }

  /* Note the 1 <--> 2 swap between the post-proc world and the LI world. */
  else if (!strcmp(colName, "phi1")) {
    retstr=XLALStringDuplicate("phi_spin2");
  }

  else if (!strcmp(colName, "phi2")) {
    retstr=XLALStringDuplicate("phi_spin1");
  }

  else if (!strcmp(colName, "theta1")) {
    retstr=XLALStringDuplicate("theta_spin2");
  }

  else if (!strcmp(colName, "theta2")) {
    retstr=XLALStringDuplicate("theta_spin1");
  }

  else if (!strcmp(colName, "a1")) {
    retstr=XLALStringDuplicate("a_spin2");
  }
    
  else if (!strcmp(colName, "a2")) {
    retstr=XLALStringDuplicate("a_spin1");
  }
  else retstr=XLALStringDuplicate(colName);
  return retstr;
}

void LALInferenceSortVariablesByName(LALInferenceVariables *vars)
{
  LALInferenceVariables tmp;
  tmp.head=NULL;
  tmp.dimension=0;
  LALInferenceVariableItem *thisitem,*ptr;
  LALInferenceVariables *new=calloc(1,sizeof(*new));
  if(!vars){
    XLAL_ERROR_VOID(XLAL_EFAULT, "Received null input pointer.");
  }
  while(vars->head)
  {
    thisitem=vars->head;
    for (ptr=thisitem->next;ptr;ptr=ptr->next){
      if(strcmp(ptr->name,thisitem->name)<0)
        thisitem=ptr;
    }
    LALInferenceAddVariable(&tmp, thisitem->name, thisitem->value, thisitem->type, thisitem->vary);
    LALInferenceRemoveVariable(vars,thisitem->name);
  }
  vars->head=tmp.head;
  vars->dimension=tmp.dimension;
  return;
}

/** Append the sample to a file. file pointer is stored in state->algorithmParams as a
 * LALInferenceVariable called "outfile", as a void ptr.
 * Caller is responsible for opening and closing file.
 * Variables are alphabetically sorted before being written
 */
void LALInferenceLogSampleToFile(LALInferenceRunState *state, LALInferenceVariables *vars)
{
  FILE *outfile=NULL;
  if(LALInferenceCheckVariable(state->algorithmParams,"outfile"))
    outfile=*(FILE **)LALInferenceGetVariable(state->algorithmParams,"outfile");
  /* Write out old sample */
  if(outfile==NULL) return;
  LALInferenceSortVariablesByName(vars);
  LALInferencePrintSample(outfile,vars);
  fprintf(outfile,"\n");
}

/** Append the sample to an array which can be later processed by the user.
 * Array is stored as a C array in a LALInferenceVariable in state->algorithmParams
 * called "outputarray". Number of items in the array is stored as "N_outputarray".
 * Will create the array and store it in this way if it does not exist.
 * DOES NOT FREE ARRAY, user must clean up after use.
 * Also outputs sample to disk if possible */
void LALInferenceLogSampleToArray(LALInferenceRunState *state, LALInferenceVariables *vars)
{
  LALInferenceVariables *output_array=NULL;
  UINT4 N_output_array=0;
  LALInferenceLogSampleToFile(state,vars);

  /* Set up the array if it is not already allocated */
  if(LALInferenceCheckVariable(state->algorithmParams,"outputarray"))
    output_array=*(LALInferenceVariables **)LALInferenceGetVariable(state->algorithmParams,"outputarray");
  else
    LALInferenceAddVariable(state->algorithmParams,"outputarray",&output_array,LALINFERENCE_void_ptr_t,LALINFERENCE_PARAM_OUTPUT);

  if(LALInferenceCheckVariable(state->algorithmParams,"N_outputarray"))
    N_output_array=*(INT4 *)LALInferenceGetVariable(state->algorithmParams,"N_outputarray");
  else
    LALInferenceAddVariable(state->algorithmParams,"N_outputarray",&N_output_array,LALINFERENCE_INT4_t,LALINFERENCE_PARAM_OUTPUT);

  /* Expand the array for new sample */
  output_array=realloc(output_array, (N_output_array+1) *sizeof(LALInferenceVariables));
  if(!output_array){
    XLAL_ERROR_VOID(XLAL_EFAULT, "Unable to allocate array for samples.");
  }
  else
  {
    /* Save sample and update */
    memset(&(output_array[N_output_array]),0,sizeof(LALInferenceVariables));
    LALInferenceCopyVariables(vars,&output_array[N_output_array]);
    N_output_array++;

    LALInferenceSetVariable(state->algorithmParams,"outputarray",&output_array);
    LALInferenceSetVariable(state->algorithmParams,"N_outputarray",&N_output_array);
  }
  return;
}

void LALInferenceMcEta2Masses(double mc, double eta, double *m1, double *m2)
/*  Compute individual companion masses (m1, m2)   */
/*  for given chirp mass (m_c) & mass ratio (eta)  */
/*  (note: m1 >= m2).                              */
{
  double root = sqrt(0.25-eta);
  double fraction = (0.5+root) / (0.5-root);
  *m2 = mc * (pow(1+fraction,0.2) / pow(fraction,0.6));
  *m1 = mc * (pow(1+1.0/fraction,0.2) / pow(1.0/fraction,0.6));
  return;
}

void LALInferenceMcQ2Masses(double mc, double q, double *m1, double *m2)
/*  Compute individual companion masses (m1, m2)   */
/*  for given chirp mass (m_c) & asymmetric mass   */
/*  ratio (q).  note: q = m2/m1, where m1 >= m2    */
{
  double factor = mc * pow(1 + q, 1.0/5.0);
  *m1 = factor * pow(q, -3.0/5.0);
  *m2 = factor * pow(q, +2.0/5.0);
  return;
}
void LALInferenceQ2Eta(double q, double *eta)
/*  Compute symmetric mass ratio eta from the     */
/*  asymmetric mass ratio q.                      */
{
  *eta = q/((1+q)*(1+q));
  return;
}

static void deleteCell(LALInferenceKDCell *cell) {
  if (cell == NULL) {
    return; /* Our work here is done. */
  } else {
    size_t i;

    deleteCell(cell->left);
    deleteCell(cell->right);

    if (cell->lowerLeft != NULL) XLALFree(cell->lowerLeft);
    if (cell->upperRight != NULL) XLALFree(cell->upperRight);
    if (cell->ptsMean != NULL) XLALFree(cell->ptsMean);
    if (cell->eigenMin != NULL) XLALFree(cell->eigenMin);
    if (cell->eigenMax != NULL) XLALFree(cell->eigenMax);

    for (i = 0; i < cell->ptsSize; i++) {
      if (cell->pts[i] != NULL) XLALFree(cell->pts[i]);
    }
    if (cell->pts != NULL) XLALFree(cell->pts);

    if (cell->ptsCov != NULL) {
      for (i = 0; i < cell->dim; i++) {
        XLALFree(cell->ptsCov[i]);
      }
      XLALFree(cell->ptsCov);
    }

    if (cell->ptsCovEigenVects != NULL) {
      for (i = 0; i < cell->dim; i++) {
        XLALFree(cell->ptsCovEigenVects[i]);
      }
      XLALFree(cell->ptsCovEigenVects);
    }

    XLALFree(cell);

    return;
  }
}

typedef enum {
  LEFT,
  RIGHT,
  TOP
} cellType;

static LALInferenceKDCell *newCell(size_t ndim, REAL8 *lowerLeft, REAL8 *upperRight, size_t level, cellType type) {
  LALInferenceKDCell *cell = XLALCalloc(1, sizeof(LALInferenceKDCell));
  size_t i;

  cell->lowerLeft = XLALCalloc(ndim, sizeof(REAL8));
  cell->upperRight = XLALCalloc(ndim, sizeof(REAL8));
  cell->ptsMean = XLALCalloc(ndim, sizeof(REAL8));
  cell->eigenMin = XLALCalloc(ndim, sizeof(REAL8));
  cell->eigenMax = XLALCalloc(ndim, sizeof(REAL8));

  cell->pts = XLALCalloc(1, sizeof(REAL8 *));
  cell->ptsSize = 1;
  cell->npts = 0;
  cell->dim = ndim;

  cell->ptsCov = XLALCalloc(ndim, sizeof(REAL8 *));
  cell->ptsCovEigenVects = XLALCalloc(ndim, sizeof(REAL8 *));
  for (i = 0; i < ndim; i++) {
    cell->ptsCov[i] = XLALCalloc(ndim, sizeof(REAL8));
    cell->ptsCovEigenVects[i] = XLALCalloc(ndim, sizeof(REAL8));
  }
  
  memcpy(cell->upperRight, upperRight, ndim*sizeof(REAL8));
  memcpy(cell->lowerLeft, lowerLeft, ndim*sizeof(REAL8));
  if (type == LEFT) {
    cell->upperRight[level] = 0.5*(lowerLeft[level] + upperRight[level]);
  } else if (type == RIGHT) {
    cell->lowerLeft[level] = 0.5*(lowerLeft[level] + upperRight[level]);
  } else {
    /* Do not change lowerLeft or upperRight, since this is the top-level cell. */
  }

  return cell;
}

LALInferenceKDTree *LALInferenceKDEmpty(REAL8 *lowerLeft, REAL8 *upperRight, size_t ndim) {
  LALInferenceKDCell *cell = newCell(ndim, lowerLeft, upperRight, 0, TOP);
  return (LALInferenceKDTree *)cell;
}

static int equalPoints(REAL8 *a, REAL8 *b, size_t n) {
  size_t i;
  for (i = 0; i < n; i++) {
    if (a[i] != b[i]) return 0;
  }
  return 1;
}

static int cellAllEqualPoints(LALInferenceKDCell *cell) {
  if (cell->npts <= 1) {
    return 1;
  } else {
    size_t i;
    REAL8 *pt0 = cell->pts[0];
    
    for (i = 1; i < cell->npts; i++) {
      if (!equalPoints(pt0, cell->pts[i], cell->dim)) return 0;
    }

    return 1;
  }
}

static void addPtToCellPts(LALInferenceKDCell *cell, REAL8 *pt) {
  if (cell->npts == cell->ptsSize) {
    REAL8 **newPts = XLALCalloc(2*cell->ptsSize, sizeof(REAL8 *));

    memcpy(newPts, cell->pts, cell->ptsSize*sizeof(REAL8 *));

    free(cell->pts);
    cell->pts=newPts;
    cell->ptsSize *= 2;
  }

  cell->pts[cell->npts] = XLALCalloc(cell->dim, sizeof(REAL8));
  memcpy(cell->pts[cell->npts], pt, cell->dim*sizeof(REAL8));
  cell->npts += 1;
  cell->eigenFrameStale = 1;
}

/* The following routine handles inserting new points into the tree.
   It is organized recursively, inserting into cells based on the
   following three cases:

   1. There are no points in the cell.  Then insert the given point,
   and stop; we're at a leaf.

   2. Both of the cell's sub-cells are NULL.  Then the cell is a leaf
   (with some number of points), and the first order of business is to
   push the existing points down a level, then insert into this cell,
   and the appropriate sub-cell.

   3. This cell has non-null sub-cells.  Then it is not a leaf cell,
   and we should just insert into this cell, and add the point to the
   appropriate sub-cell.

*/
static int insertIntoCell(LALInferenceKDCell *cell, REAL8 *pt, size_t level) {
  size_t dim = cell->dim;
  size_t nextLevel = (level+1)%dim;
  level = level%dim;

  if (cell->npts == 0) {
    /* Insert this point into the cell, and quit. */
    addPtToCellPts(cell, pt);
    return XLAL_SUCCESS;
  } else if (cell->left == NULL && cell->right == NULL) {
    /* This cell is a leaf node.  Insert the point, then (unless the
       cell stores many copies of the same point), push everything
       down a level. */
    addPtToCellPts(cell, pt);

    if (cellAllEqualPoints(cell)) {
      /* Done, since there are many copies of the same point---cell
         remains a leaf. */
      return XLAL_SUCCESS;
    } else {
      size_t i;
      REAL8 mid = 0.5*(cell->lowerLeft[level] + cell->upperRight[level]);
      
      cell->left = newCell(dim, cell->lowerLeft, cell->upperRight, level, LEFT);
      cell->right = newCell(dim, cell->lowerLeft, cell->upperRight, level, RIGHT);

      for (i = 0; i < cell->npts; i++) {
        REAL8 *lowerPt = cell->pts[i];
        
        if (lowerPt[level] <= mid) {
          insertIntoCell(cell->left, lowerPt, nextLevel);
        } else {
          insertIntoCell(cell->right, lowerPt, nextLevel);
        }
      }

      return XLAL_SUCCESS;
    }
  } else {
    /* This is not a leaf cell, so insert, and then move down the tree. */
    REAL8 mid = 0.5*(cell->lowerLeft[level] + cell->upperRight[level]);

    addPtToCellPts(cell, pt);
    
    if (pt[level] <= mid) {
      return insertIntoCell(cell->left, pt, nextLevel);
    } else {
      return insertIntoCell(cell->right, pt, nextLevel);
    }
  }
}

static int inBounds(REAL8 *pt, REAL8 *low, REAL8 *high, size_t n) {
  size_t i;

  for (i = 0; i < n; i++) {
    if (pt[i] < low[i] || pt[i] > high[i]) return 0;
  }

  return 1;
}

static void computeMean(LALInferenceKDCell *cell) {
  REAL8 **pts = cell->pts;
  REAL8 *mean = cell->ptsMean;
  size_t dim = cell->dim;
  size_t npts = cell->npts;
  size_t i;

  if (pts == NULL) {
    return;
  }

  if (mean == NULL) XLAL_ERROR_VOID(XLAL_EINVAL, "given improperly initialized cell");

  for (i = 0; i < npts; i++) {
    size_t j;
    for (j = 0; j < dim; j++) {
      mean[j] += pts[i][j];
    }
  }

  for (i = 0; i < dim; i++) {
    mean[i] /= npts;
  }
}

static void computeCovariance(LALInferenceKDCell *cell) {
  REAL8 **cov = cell->ptsCov;
  REAL8 **pts = cell->pts;
  REAL8 *mu = cell->ptsMean;
  size_t npts = cell->npts;
  size_t dim = cell->dim;
  size_t i;

  if (pts == NULL) return;

  if (mu == NULL) {
    XLAL_ERROR_VOID(XLAL_EINVAL, "given cell with NULL mean");
  }

  if (cov == NULL) XLAL_ERROR_VOID(XLAL_EINVAL, "given improperly initialized cell");

  for (i = 0; i < npts; i++) {
    size_t j;

    REAL8 *pt = pts[i];

    for (j = 0; j < dim; j++) {
      size_t k;
      REAL8 ptj = pt[j];
      REAL8 muj = mu[j];

      for (k = 0; k < dim; k++) {
        REAL8 ptk = pt[k];
        REAL8 muk = mu[k];

        cov[j][k] += (ptj - muj)*(ptk-muk);
      }
    }
  }

  for (i = 0; i < cell->dim; i++) {
    size_t j;
    for (j = 0; j < cell->dim; j++) {
      cov[i][j] /= (npts - 1);
    }
  }
}

static void computeEigenVectorsCleanup(gsl_matrix *A, gsl_matrix *evects, gsl_vector *evals,
                                       gsl_eigen_symmv_workspace *ws) {
  if (A != NULL) gsl_matrix_free(A);
  if (evects != NULL) gsl_matrix_free(evects);
  if (evals != NULL) gsl_vector_free(evals);
  if (ws != NULL) gsl_eigen_symmv_free(ws);
}

static void computeEigenVectors(LALInferenceKDCell *cell) {
  size_t dim = cell->dim;
  gsl_matrix *A = gsl_matrix_alloc(dim, dim);
  gsl_matrix *evects = gsl_matrix_alloc(dim, dim);
  gsl_vector *evals = gsl_vector_alloc(dim);
  gsl_eigen_symmv_workspace *ws = gsl_eigen_symmv_alloc(dim);
  
  REAL8 **covEVs = cell->ptsCovEigenVects;
  REAL8 **cov = cell->ptsCov;

  size_t i;
  int status;

  if (A == NULL || evects == NULL || evals == NULL || ws == NULL) {
    computeEigenVectorsCleanup(A, evects, evals, ws);
    XLAL_ERROR_VOID(XLAL_ENOMEM);
  }

  if (cov == NULL) {
    computeEigenVectorsCleanup(A, evects, evals, ws);
    return;
  }

  if (covEVs == NULL) XLAL_ERROR_VOID(XLAL_EINVAL, "given improperly initialized cell");

  /* Copy covariance matrix into A. */
  for (i = 0; i < dim; i++) {
    size_t j;

    for (j = 0; j < dim; j++) {
      gsl_matrix_set(A, i, j, cov[i][j]);
    }
  }

  /* Compute evecs. */
  if ((status = gsl_eigen_symmv(A, evals, evects, ws)) != GSL_SUCCESS) {
    computeEigenVectorsCleanup(A, evects, evals, ws);
    XLAL_ERROR_VOID(status, "gsl error");
  }

  /* Copy eigenvector matrix into covEVs; [i][j] is the jth component
     of the ith eigenvector. */
  for (i = 0; i < dim; i++) {
    size_t j;
    
    for (j = 0; j < dim; j++) {
      covEVs[i][j] = gsl_matrix_get(evects, j, i);
    }
  }

  computeEigenVectorsCleanup(A, evects, evals, ws);
}

/* xe = eigenvs^T x */
static void toEigenFrame( size_t dim,  REAL8 **eigenvs,  REAL8 *x, REAL8 *xe) {
  size_t j;

  memset(xe, 0, dim*sizeof(REAL8));

  for (j = 0; j < dim; j++) {
    size_t i;
     REAL8 xj = x[j];
     REAL8 *evj = eigenvs[j];
    for (i = 0; i < dim; i++) {
      xe[i] += evj[i]*xj;
    }
  }
}

/* x = eigenvs xe */
static void fromEigenFrame( size_t dim,  REAL8 **eigenvs,  REAL8 *xe, REAL8 *x) {
  size_t i;
  
  memset(x, 0, dim*sizeof(REAL8));

  for (i = 0; i < dim; i++) {
    size_t j;
     REAL8 *evi = eigenvs[i];
    for (j = 0; j < dim; j++) {
      x[i] += evi[j]*xe[j];
    }
  }
}

static void computeEigenMinMax(LALInferenceKDCell *cell) {
  REAL8 **pts = cell->pts;
  size_t dim = cell->dim;
  size_t npts = cell->npts;
  REAL8 **eigenvs = cell->ptsCovEigenVects;
  REAL8 *mu = cell->ptsMean;

  REAL8 *min = cell->eigenMin;
  REAL8 *max = cell->eigenMax;

  REAL8 *xe = NULL, *x = NULL;

  size_t i;

  xe = XLALCalloc(dim, sizeof(REAL8));
  if (xe == NULL) XLAL_ERROR_VOID(XLAL_ENOMEM);

  x = XLALCalloc(dim, sizeof(REAL8));
  if (x == NULL) {
    XLALFree(xe);
    XLAL_ERROR_VOID(XLAL_ENOMEM);
  }

  if (pts==NULL) {
    XLALFree(x);
    XLALFree(xe);
    return;
  }

  if (eigenvs == NULL || mu == NULL) {
    XLALFree(x);
    XLALFree(xe);
    XLAL_ERROR_VOID(XLAL_EINVAL, "NULL cell components");
  }

  if (min == NULL) XLAL_ERROR_VOID(XLAL_EINVAL, "given improperly initialized cell");

  if (max == NULL) XLAL_ERROR_VOID(XLAL_EINVAL, "given improperly initialized cell");

  for (i = 0; i < dim; i++) {
    min[i] = 1.0/0.0;
    max[i] = -1.0/0.0;
  }

  for (i = 0; i < npts; i++) {
    size_t j;

    for (j = 0; j < dim; j++) {
      x[j] = pts[i][j] - mu[j];
    }

    toEigenFrame(dim, eigenvs, x, xe);

    for (j = 0; j < dim; j++) {
      if (xe[j] < min[j]) min[j] = xe[j];
      if (xe[j] > max[j]) max[j] = xe[j];
    }
  }
}

static void updateEigenSystem(LALInferenceKDCell *cell) {
  computeMean(cell);
  computeCovariance(cell);
  computeEigenVectors(cell);
  computeEigenMinMax(cell);

  cell->eigenFrameStale = 0;
}

int LALInferenceKDAddPoint(LALInferenceKDTree *tree, REAL8 *pt) {
  LALInferenceKDCell *cell = (LALInferenceKDCell *)tree;

  if (tree == NULL) XLAL_ERROR(XLAL_EINVAL, "given NULL tree");

  if (!inBounds(pt, cell->lowerLeft, cell->upperRight, cell->dim))
    XLAL_ERROR(XLAL_EINVAL, "given point that is not in global tree bounds");

  return insertIntoCell(cell, pt, 0);
}

static LALInferenceKDCell *doFindCell(LALInferenceKDCell *cell, REAL8 *pt, size_t dim, size_t Npts, size_t level) {
  if (cell == NULL) {
    /* If we encounter a NULL cell, then pass it up the chain. */
    return cell;
  } else if (cell->npts == 0) {
    XLAL_ERROR_NULL(XLAL_FAILURE, "could not find cell containing point");
  } else if (cell->npts == 1 || cell->npts < Npts) {
    return cell;
  } else {
    REAL8 mid = 0.5*(cell->lowerLeft[level] + cell->upperRight[level]);

    if (pt[level] <= mid) {
      LALInferenceKDCell *maybeCell = doFindCell(cell->left, pt, dim, Npts, (level+1)%dim);
      if (maybeCell == NULL) {
        return cell; /* If a NULL comes up from below, then this cell
                        is the one with the fewest points containing
                        pt. */
      } else {
        return maybeCell;
      }
    } else {
      LALInferenceKDCell *maybeCell = doFindCell(cell->right, pt, dim, Npts, (level+1)%dim);
      if (maybeCell == NULL) {
        return cell;
      } else {
        return maybeCell;
      }
    }
  }
}

LALInferenceKDCell *LALInferenceKDFindCell(LALInferenceKDTree *tree, REAL8 *pt, size_t Npts) {
  LALInferenceKDCell *cell = (LALInferenceKDCell *)tree;
  return doFindCell(cell, pt, cell->dim, Npts, 0);
}

double LALInferenceKDLogCellVolume(LALInferenceKDCell *cell) {
  size_t ndim = cell->dim;
  size_t i;
  REAL8 logVol = 0.0;
  for (i = 0; i < ndim; i++) {
    logVol += log(cell->upperRight[i] - cell->lowerLeft[i]);
  }

  return logVol;
}

double LALInferenceKDLogCellEigenVolume(LALInferenceKDCell *cell) {
  double logVol = 0.0;
  size_t ndim = cell->dim;
  size_t i;

  if (cell->eigenFrameStale) {
    updateEigenSystem(cell);
  }
  
  for (i = 0; i < ndim; i++) {
    logVol += log(cell->eigenMax[i] - cell->eigenMin[i]);
  }

  return logVol;
}

void LALInferenceKDVariablesToREAL8(LALInferenceVariables *params, REAL8 *pt, LALInferenceVariables *template) {
  LALInferenceVariableItem *templateItem = template->head;
  size_t i = 0;
  while (templateItem != NULL) {
    if (templateItem->vary != LALINFERENCE_PARAM_FIXED && 
        templateItem->vary != LALINFERENCE_PARAM_OUTPUT ) {
      pt[i] = *(REAL8 *)LALInferenceGetVariable(params, templateItem->name);
      i++;
    }
    templateItem = templateItem->next;
  }
}

void LALInferenceKDREAL8ToVariables(LALInferenceVariables *params, REAL8 *pt, LALInferenceVariables *template) {
  LALInferenceVariableItem *templateItem = template->head;
  size_t i = 0;
  while (templateItem != NULL) {
    if (templateItem->vary != LALINFERENCE_PARAM_FIXED && 
        templateItem->vary != LALINFERENCE_PARAM_OUTPUT) {
      LALInferenceSetVariable(params, templateItem->name, &(pt[i]));
      i++;
    }
    templateItem = templateItem->next;
  }
}

void LALInferenceKDDrawEigenFrame(gsl_rng *rng, LALInferenceKDTree *tree, REAL8 *pt, size_t Npts) {
  LALInferenceKDCell *topCell = (LALInferenceKDCell *)tree;

  if (topCell == NULL || topCell->npts == 0) XLAL_ERROR_VOID(XLAL_EINVAL, "cannot draw from empty cell");

  LALInferenceKDCell *cell = LALInferenceKDFindCell(tree, topCell->pts[gsl_rng_uniform_int(rng, topCell->npts)], Npts);

  if (cell->npts == 1) {
    /* If there is one point, then the covariance matrix is undefined,
       and we draw from the rectangular area. */
    size_t i;
    
    for (i = 0; i < cell->dim; i++) {
      pt[i] = cell->lowerLeft[i] + gsl_rng_uniform(rng)*(cell->upperRight[i] - cell->lowerLeft[i]);
    }
  } else {
    REAL8 *ept = XLALCalloc(cell->dim, sizeof(REAL8));
    size_t i;

    if (cell->eigenFrameStale) {
      updateEigenSystem(cell);
    }

    do {
      for (i = 0; i < cell->dim; i++) {
        ept[i] = cell->eigenMin[i] + gsl_rng_uniform(rng)*(cell->eigenMax[i] - cell->eigenMin[i]);
      }
      
      fromEigenFrame(cell->dim, cell->ptsCovEigenVects, ept, pt);
      
      for (i = 0; i < cell->dim; i++) {
        pt[i] += cell->ptsMean[i];
      }
    } while (!inBounds(pt, cell->lowerLeft, cell->upperRight, cell->dim));

    XLALFree(ept);
  }
}

static int inEigenBox(LALInferenceKDCell *cell,  REAL8 *pt) {
  REAL8 *shiftedPt, *ept;
  size_t i;

  if (cell->eigenFrameStale) {
    updateEigenSystem(cell);
  }

  shiftedPt = XLALCalloc(cell->dim, sizeof(REAL8));
  ept = XLALCalloc(cell->dim, sizeof(REAL8));

  for (i = 0; i < cell->dim; i++) {
    shiftedPt[i] = pt[i] - cell->ptsMean[i];
  }

  toEigenFrame(cell->dim, cell->ptsCovEigenVects, shiftedPt, ept);

  for (i = 0; i < cell->dim; i++) {
    if (ept[i] < cell->eigenMin[i] || ept[i] > cell->eigenMax[i]) {
      XLALFree(shiftedPt);
      XLALFree(ept);
      return 0;
    }
  }

  XLALFree(shiftedPt);
  XLALFree(ept);
  return 1;
}

REAL8 LALInferenceKDLogProposalRatio(LALInferenceKDTree *tree, REAL8 *current, 
                                     REAL8 *proposed, size_t Npts) {
  LALInferenceKDCell *currentCell = LALInferenceKDFindCell(tree, current, Npts);
  LALInferenceKDCell *proposedCell = LALInferenceKDFindCell(tree, proposed, Npts);

  LALInferenceKDCell *topCell = (LALInferenceKDCell *)tree;

  size_t npts = topCell->npts;

  REAL8 logCurrentVolume, logProposedVolume;

  if (currentCell->npts > 1) {
    if (currentCell->eigenFrameStale) {
      updateEigenSystem(currentCell);
    }
    
    if (!inEigenBox(currentCell, current)) {
      return log(0.0); /* If the current point is not in the eigen box
                          of the current cell, then we cannot jump
                          back there.  */
    } else {
      logCurrentVolume = LALInferenceKDLogCellEigenVolume(currentCell);
    }
  } else {
    logCurrentVolume = LALInferenceKDLogCellVolume(currentCell);
  }

  if (proposedCell->npts > 1) {
    /* Since we just proposed out of this cell, its eigenframe should
       *not* be stale. */
    if (proposedCell->eigenFrameStale) {
      XLAL_ERROR_REAL8(XLAL_EINVAL, "proposed cell eigen-frame is stale");
    }

    logProposedVolume = LALInferenceKDLogCellEigenVolume(proposedCell);
  } else {
    logProposedVolume = LALInferenceKDLogCellVolume(proposedCell);
  }

  REAL8 logCurrentCellFactor = log((REAL8)currentCell->npts / npts);
  REAL8 logProposedCellFactor = log((REAL8)proposedCell->npts / npts);

  return logCurrentCellFactor + logCurrentVolume - logProposedCellFactor - logProposedVolume;
}

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
#include <lal/LALInference.h>
#include <lal/Units.h>
#include <lal/FrequencySeries.h>
#include <lal/TimeFreqFFT.h>
#include <lal/VectorOps.h>
#include <lal/Date.h>
#include <lal/XLALError.h>

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
    XLALPrintError(" Error in getItemNr(): requesting zero or negative idx entry.\n");
    XLAL_ERROR_NULL(XLAL_EINVAL);
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
    XLALPrintError(" ERROR in getVariable(): entry \"%s\" not found.\n", name);
    XLAL_ERROR_NULL(XLAL_EFAILED);
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

LALInferenceVariableType LALInferenceGetVariableTypeByIndex(LALInferenceVariables *vars, int idx)
/* Returns type of the i-th entry, */
/* where  1 <= idx <= dimension. */
{
  LALInferenceVariableItem *item;
  if ((idx < 1) || (idx > vars->dimension)){
    XLALPrintError(" ERROR in LALInferenceGetVariableTypeByIndex(...,idx=%d): idx needs to be 1 <= idx <= dimension = %d.\n", 
            idx, vars->dimension);
    XLAL_ERROR(XLAL_EINVAL);
  }
  item = LALInferenceGetItemNr(vars, idx);
  return(item->type);
}


char *LALInferenceGetVariableName(LALInferenceVariables *vars, int idx)
/* Returns (pointer to) the name of the i-th entry, */
/* where  1 <= idx <= dimension.                  */
{
  LALInferenceVariableItem *item;
  if ((idx < 1) | (idx > vars->dimension)){
    XLALPrintError(" ERROR in LALInferenceGetVariableName(...,idx=%d): idx needs to be 1 <= idx <= dimension = %d.\n", 
            idx, vars->dimension);
    XLAL_ERROR_NULL(XLAL_EINVAL);
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
    XLALPrintError(" ERROR in LALInferenceSetVariable(): entry \"%s\" not found.\n", name);
    XLAL_ERROR_VOID(XLAL_EINVAL);
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
	    XLALPrintError(" ERROR in LALInferenceAddVariable(): Cannot re-add \"%s\" as previous definition has wrong type.\n",name);
	    XLAL_ERROR_VOID(XLAL_EFAILED);
	  }
	  LALInferenceSetVariable(vars,name,value);
	  return;
  }
	
  if(!value) {
    XLALPrintError("Unable to access value through null pointer in LALInferenceAddVariable, trying to add %s\n",name);
    XLAL_ERROR_VOID(XLAL_EFAULT);
  }

  LALInferenceVariableItem *new=XLALMalloc(sizeof(LALInferenceVariableItem));

  memset(new,0,sizeof(LALInferenceVariableItem));
	if(new) {
		new->value = (void *)XLALMalloc(LALInferenceTypeSize[type]);
	}
  if(new==NULL||new->value==NULL) {
    XLALPrintError(" ERROR in LALInferenceAddVariable(): unable to allocate memory for list item.\n");
    XLAL_ERROR_VOID(XLAL_ENOMEM);
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
  if(!this) {XLALPrintWarning(" WARNING in LALInferenceRemoveVariable(): entry \"%s\" not found.\n",name); return;}
  if(!parent) vars->head=this->next;
  else parent->next=this->next;
  XLALFree(this->value);
  this->value=NULL;
  XLALFree(this);
  this=NULL;
  vars->dimension--;
  return;
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
  /* Check that the source and origin differ */
  if(origin==target) return;

  LALInferenceVariableItem *ptr;
  if(!origin)
  {
	  XLALPrintError("Unable to access origin pointer in copyVariables\n");
	  XLAL_ERROR_VOID(XLAL_EFAULT);
  }

  /* Make sure the structure is initialised */
  if(!target) fprintf(stderr,"ERROR: Unable to copy to uninitialised LALInferenceVariables structure\n");
  /* first dispose contents of "target" (if any): */
  LALInferenceDestroyVariables(target);
  
  /* then copy over elements of "origin": */
  ptr = origin->head;
  if(!ptr)
  {
	  XLALPrintError("Bad LALInferenceVariable structure found while trying to copy\n");
	  XLAL_ERROR_VOID(XLAL_EFAULT);
  }
  while (ptr != NULL) {
	  if(!ptr->value || !ptr->name){
		  XLALPrintError("Badly formed LALInferenceVariableItem structure found in copyVariables!\n");
		  XLAL_ERROR_VOID(XLAL_EFAULT);
	  }
    LALInferenceAddVariable(target, ptr->name, ptr->value, ptr->type, ptr->vary);
    ptr = ptr->next;
  }
  return;
}

/** Prints a variable item to a string (must be pre-allocated!) */
void LALInferencePrintVariableItem(char *out, LALInferenceVariableItem *ptr)
{
  if(ptr==NULL) {
    XLALPrintError("Null LALInferenceVariableItem *");
    XLAL_ERROR_VOID(XLAL_EFAULT);
  }
  if(out==NULL) {
    XLALPrintError("Null output string *");
    XLAL_ERROR_VOID(XLAL_EFAULT);
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
					fprintf(fp, "%9.5f", *(REAL4 *) ptr->value);
					break;
				case LALINFERENCE_REAL8_t:
					fprintf(fp, "%9.5f", *(REAL8 *) ptr->value);
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
            fprintf(stderr, " WARNING: compareVariables() cannot yet compare \"gslMatrix\" type entries.\n");
            fprintf(stderr, "          (entry: \"%s\").\n", ptr1->name);
            fprintf(stderr, "          For now entries are by default assumed different.\n");
            result = 1;
            break;
          default:
            XLALPrintError( " ERROR: encountered unknown LALInferenceVariables type in compareVariables()\n");
            XLALPrintError( "        (entry: \"%s\").\n", ptr1->name);
            XLAL_ERROR(XLAL_EFAILED);
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
  if (j!=2) fprintf(stderr, " ERROR: argument vector \"%s\" not well-formed!\n", input);
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
        fprintf(stderr, " WARNING: orphaned first command line argument \"%s\" in parseCommandLine().\n", argv[i]);
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
        fprintf(stderr, " WARNING: orphaned command line argument \"%s\" in parseCommandLine().\n", argv[i]);
        state = 4;
      }     
    }
    ++i;
  }
  if (state==4) {
    XLALPrintError(" ERROR in parseCommandLine(): failed parsing command line options.\n");
    XLAL_ERROR_NULL(XLAL_EFAILED);
  }
  return(head);
}


void LALInferencePrintCommandLine(ProcessParamsTable *procparams, char *str)
{
	ProcessParamsTable *this=procparams;
	strcpy (str,"Command line: ");
	//strcat (str,this->program);
	while (this!=NULL) { 
		strcat (str," ");
		strcat (str,this->param);
		strcat (str," ");
		strcat (str,this->value);
		this=this->next;
	}
}



void LALInferenceExecuteFT(LALInferenceIFOData *IFOdata)
/* Execute (forward, time-to-freq) Fourier transform.         */
/* Contents of IFOdata->timeModelh... are windowed and FT'ed, */
/* results go into IFOdata->freqModelh...                     */
/*  CHECK: keep or drop normalisation step here ?!?  */
{
  UINT4 i;
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



void LALInferenceExecuteInvFT(LALInferenceIFOData *IFOdata)
/* Execute inverse (freq-to-time) Fourier transform. */
/* Results go into 'IFOdata->timeModelh...'          */
{
  while (IFOdata != NULL) {
    if (IFOdata->freqToTimeFFTPlan==NULL) {
      XLALPrintError(" ERROR in executeInvFT(): encountered unallocated 'freqToTimeFFTPlan'.\n");
      XLAL_ERROR_VOID(XLAL_EFAULT);
    }

    /*  h+ :  */
    if (IFOdata->timeModelhPlus==NULL) {
      XLALPrintError(" ERROR in executeInvFT(): encountered unallocated 'timeModelhPlus'.\n");
      XLAL_ERROR_VOID(XLAL_EFAULT);
    }
    if (IFOdata->freqModelhPlus==NULL) {
      XLALPrintError(" ERROR in executeInvFT(): encountered unallocated 'freqModelhPlus'.\n");
      XLAL_ERROR_VOID(XLAL_EFAULT);
    }
    
    XLALREAL8FreqTimeFFT(IFOdata->timeModelhPlus, IFOdata->freqModelhPlus, IFOdata->freqToTimeFFTPlan);

    if (*XLALGetErrnoPtr()) {
      XLALPrintError( "XLAL Error: %s (in %s, line %d)\n",
              XLALErrorString(xlalErrno), __FILE__, __LINE__);
      XLAL_ERROR_VOID(XLAL_EFAILED);
    }
    
    /*  hx :  */
    if (IFOdata->timeModelhCross==NULL) {
      XLALPrintError(" ERROR in executeInvFT(): encountered unallocated 'timeModelhCross'.\n");
      XLAL_ERROR_VOID(XLAL_EFAULT);
    }
    if (IFOdata->freqModelhCross==NULL) {
      XLALPrintError(" ERROR in executeInvFT(): encountered unallocated 'freqModelhCross'.\n");
      XLAL_ERROR_VOID(XLAL_EFAULT);
    }
    
    XLALREAL8FreqTimeFFT(IFOdata->timeModelhCross, IFOdata->freqModelhCross, IFOdata->freqToTimeFFTPlan);

    if (xlalErrno) {
      XLALPrintError( "XLAL Error: %s (in %s, line %d)\n",
              XLALErrorString(xlalErrno), __FILE__, __LINE__);
      XLAL_ERROR_VOID(XLAL_EFAILED);
    }
    
    IFOdata=IFOdata->next;
  }
}

int LALInferenceProcessParamLine(FILE *inp, char **headers, LALInferenceVariables *vars) {
  size_t i;

  for (i = 0; headers[i] != NULL; i++) {
    double param;
    int nread;
    
    nread = fscanf(inp, " %lg ", &param);

    if (nread != 1) {
      XLALPrintError( "Could not read parameter value, the %zu parameter in the row (in %s, line %d)\n",
              i, __FILE__, __LINE__);
      XLAL_ERROR(XLAL_EFAILED);
    }

    LALInferenceAddVariable(vars, headers[i], &param, LALINFERENCE_REAL8_t, LALINFERENCE_PARAM_FIXED);
  }

  return 0;
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
    XLALPrintError("Error reading header line from file (in %s, line %d)\n",
            __FILE__, __LINE__);
    XLAL_ERROR_NULL(XLAL_EFAILED);
  } else if (strlen(header) >= MAXSIZE-1) {
    /* Probably ran out of space before reading the entire line. */
    XLALPrintError("Header line too long (more than %zu chars) in %s, line %d.\n",
            MAXSIZE-1, __FILE__, __LINE__);
    XLAL_ERROR_NULL(XLAL_EFAILED);
  }

  /* Sure hope we read the whole line. */
  colNamesMaxLen=2;
  colNames=(char **)malloc(2*sizeof(char *));

  if (!colNames) {
    XLALPrintError("Failed to allocate colNames (in %s, line %d).\n",
            __FILE__, __LINE__);
    XLAL_ERROR_NULL(XLAL_ENOMEM);
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
        XLALPrintError("Failed to realloc colNames (in %s, line %d).\n",
                __FILE__, __LINE__);
	XLAL_ERROR_NULL(XLAL_ENOMEM);
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
    XLALPrintError("Received null input pointer");
    XLAL_ERROR_VOID(XLAL_EFAULT);
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
    XLALPrintError("Unable to allocate array for samples\n");
    XLAL_ERROR_VOID( XLAL_EFAULT );
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

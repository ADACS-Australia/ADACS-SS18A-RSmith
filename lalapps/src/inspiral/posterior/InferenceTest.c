#include <stdio.h>
#include "LALInference.h"


LALVariables variables;
REAL4 number,five;
ProcessParamsTable *ppt, *ptr;
int i;


int main(int argc, char *argv[]){
	number = 10.0;
	five=5.0;
	variables.head=NULL;
	variables.dimension=0;
	addVariable(&variables,"number",&number,REAL4_t);
	number=*(REAL4 *)getVariable(&variables,"number");
	fprintf(stdout,"Got %lf\n",number);
	setVariable(&variables,"number",&five);
	number=*(REAL4 *)getVariable(&variables,"number");
	fprintf(stdout,"Got %lf\n",number);
	fprintf(stdout,"Checkvariable?: %i\n",checkVariable(&variables,"number"));	
	removeVariable(&variables,"number");
	fprintf(stdout,"Removed, Checkvariable?: %i\n",checkVariable(&variables,"number"));
	destroyVariables(&variables);
  ppt = (ProcessParamsTable*) parseCommandLine(argc,argv);
  printf("parsed command line arguments:\n");
  ptr = ppt;
  i=1;
  while (ptr != NULL){
    printf(" (%d)  %s  %s  %s  \"%s\"\n", i, ptr->program, ptr->param, ptr->type, ptr->value);
    ptr = ptr->next;
    ++i;
  }	
}

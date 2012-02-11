/*
*  Copyright (C) 2007 Chad Hanna, Benjamin Owen
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

#if 0
/*________________________________________________________________________
 *
 * File Name: MakeTemplateBank.c
 *
 * Author: Hanna C. R.
 *
 *
 *________________________________________________________________________
 */

#include<lal/LALStdlib.h>
#include<lal/LALStatusMacros.h>
#include<lal/InspiralBankGeneration.h>
#include<lal/LIGOMetadataTables.h>

void
LALMakeTemplateBank(
     LALStatus *status,
     TemplateBankType *type,
     MakeTemplateBankInput *input,
     MetadataTable *table)
{
  INITSTATUS(status);
  ATTATCHSTATUSPTR(status);
  if(type == NULL){
    ABORT(status, TEMPLATEBANKGENERATIONH_ENULL, TEMPLATEBANKGENERATIONH_MSGENULL);
    }
  if(table == NULL){
    ABORT(status, TEMPLATEBANKGENERATIONH_ENULL, TEMPLATEBANKGENERATIONH_MSGENULL);
    }
  if (input == NULL){
    ABORT(status, TEMPLATEBANKGENERATIONH_ENULL, TEMPLATEBANKGENERATIONH_MSGENULL);
    }

/* look at inspiral Searches */
#if 0
  if ((*type >= 100) && (*type < 200)){
    printf("\nInside if type statement in MakeTemplateBank\n");
    TRY(LALInspiralBankGeneration(status->statusPtr,
                              type,
                              input->InspiralInput,
                              table->snglInspiralTable), status);
    printf("Just called LALInspiralBankGeneration\n");
    }
#endif



  /*
  if ((*type >= 200) && (*type < 300)){
    LALPulsarBankGeneration(status->statusPointer, type,
                            input.PulsarInput,
                            table.pulsarTable);
  }


  if ((*type >= 300) && (*type < 400)){
    LALBurstBankGeneration(status->statusPointer, type,
                           input.BurstInput,
                           table.snglBurstTable);
  }*/

  DETATCHSTATUSPTR(status);
  RETURN(status);

} /* LALMakeTemplateBank() */


#endif

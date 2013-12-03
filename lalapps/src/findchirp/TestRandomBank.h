/*
*  Copyright (C) 2007 Thomas Cokelaer
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

#include <processtable.h>
#include <stdio.h>
#include <gsl/gsl_histogram.h>
#include <lalapps.h>

#include <lal/LALNoiseModels.h>
#include <lal/LALInspiralBank.h>
#include <lal/RealFFT.h>
#include <lal/AVFactories.h>
#include <lal/SeqFactories.h>
#include <lal/LIGOMetadataUtils.h>
#include <lal/LIGOMetadataTables.h>

#include <lal/LIGOLwXMLHeaders.h>
#include <lal/LALConfig.h>
#include <lal/LALStdio.h>
#include <lal/LALStdlib.h>
#include <lal/LALError.h>
#include <lal/LALDatatypes.h>
#include <lal/AVFactories.h>
#include <lal/LALFrStream.h>
#include <lal/FrameCalibration.h>
#include <lal/Window.h>
#include <lal/TimeFreqFFT.h>
#include <lal/IIRFilter.h>
#include <lal/ResampleTimeSeries.h>
#include <lal/BandPassTimeSeries.h>
#include <lal/LIGOMetadataTables.h>
#include <lal/LIGOLwXML.h>
#include <lal/LIGOLwXMLRead.h>
#include <lal/LIGOLwXMLInspiralHeaders.h>
#include <lal/Date.h>
#include <lal/Units.h>
#include <lal/FindChirp.h>
#include <lal/PrintFTSeries.h>


/* Here, I defined my own xml table outside the lal strcuture although
   it can be put  into the liXmlHeader files I guess. I dont want to
   use the lal definition for the  time being in order to avoid any 
   overlap with other users. */


#define myfprintf(fp,oldmacro) PRINT_ ## oldmacro(fp)

#define MAXIFO 2
#define BANKEFFICIENCY_PARAMS_ROW \
"         %f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%d,%d,%d"
#define BANKEFFICIENCY_PARAMS_ROW_SPACE \
"         %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %d %d %d "

/*do not use capital here for future mysql migration */
#define PRINT_LIGOLW_XML_BANKEFFICIENCY(fp) ( \
fputs( "   <Table Name=\"bankefficiencygroup:bankefficiency:table\">\n", fp) == EOF || \
fputs( "      <Column Name=\"bankefficiencygroup:bankefficiency:psi0\"                Type=\"real_4\"/>\n", fp) == EOF ||  \
fputs( "      <Column Name=\"bankefficiencygroup:bankefficiency:psi3\"                Type=\"real_4\"/>\n", fp) == EOF ||  \
fputs( "      <Column Name=\"bankefficiencygroup:bankefficiency:beta\"                Type=\"real_4\"/>\n", fp) == EOF ||  \
fputs( "      <Column Name=\"bankefficiencygroup:bankefficiency:psi0_sim\"            Type=\"real_4\"/>\n", fp) == EOF ||  \
fputs( "      <Column Name=\"bankefficiencygroup:bankefficiency:psi3_sim\"            Type=\"real_4\"/>\n", fp) == EOF ||  \
fputs( "      <Column Name=\"bankefficiencygroup:bankefficiency:beta_sim\"            Type=\"real_4\"/>\n", fp) == EOF ||  \
fputs( "      <Column Name=\"bankefficiencygroup:bankefficiency:tau0\"                Type=\"real_4\"/>\n", fp) == EOF ||  \
fputs( "      <Column Name=\"bankefficiencygroup:bankefficiency:tau3\"                Type=\"real_4\"/>\n", fp) == EOF ||  \
fputs( "      <Column Name=\"bankefficiencygroup:bankefficiency:tau0_sim\"            Type=\"real_4\"/>\n", fp) == EOF ||  \
fputs( "      <Column Name=\"bankefficiencygroup:bankefficiency:tau3_sim\"            Type=\"real_4\"/>\n", fp) == EOF ||  \
fputs( "      <Column Name=\"bankefficiencygroup:bankefficiency:spin1_x\"            Type=\"real_4\"/>\n", fp) == EOF ||  \
fputs( "      <Column Name=\"bankefficiencygroup:bankefficiency:spin1_y\"            Type=\"real_4\"/>\n", fp) == EOF ||  \
fputs( "      <Column Name=\"bankefficiencygroup:bankefficiency:spin1_z\"            Type=\"real_4\"/>\n", fp) == EOF ||  \
fputs( "      <Column Name=\"bankefficiencygroup:bankefficiency:spin2_x\"            Type=\"real_4\"/>\n", fp) == EOF ||  \
fputs( "      <Column Name=\"bankefficiencygroup:bankefficiency:spin2_y\"            Type=\"real_4\"/>\n", fp) == EOF ||  \
fputs( "      <Column Name=\"bankefficiencygroup:bankefficiency:spin2_z\"            Type=\"real_4\"/>\n", fp) == EOF ||  \
fputs( "      <Column Name=\"bankefficiencygroup:bankefficiency:theta0\"            Type=\"real_4\"/>\n", fp) == EOF ||  \
fputs( "      <Column Name=\"bankefficiencygroup:bankefficiency:phi0\"            Type=\"real_4\"/>\n", fp) == EOF ||  \
fputs( "      <Column Name=\"bankefficiencygroup:bankefficiency:ffinal\"              Type=\"real_4\"/>\n", fp) == EOF ||  \
fputs( "      <Column Name=\"bankefficiencygroup:bankefficiency:ffinal_sim\"          Type=\"real_4\"/>\n", fp) == EOF ||  \
fputs( "      <Column Name=\"bankefficiencygroup:bankefficiency:mass1_sim\"           Type=\"real_4\"/>\n", fp) == EOF ||  \
fputs( "      <Column Name=\"bankefficiencygroup:bankefficiency:mass2_sim\"           Type=\"real_4\"/>\n", fp) == EOF ||  \
fputs( "      <Column Name=\"bankefficiencygroup:bankefficiency:phase_sim\"           Type=\"real_4\"/>\n", fp) == EOF ||  \
fputs( "      <Column Name=\"bankefficiencygroup:bankefficiency:snr\"                 Type=\"real_4\"/>\n", fp) == EOF ||  \
fputs( "      <Column Name=\"bankefficiencygroup:bankefficiency:snr_at_ta\"           Type=\"real_4\"/>\n", fp) == EOF ||  \
fputs( "      <Column Name=\"bankefficiencygroup:bankefficiency:phase\"           Type=\"real_4\"/>\n", fp) == EOF ||  \
fputs( "      <Column Name=\"bankefficiencygroup:bankefficiency:alpha_f\"             Type=\"real_4\"/>\n", fp) == EOF ||  \
fputs( "      <Column Name=\"bankefficiencygroup:bankefficiency:time\"                Type=\"int_4s\"/>\n", fp) == EOF ||  \
fputs( "      <Column Name=\"bankefficiencygroup:bankefficiency:time_sim\"            Type=\"int_4s\"/>\n", fp) == EOF ||  \
fputs( "      <Column Name=\"bankefficiencygroup:bankefficiency:nfast\"               Type=\"int_4s\"/>\n", fp) == EOF ||  \
fputs( "      <Stream Name=\"bankefficiencygroup:bankefficiency:table\"               Type=\"Local\" Delimiter=\",\">\n", fp) == EOF ) 



#define CVS_ID_STRING      "$Id$"
#define CVS_NAME_STRING    "$Name$"
#define CVS_REVISION       "$Revision$"
#define CVS_SOURCE         "$Source$"
#define CVS_DATE           "$Date$"
#define PROGRAM_NAME       "BankEfficiency"


/* --- Some Error messages --- */
#define BANKEFFICIENCY_ENORM  0
#define BANKEFFICIENCY_ESUB   1
#define BANKEFFICIENCY_EARG   2
#define BANKEFFICIENCY_EVAL   3
#define BANKEFFICIENCY_EFILE  4
#define BANKEFFICIENCY_EINPUT 5
#define BANKEFFICIENCY_EMEM   6

#define BANKEFFICIENCY_MSGENORM  "Normal exit"
#define BANKEFFICIENCY_MSGESUB   "Subroutine failed"
#define BANKEFFICIENCY_MSGEARG   "Error parsing arguments"
#define BANKEFFICIENCY_MSGEVAL   "Input argument out of valid range"
#define BANKEFFICIENCY_MSGEFILE  "Could not open file"
#define BANKEFFICIENCY_MSGEINPUT "Error reading file"
#define BANKEFFICIENCY_MSGEMEM   "Out of memory"
#define BANKEFFICIENCY_MSGPARSER "Missing arguments ??  "

/* --- Some constants --- 
 * useful to fill  InspiralTemplate, Bank and userParam structure */
#define BANKEFFICIENCY_ALPHABANK       		0.01
#define BANKEFFICIENCY_ALPHASIGNAL    		0.
#define BANKEFFICIENCY_FLOWER       		40.
#define BANKEFFICIENCY_HIGHGM                   6 
#define BANKEFFICIENCY_IETA	                1
#define BANKEFFICIENCY_IFLSO           		0.
#define BANKEFFICIENCY_LOWGM                    3
#define BANKEFFICIENCY_INSIDEPOLYGON            1
#define BANKEFFICIENCY_MMCOARSE     		0.8
#define BANKEFFICIENCY_MMFINE       		0.9
#define BANKEFFICIENCY_MMIN            		5.
#define BANKEFFICIENCY_MMAX           		20.
#define BANKEFFICIENCY_NOISEMODEL           	"LIGOI"
#define BANKEFFICIENCY_NENDPAD         		0
#define BANKEFFICIENCY_NFCUT           		5
#define BANKEFFICIENCY_NOISEAMPLITUDE  		1.
#define BANKEFFICIENCY_NSTARTPAD  		1000
#define BANKEFFICIENCY_NTRIALS         		1
#define BANKEFFICIENCY_PSI0MIN        		10.
#define BANKEFFICIENCY_PSI0MAX    		250000.
#define BANKEFFICIENCY_PSI3MIN     		-2200.
#define BANKEFFICIENCY_PSI3MAX       		-10.
#define BANKEFFICIENCY_ORDER_SIGNAL     	twoPN
#define BANKEFFICIENCY_ORDER_TEMPLATE   	twoPN
#define BANKEFFICIENCY_SIGNAL  		       -1
#define BANKEFFICIENCY_SIGNALAMPLITUDE 		10.
#define BANKEFFICIENCY_SPACE    		Psi0Psi3
#define BANKEFFICIENCY_STARTTIME       		0.
#define BANKEFFICIENCY_STARTPHASE    		0.
#define BANKEFFICIENCY_TEMPLATE        	       -1
#define BANKEFFICIENCY_TYPE            		0
#define BANKEFFICIENCY_TSAMPLING    		2048.
#define BANKEFFICIENCY_USEED        		122888

/* Other Parameters  1 = true ; 0 = false	*/
#define BANKEFFICIENCY_ALPHAFCONSTRAINT         1       /* alphaF between 0 and 1	        */
#define BANKEFFICIENCY_ALPHAFUNCONSTRAINT       0 	/* alphaF between 0 and 1	        */
#define BANKEFFICIENCY_QUIETFLAG       	        0 	/* silent                               */ 
#define BANKEFFICIENCY_FASTSIMULATION           0       /* cheating code (dont use template far 
                                                           from injection; use with care        */
#define BANKEFFICIENCY_PRINTOVERLAP             0	/* Print Best Overlap 			*/
#define BANKEFFICIENCY_PRINTBESTOVERLAP         0	/* Print Best Overlap 			*/
#define BANKEFFICIENCY_PRINTBESTTEMPLATE        0	/* Print Best Template 			*/
#define BANKEFFICIENCY_PRINTSNRHISTO            0	/* Print histogram of the n template correlation.*/
#define BANKEFFICIENCY_PRINTOVERLAP_FILE        "BE_BestOverlap.dat"	/* Print Best Overlap in a file */
#define BANKEFFICIENCY_PRINTFILTER              0	/* Print corresponding Filter		*/
#define BANKEFFICIENCY_PRINTBANK		0	/* print the template bank		*/
#define BANKEFFICIENCY_PRINTBANK_FILEASCII	"BE_Bank.dat"		/* print template bank in a file*/
#define BANKEFFICIENCY_PRINTBANKXML		0	/* print the template bank		*/
#define BANKEFFICIENCY_PRINTBANK_FILEXML	"BE_Bank.xml"		/* print template bank in a file*/
#define BANKEFFICIENCY_PRINTRESULT		1	/* print the result (ascii)		*/
#define BANKEFFICIENCY_PRINTRESULTXML		0	/* print the template bank		*/
#define BANKEFFICIENCY_PRINTRESULT_FILEXML	"BE_Result.xml"		/* print the result (xml file)  */
#define BANKEFFICIENCY_PRINTPROTOTYPE		0	/* print the overlap of the templates	*/
#define BANKEFFICIENCY_PRINTBANKOVERLAP		0	/* print the overlap of the templates	*/
#define BANKEFFICIENCY_PRINTPSD                 0	/* print psd used in <x|x>      	*/
#define BANKEFFICIENCY_PRINTPSD_FILE		"BE_PSD.dat"		/* Print Psd in a file	*/
#define BANKEFFICIENCY_PRINTTEMPLATE    	0	/* print the  BCV final template	*/
#define BANKEFFICIENCY_FAITHFULNESS             0				


#define BANKEFFICIENCY_PRINTPROTO_FILEXML	"BE_Proto.xml"	/* print the result (xml file)  */
#define BANKEFFICIENCY_READXMLBANK	        "InputBank.xml"	/* print the result (xml file)  */


/* --- temporary flag for the sampling of real psd --- */
#define None 512
#define RealNoise 511


#define BEASCII2XML_INPUT1 "Trigger.dat"
#define BEASCII2XML_INPUT2 "BE_Proto.xml"
#define BEASCII2XML_OUTPUT "Trigger.xml"
#define BEASCII2XML_BANK   "BE_Bank.xml"


/* ==================== local structures ==================== */
/* 
 *  */

/* used in BCV for the choice of constraint or unconstraint SNR */
typedef enum {
  ALPHAFConstraint,
  ALPHAFUnconstraint,
  BOTHAlphaMethod
} AlphaConstraint;


/* what kind of injections to be performed ? */
typedef enum {
  NoUserChoice,
  BBH,
  BNS,
  BHNS
} BinaryInjection;

/* what kind of noisemodel ?*/
typedef enum{
    UNITY,
    LIGOI,
    LIGOA,
    GEO,
    TAMA,
    VIRGO,
    REALPSD,
    READPSD,
    EGO
} DetectorName;

/* what detectors (real noise only )*/
typedef enum{
  L1,
  H1,
  H2,
  V1,
  G1
}
DetectorFlag;


typedef enum{
  S1,S2,S3,S4,S5,S6
}
RunFlag;


typedef struct{ 
  CHAR *calCacheName;
  CHAR *frInCacheName;
}
DataFileIn;


typedef struct{ 
  DataFileIn S2, S3, S4, S5, S6;
} 
ScientificRunIn;

typedef struct{
  CHAR *chanName;
  ScientificRunIn dataFile;
}
DetectorParamIn;

typedef struct{
  AlphaConstraint alphaFConstraint;     /* force alpha_F to be constriant between 0 and 1 */
  INT4 signal;				/* name of the random signal to inject 	*/	
  INT4 template;			/* name of the template in the bank 	*/
  INT4 ntrials;				/* number of simulations		*/
  INT4 fastSimulation;                  /* target the injection in the bank --> less 
					   computation but Match might be less than the
					   optimal  is noise is injected or template and 
					   signal are not faithful (ie: BCV against physical
					   template */

  INT4 lengthFactor;			/* multiply estimated length of filters by that factor */
  INT4 printSNRHisto;		
  INT4 printBank;			/* print bank of templates 		*/
  INT4 printResultXml;
  INT4 printPrototype;
  INT4 printPsd;                        /* print the psd used in <x|x>          */
  INT4 realNoise;
  BinaryInjection binaryInjection;      /*injection will be set by the mass-range*/
  INT4 printBestOverlap, printBestTemplate, extraFinalPrinting ;

  INT4 faithfulness;
  INT4 snrAtCoaTime;
  double m1,m2, psi0,psi3, tau0, tau3, template_m1, template_m2;
  DetectorName noiseModel;
  REAL4   maxTotalMass;
  char *chanName;
  CHAR *calCacheName;
  CHAR *frInCacheName;
  INT4 startTime;
  DetectorParamIn L1;
  DetectorParamIn H1;
  DetectorParamIn H2;
  INT4 numSeconds;
  DetectorFlag detector;
  RunFlag run;
  REAL4 signalfFinal;
  INT4 startPhase;
  CHAR *inputPSD;
  INT4 useed;
  INT4 dataCheckPoint;
}
UserParametersIn;

/* struct to save the results */
typedef struct{
  REAL4  rhoMax;
  INT4   rhoBin;
  REAL4  alpha;
  REAL4  phase;
  REAL4  freq;
  INT4   layer;
  INT4   templateNumber;
  InspiralTemplate bestTemplate;

  InspiralTemplate bestUTemplate;
  REAL4 snrAtCoaTime;
} OverlapOutputIn;

/* structure to output the results */
typedef struct{
  REAL4 tau0_inject;
  REAL4 tau0_trigger;
  REAL4 tau3_inject;
  REAL4 tau3_trigger;
  REAL4 psi0_trigger;
  REAL4 psi3_trigger;
  REAL4 beta_trigger;
  REAL4 beta_inject;
  REAL4 psi0_inject;
  REAL4 psi3_inject;
  REAL4 fend_trigger;
  REAL4 fend_inject;
  REAL4 mass1_inject;
  REAL4 mass2_inject;
  INT4 layer;
  REAL4 rho_final;
  REAL4 alphaF;
  INT4 bin;
  REAL4 phase;
  UINT4 ntrial;
  UINT4 nfast;
  REAL4 snrAtCoaTime;
} ResultIn;


/* BCV moments */
typedef struct{
  REAL4Vector a11;
  REAL4Vector a21;
  REAL4Vector a22;
} BEMoments;


/* vectors for BCV storage*/
typedef struct{
  REAL4Vector fm5_3;
  REAL4Vector fm2_3;
  REAL4Vector fm7_6;
  REAL4Vector fm1_2;
} BEPowerVector;


/* As to be cleaned
 * Function to store the optimal  parameters of the overlap 
 * lmax : number of the layer
 * fMax : frequency cutoff of the template
 * jmax: number of the best template. use if one want to use the FMaximization option 
 * */
void
KeepHighestValues(OverlapOutputIn in , 
		  OverlapOutputIn *out
		  );


/* function to create the filters in the BCV overlap.
 * Input: 	- Vectors F^(-5/3) and so on
 * 		- Matrix of the moments for the amplitude
 * 		- indice at which the template starts
 * 		- psi0 and psi3 for the phase
 * Output	- The two orthogonal filters 
 * */
void 
LALCreateBCVFilters(REAL4Vector 	*Filter1,
		    REAL4Vector 	*Filter2,
		    BEPowerVector  *powerVector,
		    BEMoments      *moments,
		    UINT4 		kMin,
		    UINT4 		kMax,
		    REAL4 		psi0,
		    REAL4 		psi3);

void 
LALCreateBCVSpinFilters(REAL4Vector 	*FilterBCVSpin1,
			REAL4Vector 	*FilterBCVSpin2,
			REAL4Vector 	*FilterBCVSpin3,
			InspiralWaveOverlapIn   *overlapin,
			BEPowerVector  *powerVector,
			BEMoments      *moments,
			UINT4 		kMin,
			UINT4 		kMax);
		
void 
LALBankPrintAscii(MetadataTable          templateBank ,
		  UINT4                  numCoarse,
		  InspiralCoarseBankIn   coarseBankIn );

void
LALBankPrintXML(MetadataTable templateBank,
		InspiralCoarseBankIn   coarseBankIn,
		RandomInspiralSignalIn randIn,
		UserParametersIn           userParam);
void
CreateListfromTmplt(InspiralTemplate *insptmplt, 
		    SnglInspiralTable *tmpltCurrent);


void
BEGenerateInputData(LALStatus *status,
		    REAL4Vector * signal,
		    RandomInspiralSignalIn  *randIn,
		    UserParametersIn           userParam);

void
BEInitOverlapOutputIn(OverlapOutputIn *this);

/* Function to compute a orthogonal vector in BCV
 * */
void
LALGetOrthogonalFilter(REAL4Vector *filter);


/* Functon to compute the overlap between an injected signal 
 * and a set of templates based on BCV templates. 
 * It returns the overlap, the phase and alpha parameter. 
 * */
void
LALWaveOverlapBCV(LALStatus 		  *status,
		  REAL4Vector             *correlation,
		  InspiralWaveOverlapIn   *overlapin,
		  REAL4Vector             *Filter1, 
		  REAL4Vector             *Filter2,
		  UserParametersIn        userParam, 
		  OverlapOutputIn         *OverlapOutput,
		  BEMoments               *moments);

void
LALWaveOverlapBCVSpin(LALStatus 		  *status,
		      REAL4Vector             *correlation,
		      InspiralWaveOverlapIn   *overlapin,
		      REAL4Vector             *Filter1, 
		      REAL4Vector             *Filter2,
		      REAL4Vector             *Filter3,
		      UserParametersIn        userParam, 
		      OverlapOutputIn         *OverlapOutput
		      );


/* Function to store the moments needed by the BCV overlap process 
 * a11, a22, a21 are three vectors which stored the three components 
 * of the matrix needed in the BCV maximization process. 
 * */
void
LALCreateBCVMomentVector(BEMoments            *moments,
			 REAL8FrequencySeries *psd,
			 REAL8 sampling, REAL8 fLower,
			 INT4 length);

/* Function to create Vectors of the form y(f)= f^(-a/b). 
 * where f is the frequency. 
 * */
void 
LALCreateVectorFreqPower( REAL4Vector	 	*vector,
			  InspiralTemplate 	params,
			  int 			a,
			  int 			b);


/* function to generate waveform (has to be tested). 
 * */
void
LALGenerateWaveform(LALStatus 			*status,
		    REAL4Vector 		*signal,
		    RandomInspiralSignalIn 	*params);


void 
GetResult(
    LALStatus 			*status,
    InspiralTemplate     	*list,
    InspiralTemplate       	injected,
    OverlapOutputIn 	        bestOverlapout, 
    ResultIn                    *result,
    UserParametersIn            userParam );

/* Init the CoarseBank structure 
 * */
void
InitInspiralCoarseBankIn(
    InspiralCoarseBankIn 	*coarseIn);

/* Init the random structure 
 * */
void
InitRandomInspiralSignalIn(
    RandomInspiralSignalIn 	*randIn);

/* Init the UserParametersIn structure
 * */
void
InitUserParametersIn(
    UserParametersIn    *userParam);

/* Function to initialize the different strucutre */
void
ParametersInitialization(
    InspiralCoarseBankIn        *coarseIn, 
    RandomInspiralSignalIn	*randIn, 
    UserParametersIn		*userParam);
/* Parsing function 
 * */
void
ParseParameters(
    int         *argc, 
    char 	**argv,
    InspiralCoarseBankIn 	*coarseIn,
    RandomInspiralSignalIn 	*randIn,
    UserParametersIn 		*userParam);		     

/* function to check validity of some parameters
 * */
void 	
UpdateParams(InspiralCoarseBankIn 	*coarseIn,
	     RandomInspiralSignalIn 	*randIn,
	     UserParametersIn 		*userParam);


/* Default values
 * */
void 
SetDefault(InspiralCoarseBankIn 	*coarseIn, 
	   RandomInspiralSignalIn 	*randIn,
	   UserParametersIn 		*userParam);

/* Help Function 
 * */
void
Help();

/* Print Function for final results 
 * */
void
PrintResults(
    ResultIn                    result,
    RandomInspiralSignalIn      rand);



/* Print each  template bank overlap 
 * */
void
PrintBankOverlap(InspiralTemplateList 	**list,
		 int 			sizeBank,
		 float 			*overlap,
		 InspiralCoarseBankIn 	coarseIn);

/* Print the template bank coordinates in ascii format 
 * */
void
BEPrintBank(
    InspiralCoarseBankIn 	coarse, 
    InspiralTemplateList 	**list,
    UINT4 			sizeBank);

/* print the template bank coordinates  in xml format 
 * */
void 
BEPrintBankXml(
    InspiralTemplateList        *coarseList,
    UINT4 		        numCoarse,
    InspiralCoarseBankIn        coarseIn,
    RandomInspiralSignalIn      randIn,
    UserParametersIn            userParam
    );

void 
BEGetMaximumSize(
    LALStatus  *status, 		      
    RandomInspiralSignalIn  randIn,
    InspiralCoarseBankIn    coarseBankIn, 
    UINT4 *length
    );


void
BECreatePsd(
    LALStatus                   *status, 
    InspiralCoarseBankIn        *coarseBankIn, 
    RandomInspiralSignalIn      *randIn,
    UserParametersIn            userParam);

/* print an error  message 
 * */
void 
BEPrintError(char *chaine);

void
BEFillProc(
    ProcessParamsTable          *proc,
    InspiralCoarseBankIn        coarseIn,
    RandomInspiralSignalIn      randIn,
    UserParametersIn            userParam);

/* xml file for the standalone code */
void 
BEPrintResultsXml( 
    InspiralCoarseBankIn         coarseBankIn,
    RandomInspiralSignalIn       randIn,
    UserParametersIn             userParam,
    ResultIn                     trigger
    );
void 
BEPrintResultsXml2( 
    InspiralCoarseBankIn         coarseBankIn,
    RandomInspiralSignalIn       randIn,
    UserParametersIn             userParam,
    ResultIn                     trigger
    );

void 
BEPrintProtoXml(
    InspiralCoarseBankIn          coarseIn,
    RandomInspiralSignalIn        randIn,
    UserParametersIn              userParam
    );

void
BEReadXmlBank(  LALStatus  *status, 
		CHAR *bankFileName, 
		InspiralTemplateList **list,
		INT4 *sizeBank, 
		InspiralCoarseBankIn coarseIn);


void 
LALCreateRealPsd(LALStatus *status, 
		 InspiralCoarseBankIn *bankIn,
		 RandomInspiralSignalIn randIn, 
		 UserParametersIn userParam);




typedef struct 
{  
  LIGOTimeGPS   gpsStartTime;
  LIGOTimeGPS   gpsEndTime;
  INT4          padData;                  
  INT4          numPoints;
  INT4          numSegments;
  CHAR          ifo[3];                           

  INT4          inputDataLength;
  INT4          resampFiltType;
  INT4          sampleRate;   
  INT4          highPass;     
  REAL4         highPassFreq; 
  INT4          highPassOrder;
  REAL4         highPassAtten;
  REAL4         fLow;
  INT4          specType;           

  CHAR          *calCacheName;
  INT4          pointCal;           
  REAL4         dynRangeExponent;          
  REAL4         geoHighPassFreq;            
  INT4          geoHighPassOrder;           
  REAL4         geoHighPassAtten; 
  
  INT4          randomSeed; 
  REAL4         gaussVar;   
  INT4          gaussianNoise;
 
  CHAR          *fqChanName;
  CHAR          *injectionFile;
  CHAR          *frInCacheName;
}
InspiralPipelineIn;



void SetInspiralPipelineParams(InspiralPipelineIn *param,
			      RandomInspiralSignalIn randIn);




typedef struct 
{
  INT4 numPoints;
  REAL4 gaussVar;
  REAL4 inputDeltaT;
  INT4 specType;
}
WindowSpectrumIn;

void 
LALComputeWindowSpectrum(LALStatus *status, 
			      WindowSpectrumIn *param,
			      REAL4FrequencySeries  *spec,
			      REAL4TimeSeries *chan);


void 
BECreateBank(
    LALStatus                   *status, 
    InspiralCoarseBankIn        *coarseBankIn,	
    InspiralTemplateList      	**list,
    INT4                        *sizeBank);



void
BECreatePowerVector(
    LALStatus              *status, 
    BEPowerVector          *powerVector,
    RandomInspiralSignalIn  randIn, 
    INT4                    length);


void 
LALInspiralOverlapBCV(
    LALStatus                   *status,
    InspiralTemplate            *list,
    BEPowerVector               *powerVector,
    UserParametersIn            *userParam, 
    RandomInspiralSignalIn      *randIn,
    REAL4Vector                 *Filter1,
    REAL4Vector                 *Filter2,
    InspiralWaveOverlapIn       *overlapin,
    OverlapOutputIn             *output,
    REAL4Vector                 *correlation,
    BEMoments                   *moments);

void 
LALInspiralOverlapBCVSpin(
    LALStatus                   *status,
    InspiralTemplate            *list,
    BEPowerVector               *powerVector,
    UserParametersIn            *userParam, 
    RandomInspiralSignalIn      *randIn,
    REAL4Vector                 *Filter1,
    REAL4Vector                 *Filter2,
    REAL4Vector                 *Filter3,
    InspiralWaveOverlapIn       *overlapin,
    OverlapOutputIn             *output,
    REAL4Vector                 *correlation,
    BEMoments                   *moments);




void
BEParseGetInt(  CHAR **argv, 
                INT4 *index,
                INT4 *data);

void
BEParseGetDouble(CHAR    **argv, 
		 INT4    *index,
		 REAL8    *data);

void
BEParseGetDouble2(CHAR    **argv, 
		  INT4    *index,
		  REAL8    *data1,                
		  REAL8    *data2);

void
BEParseGetString(  CHAR    **argv, 
		   INT4    *index );


void
PrintParameters(InspiralCoarseBankIn 	coarse,
		RandomInspiralSignalIn 	rand,
		UserParametersIn    	other);

CHAR* GetStringFromGridType(INT4 input);
CHAR* GetStringFromSimulationType(INT4 input);
CHAR* GetStringFromDetector(INT4 input);
CHAR* GetStringFromTemplate(INT4 input);
CHAR* GetStringFromNoiseModel(INT4 input);
CHAR* GetStringFromScientificRun(INT4 input);


void
BEAscii2XmlHelp(void);

void 
BEAscii2Xml();


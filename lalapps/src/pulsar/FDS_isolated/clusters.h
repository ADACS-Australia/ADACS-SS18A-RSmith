#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <lal/LALDatatypes.h>


/* #define BUFFERSIZE 1024 */


typedef struct Clusterstag {
  INT2  Nclusters;     //how many clusters
  UINT4  *NclustPoints; //for each cluster: how many points it has
  UINT4  *Iclust;       //index of each cluster point, for all clusters
  REAL8 *clusters;     //value of ratio for each cluster point, for all clusters
} Clusters;

typedef struct ClustersParamstag {
  INT4  wings;
  INT2  smallBlock;
} ClustersParams;

typedef struct Outlierstag {
  UINT4  Noutliers;
  INT4  rightwing;
  INT4  leftwing;
  UINT4  *outlierIndexes; // indexes in OutliersInput->data vector
  REAL8 *ratio;
} Outliers;

typedef struct OutliersInputtag {
  REAL8Vector *data;
  INT4        ifmin;
} OutliersInput;

typedef struct OutliersParamstag {
  REAL8Vector *Floor;
  REAL4       Thr; 
  INT4        wings;
  INT4        ifmin;
} OutliersParams;

typedef struct ClustersInputtag {
  OutliersInput  *outliersInput;
  OutliersParams *outliersParams;
  Outliers       *outliers;
} ClustersInput;



/* Function Prototypes */

int DetectClusters(ClustersInput *input, ClustersParams *params, Clusters *output);
int ComputeOutliers(OutliersInput *outliersInput, OutliersParams *outlierParams, Outliers *outliers);








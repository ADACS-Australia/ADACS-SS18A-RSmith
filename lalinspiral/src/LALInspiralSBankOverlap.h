#include <stdlib.h>
#include <lal/LALAtomicDatatypes.h>
#include <lal/ComplexFFT.h>
#include <lal/FrequencySeries.h>
#include <sys/types.h>

typedef struct tagWS {
    size_t n;
    COMPLEX8FFTPlan *plan;
    COMPLEX8Vector *zf;
    COMPLEX8Vector *zt;
} WS;

WS *XLALCreateSBankWorkspaceCache(void);
void XLALDestroySBankWorkspaceCache(WS *workspace_cache);
REAL8 XLALInspiralSBankComputeMatch(const COMPLEX8FrequencySeries *inj, const COMPLEX8FrequencySeries *tmplt, WS *workspace_cache);

REAL8 XLALInspiralSBankComputeRealMatch(const COMPLEX8FrequencySeries *inj, const COMPLEX8FrequencySeries *tmplt, WS *workspace_cache);

REAL8 XLALInspiralSBankComputeMatchMaxSkyLoc(const COMPLEX8FrequencySeries *hp, const COMPLEX8FrequencySeries *hc, const REAL8 hphccorr, const COMPLEX8FrequencySeries *proposal, WS *workspace_cache1, WS *workspace_cache2);

REAL8 XLALInspiralSBankComputeMatchMaxSkyLocNoPhase(const COMPLEX8FrequencySeries *hp, const COMPLEX8FrequencySeries *hc, const REAL8 hphccorr, const COMPLEX8FrequencySeries *proposal, WS *workspace_cache1, WS *workspace_cache2);

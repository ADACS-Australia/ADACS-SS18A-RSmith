/**
 * \file
 * \ingroup pulsarApps
 * \author Matthew Pitkin, John Veitch, Colin Gill
 *
 * \brief Pulsar model functions for use in parameter estimation codes for
 * targeted pulsar searches.
 */

#include "ppe_models.h"

RCSID("$Id$");

/******************************************************************************/
/*                            MODEL FUNCTIONS                                 */
/******************************************************************************/

/** \brief Defines the pulsar model/template to use
 * 
 * This function is the wrapper for functions defining the pulsar model 
 * template to be used in the analysis. It also uses \c rescale_parameter to
 * scale any parameters back to there true values for use in the model and 
 * places them into a \c BinaryPulsarParams structure.
 * 
 * Note: Any additional models should be added into this function.
 * 
 * \param data [in] The data structure hold data and current parameter info
 * 
 * \sa rescale_parameter
 * \sa pulsar_model
 */
void get_pulsar_model( LALInferenceIFOData *data ){
  BinaryPulsarParams pars;
  
  /* set model parameters (including rescaling) */
  pars.h0 = rescale_parameter( data, "h0" );
  pars.cosiota = rescale_parameter( data, "cosiota" );
  pars.psi = rescale_parameter( data, "psi" );
  pars.phi0 = rescale_parameter( data, "phi0" );
  
  /*pinned superfluid parameters*/
  pars.h1 = rescale_parameter( data, "h1" );
  pars.lambda = rescale_parameter( data, "lambda" );
  pars.theta = rescale_parameter( data, "theta" );
 
  /* set the potentially variable parameters */
  pars.pepoch = rescale_parameter( data, "pepoch" );
  pars.posepoch = rescale_parameter( data, "posepoch" );
 
  pars.ra = rescale_parameter( data, "ra" );
  pars.pmra = rescale_parameter( data, "pmra" );
  pars.dec = rescale_parameter( data, "dec" );
  pars.pmdec = rescale_parameter( data, "pmdec" );
 
  pars.f0 = rescale_parameter( data, "f0" );
  pars.f1 = rescale_parameter( data, "f1" );
  pars.f2 = rescale_parameter( data, "f2" );
  pars.f3 = rescale_parameter( data, "f3" );
  pars.f4 = rescale_parameter( data, "f4" );
  pars.f5 = rescale_parameter( data, "f5" );
  
  /* binary system model - NOT pulsar model */
  pars.model = *(CHAR**)LALInferenceGetVariable( data->modelParams, "model" );

  /* binary parameters */
  if( pars.model != NULL ){
    pars.e = rescale_parameter( data, "e" );
    pars.w0 = rescale_parameter( data, "w0" );
    pars.Pb = rescale_parameter( data, "Pb" );
    pars.x = rescale_parameter( data, "x" );
    pars.T0 = rescale_parameter( data, "T0" );
    
    pars.e2 = rescale_parameter( data, "e2" );
    pars.w02 = rescale_parameter( data, "w02" );
    pars.Pb2 = rescale_parameter( data, "Pb2" );
    pars.x2 = rescale_parameter( data, "x2" );
    pars.T02 = rescale_parameter( data, "T02" );
   
    pars.e3 = rescale_parameter( data, "e3" );
    pars.w03 = rescale_parameter( data, "w03" );
    pars.Pb3 = rescale_parameter( data, "Pb3" );
    pars.x3 = rescale_parameter( data, "x3" );
    pars.T03 = rescale_parameter( data, "T03" );
    
    pars.xpbdot = rescale_parameter( data, "xpbdot" );
    pars.eps1 = rescale_parameter( data, "eps1" );
    pars.eps2 = rescale_parameter( data, "eps2" );
    pars.eps1dot = rescale_parameter( data, "eps1dot" );
    pars.eps2dot = rescale_parameter( data, "eps2dot" );
    pars.Tasc = rescale_parameter( data, "Tasc" );
   
    pars.wdot = rescale_parameter( data, "wdot" );
    pars.gamma = rescale_parameter( data, "gamma" );
    pars.Pbdot = rescale_parameter( data, "Pbdot" );
    pars.xdot = rescale_parameter( data, "xdot" );
    pars.edot = rescale_parameter( data, "edot" );
   
    pars.s = rescale_parameter( data, "s" );
    pars.dr = rescale_parameter( data, "dr" );
    pars.dth = rescale_parameter( data, "dth" );
    pars.a0 = rescale_parameter( data, "a0" );
    pars.b0 = rescale_parameter( data, "b0" ); 

    pars.M = rescale_parameter( data, "M" );
    pars.m2 = rescale_parameter( data, "m2" );
  }

  /* now get pulsar model */
  pulsar_model( pars, data );
    
}


/** \brief Rescale parameter back to its true value
 * 
 * This function will rescale a parameter to its true value using the scale
 * factor and minimum scale value.
 * 
 * \param data [in] data structure containing parameter information
 * \param parname [in] name of the parameter requiring rescaling
 * 
 * \return Rescaled parameter value
 */
REAL8 rescale_parameter( LALInferenceIFOData *data, const CHAR *parname ){
  REAL8 par = 0., scale = 0., offset = 0.;
  CHAR scaleName[VARNAME_MAX] = "";
  CHAR offsetName[VARNAME_MAX] = "";
  
  sprintf(scaleName, "%s_scale", parname);
  sprintf(offsetName, "%s_scale_min", parname);
  
  scale = *(REAL8*)LALInferenceGetVariable( data->dataParams, scaleName );
  offset = *(REAL8*)LALInferenceGetVariable( data->dataParams, offsetName );
  
  par = *(REAL8*)LALInferenceGetVariable( data->modelParams, parname );
  
  par = par*scale + offset;
  
  return par;
}


/** \brief Generate the model of the neutron star signal
 *
 * The function requires that the pulsar model is set using the \c model-type
 * command line argument (this is set in \c main, and if not specified defaults
 * to a \c triaxial model). Currently the model can be \c triaxial for
 * quadrupole emission from a triaxial star at twice the rotation freqeuncy, or
 * \c pinsf for a two component emission model with emission at the rotation
 * frequency <i>and</i> twice the rotation frequency. Depending on the specified
 * model the function calls the appropriate model function.
 * 
 * Firstly the time varying amplitude of the signal will be calculated based on 
 * the antenna pattern and amplitude parameters. Then, if searching over phase 
 * parameters, the phase evolution of the signal will be calculated. The
 * difference between the new phase model, \f$\phi(t)_n\f$, and that used to
 * heterodyne the data, \f$\phi(t)_h\f$, (stored in \c data->timeData->data)
 * will be calculated and the complex signal model, \f$M\f$, modified
 * accordingly:
 * \f[
 * M'(t) = M(t)\exp{i(-(\phi(t)_n - \phi(t)_h))}. 
 * \f]
 * 
 * \param params [in] A \c BinaryPulsarParams structure containing the model
 * parameters
 * \param data [in] The data structure containing the detector data and
 * additional info
 * 
 * \sa get_triaxial_amplitude_model
 * \sa get_pinsf_amplitude_model
 * \sa get_phase_model
 */
void pulsar_model( BinaryPulsarParams params, 
                   LALInferenceIFOData *data ){
  INT4 i = 0, length = 0;
  UINT4 j = 0;
  CHAR *modeltype = NULL;
  
  /* check model type to get amplitude model */
  modeltype = *(CHAR**)LALInferenceGetVariable( data->dataParams, "modeltype" );
  
  if ( !strcmp( modeltype, "triaxial" ) ){
    get_triaxial_amplitude_model( params, data );
  }
  else if ( !strcmp( modeltype, "pinsf" ) ){
    get_pinsf_amplitude_model( params, data );
  }
  /* ADD NEW MODELS HERE */
  else{
    fprintf(stderr, "Error... model '%s' is not defined!\n", modeltype);
    exit(0);
  }
   
  /* get difference in phase for f component and perform extra heterodyne */
  REAL8Vector *freqFactors = NULL;
  freqFactors = *(REAL8Vector **)LALInferenceGetVariable( data->dataParams,
                                                          "freqfactors" );
  
  for( j = 0; j < freqFactors->length; j++ ){
    REAL8Vector *dphi = NULL;
    
    /* move data pointer along one as one iteration of the model is held over
    j data structures, moved this from bottom of loop so actioned for 2nd run
through the loop.*/
    if ( j > 0 ){ 
      data = data->next;
    }
    
    length = data->compModelData->data->length;
    /* the timeData vector within the LALIFOData structure contains the
     phase calculated using the initial (heterodyne) values of the phase
     parameters */
    if ( varyphase ){
      if ( (dphi = get_phase_model( params, data, 
        freqFactors->data[j] )) != NULL ){
        for( i=0; i<length; i++ ){
          COMPLEX16 M;
          REAL8 dphit;
          REAL4 sp, cp;
    
          dphit = -fmod(dphi->data[i] - data->timeData->data->data[i], 1.);
    
          sin_cos_2PI_LUT( &sp, &cp, dphit );
    
          M.re = data->compModelData->data->data[i].re;
          M.im = data->compModelData->data->data[i].im;
    
          /* heterodyne */
          data->compModelData->data->data[i].re = M.re*cp - M.im*sp;
          data->compModelData->data->data[i].im = M.im*cp + M.re*sp;
        }
      }
    }
    XLALDestroyREAL8Vector( dphi );
  }
}


/** \brief The phase evolution of a source
 *
 * This function will calculate the phase evolution of a source at a particular
 * sky location as observed at Earth. The phase evolution is described by a 
 * Taylor expansion:
 * \f[
 * \phi(T) = \sum_{k=1}^n \frac{f^{(k-1)}{k!} T^k,
 * \f]
 * where \f$f^x\f$ is the xth time derivative of the gravitational wave
 * frequency, and \f$T\f$ is the pulsar proper time. Frequency time derivatives
 * are currently allowed up to the fifth derivative. The pulsar proper time is 
 * calculated by correcting the time of arrival at Earth, \f$t\f$ to the solar
 * system barycentre and if necessary the binary system barycenter, so \f$T =
 * t + \delta{}t_{\rm SSB} + \delta{}t_{\rm BSB}\f$.
 * 
 * In this function the time delay caused needed to correct to the solar system
 * barycenter is only calculated if required i.e. if it's not been previously
 * calculated and an update is required due to a change in the sky position. The
 * same is true for the binary system time delay, which is only calculated if
 * it has not previously been obtained or needs updating due to a change in the
 * binary system parameters.
 * 
 * The solar system barycentre delay does not have to be explicitly computed
 * for every time stamp passed to it, but instead will just use linear
 * interpolation within a time range set by \c interptime.
 * 
 * \param params [in] A set of pulsar parameters
 * \param data [in] The data structure containing the detector data and
 * additional info
 * \param freqFactor [in] the multiplicative factor on the pulsar frequency for
 * a particular model
 * 
 * \return A vector of rotational phase values
 * 
 * \sa get_ssb_delay
 * \sa get_bsb_delay
 */
REAL8Vector *get_phase_model( BinaryPulsarParams params, 
                              LALInferenceIFOData *data,
                              REAL8 freqFactor ){
  INT4 i = 0, length = 0;

  REAL8 T0 = 0., DT = 0., deltat = 0., deltat2 = 0.;
  REAL8 interptime = 1800.; /* calulate every 30 mins (1800 secs) */
  
  REAL8Vector *phis = NULL, *dts = NULL, *bdts = NULL;
  
  

  /* if edat is NULL then return a NULL poniter */
  if( data->ephem == NULL )
    return NULL;

  length = data->dataTimes->length;
  
  /* allocate memory for phases */
  phis = XLALCreateREAL8Vector( length );

  /* get time delays */ 
  if( (dts = *(REAL8Vector **)LALInferenceGetVariable( data->dataParams,
      "ssb_delays" )) == NULL || varyskypos == 1 ){
    /* get time delays with an interpolation of interptime (30 mins) */
    dts = get_ssb_delay( params, data->dataTimes, data->ephem, data->detector,
                         interptime );
  }
  
  if( (bdts = *(REAL8Vector **)LALInferenceGetVariable( data->dataParams,
      "bsb_delays" )) == NULL || varybinary == 1 ){
    /* get binary system time delays */
    bdts = get_bsb_delay( params, data->dataTimes, dts );
  }
  
  for( i=0; i<length; i++){
    REAL8 realT = XLALGPSGetREAL8( &data->dataTimes->data[i] );
    
    T0 = params.pepoch;

    DT = realT - T0;

    if ( params.model != NULL )
      deltat = DT + dts->data[i] + bdts->data[i];
    else
      deltat = DT + dts->data[i];
    
    /* work out phase */
    deltat2 = deltat*deltat;
    phis->data[i] = freqFactor*deltat*(params.f0 + 
      inv_fact[2]*params.f1*deltat +
      inv_fact[3]*params.f2*deltat2 +
      inv_fact[4]*params.f3*deltat*deltat2 +
      inv_fact[5]*params.f4*deltat2*deltat2 +
      inv_fact[6]*params.f5*deltat2*deltat2*deltat);
  }
  return phis;
}


/** \brief Computes the delay between a GPS time at Earth and the solar system 
 * barycentre
 *
 * This function calculate the time delay between a GPS time at a specific 
 * location (e.g. a gravitational wave detector) on Earth and the solar system
 * barycentre. The delay consists of three components: the geometric time delay
 * (Roemer delay) \f$t_R = \mathbf{r}(t)\hat{n}/c\f$ (where \f$\mathbf{r}(t)\f$
 * is the detector's position vector at time \f$t\f$), the special relativistic
 * Einstein delay \f$t_E\f$, and the general relativistic Shapiro delay
 * \f$t_S\f$.
 * 
 * Rather than computing the time delay at every time stamp passed to the
 * function it is instead (if requested) able to perform linear interpolation
 * to a point within a range given by \c interptime. 
 *  
 * \param pars [in] A set of pulsar parameters
 * \param datatimes [in] A vector of GPS times at Earth
 * \param ephem [in] Information on the solar system ephemeris
 * \param detector [in] Information on the detector position on the Earth
 * \param interptime [in] The time (in seconds) between explicit recalculations
 * of the time delay
 * 
 * \return A vector of time delays in seconds
 *
 * \sa LALBarycenter
 * \sa LALBarycenterEarth
 */
REAL8Vector *get_ssb_delay( BinaryPulsarParams pars, 
                            LIGOTimeGPSVector *datatimes,
                            EphemerisData *ephem,
                            LALDetector *detector,
                            REAL8 interptime ){
  static LALStatus status;

  INT4 i = 0, length = 0;

  REAL8 T0 = 0., DT = 0., DTplus = 0.;

  EarthState earth, earth2;
  EmissionTime emit, emit2;

  BarycenterInput *bary = NULL;
  
  REAL8Vector *dts = NULL;

  /* if edat is NULL then return a NULL poniter */
  if( ephem == NULL )
    return NULL;
  
  /* copy barycenter and ephemeris data */
  bary = (BarycenterInput*)XLALCalloc( 1, sizeof(BarycenterInput) );
  memcpy( &bary->site, detector, sizeof(LALDetector) );
  
  bary->alpha = pars.ra;
  bary->delta = pars.dec;
  
   /* set the position and frequency epochs if not already set */
  if( pars.pepoch == 0. && pars.posepoch != 0.)
    pars.pepoch = pars.posepoch;
  else if( pars.posepoch == 0. && pars.pepoch != 0. )
    pars.posepoch = pars.pepoch;

  length = datatimes->length;
  
  /* allocate memory for times delays */
  dts = XLALCreateREAL8Vector( length );
 
  /* set 1/distance if parallax or distance value is given (1/sec) */
  if( pars.px != 0. )
    bary->dInv = pars.px*1e-3*LAL_C_SI/LAL_PC_SI;
  else if( pars.dist != 0. )
    bary->dInv = LAL_C_SI/(pars.dist*1e3*LAL_PC_SI);
  else
    bary->dInv = 0.;
  
  for( i=0; i<length; i++){
    REAL8 realT = XLALGPSGetREAL8( &datatimes->data[i] );
    
    T0 = pars.pepoch;

    DT = realT - T0;

    /* only do call to the barycentring routines once every interptime (unless
       interptime == 0), otherwise just linearly interpolate between them */
    if( i == 0 || DT > DTplus || interptime == 0 ){
      bary->tgps = datatimes->data[i];

      bary->delta = pars.dec + (realT-pars.posepoch) * pars.pmdec;
      bary->alpha = pars.ra + (realT-pars.posepoch) *
         pars.pmra/cos(bary->delta);
     
      /* call barycentring routines */
      LAL_CALL( LALBarycenterEarth( &status, &earth, &bary->tgps, ephem ),
                &status );
      
      LAL_CALL( LALBarycenter( &status, &emit, bary, &earth ), &status );

      /* add interptime to the time */
      if ( interptime > 0 ){
        DTplus = DT + interptime;
        XLALGPSAdd( &bary->tgps, interptime );

        /* No point in updating the positions as difference will be tiny */
        LAL_CALL( LALBarycenterEarth( &status, &earth2, &bary->tgps, ephem ),
                  &status );
        LAL_CALL( LALBarycenter( &status, &emit2, bary, &earth2), &status );
      }
    }

    /* linearly interpolate to get emitdt */
    if( interptime > 0. ){
      dts->data[i] = emit.deltaT + (DT - (DTplus - interptime)) *
        (emit2.deltaT - emit.deltaT)/interptime;
    }
    else
      dts->data[i] = emit.deltaT;
  }
  
  XLALFree( bary );
  
  return dts;
}


/** \brief Computes the delay between a pulsar in a binary system and the
 * barycentre of the system
 *
 * This function uses \c XLALBinaryPulsarDeltaT to calculate the time delay
 * between for a pulsar in a binary system between the time at the pulsar and
 * the time at the barycentre of the system. This includes Roemer delays and
 * relativistic delays. The orbit may be described by different models and can
 * be purely Keplarian or include various relativistic corrections.
 *
 * \param pars [in] A set of pulsar parameters
 * \param datatimes [in] A vector of GPS times
 * \param dts [in] A vector of solar system barycentre time delays
 * 
 * \return A vector of time delays in seconds
 * 
 * \sa XLALBinaryPulsarDeltaT
 */
REAL8Vector *get_bsb_delay( BinaryPulsarParams pars,
                            LIGOTimeGPSVector *datatimes,
                            REAL8Vector *dts ){
  BinaryPulsarInput binput;
  BinaryPulsarOutput boutput;
  REAL8Vector *bdts = NULL;
  
  INT4 i = 0, length = datatimes->length;
  
  bdts = XLALCreateREAL8Vector( length );
  
  for ( i = 0; i < length; i++ ){
    binput.tb = XLALGPSGetREAL8( &datatimes->data[i] ) + dts->data[i];
  
    XLALBinaryPulsarDeltaT( &boutput, &binput, &pars );
    
    bdts->data[i] = boutput.deltaT;
  }
  
  return bdts;
}


/** \brief The amplitude model of a complex heterodyned traxial neutron star
 * 
 * This function calculates the complex heterodyned time series model for a 
 * triaxial neutron star (see [\ref DupuisWoan2005]). It is defined as:
 * \f{eqnarray*}{
 * y(t) & = & \frac{h_0}{2} \left( \frac{1}{2}F_+(t,\psi)
 * (1+\cos^2\iota)\cos{\phi_0} + F_{\times}(t,\psi)\cos{\iota}\sin{\phi_0}
 * \right) + \\
 *  & & i\frac{h_0}{2}\left( \frac{1}{2}F_+(t,\psi)
 * (1+\cos^2\iota)\sin{\phi_0} - F_{\times}(t,\psi)\cos{\iota}\cos{\phi_0}
 * \right),
 * \f}
 * where \f$F_+\f$ and \f$F_{\times}\f$ are the antenna response functions for
 * the plus and cross polarisations.
 * 
 * The antenna pattern functions are contained in a 2D lookup table, so within
 * this function the correct value for the given time and \f$\psi\f$ are
 * interpolated from this lookup table using bilinear interpolation (e.g.):
 * \f{eqnarray*}{
 * F_+(\psi, t) = F_+(\psi_i, t_j)(1-\psi)(1-t) + F_+(\psi_{i+1}, t_j)\psi(1-t)
 * + F_+(\psi_i, t_{j+1})(1-\psi)t + F_+(\psi_{i+1}, t_{j+1})\psi{}t,
 * \f}
 * where \f$\psi\f$ and \f$t\f$ have been scaled to be within a unit square,
 * and \f$\psi_i\f$ and \f$t_j\f$ are the closest points within the lookup
 * table to the required values.
 * 
 * \param pars [in] A set of pulsar parameters
 * \param data [in] The data parameters giving information on the data and
 * detector
 * 
 */
void get_triaxial_amplitude_model( BinaryPulsarParams pars, 
                                   LALInferenceIFOData *data ){
  INT4 i = 0, length;
  
  REAL8 psteps, tsteps, psv, tsv;
  INT4 psibinMin, psibinMax, timebinMin, timebinMax;
  REAL8 plus, cross;
  REAL8 plus00, plus01, plus10, plus11, cross00, cross01, cross10, cross11;
  REAL8 psiScaled, timeScaled;
  REAL8 psiMin, psiMax, timeMin, timeMax;
  REAL8 T;
  REAL8 Xplus, Xcross;
  REAL8 Xpcosphi, Xccosphi, Xpsinphi, Xcsinphi;
  REAL4 sinphi, cosphi;
  
  gsl_matrix *LU_Fplus, *LU_Fcross;
  REAL8Vector *sidDayFrac = NULL;
  
  length = data->dataTimes->length;
  
  /* set lookup table parameters */
  psteps = *(INT4*)LALInferenceGetVariable( data->dataParams, "psiSteps" );
  tsteps = *(INT4*)LALInferenceGetVariable( data->dataParams, "timeSteps" );
  
  LU_Fplus = *(gsl_matrix**)LALInferenceGetVariable( data->dataParams, 
                                                     "LU_Fplus");
  LU_Fcross = *(gsl_matrix**)LALInferenceGetVariable( data->dataParams, 
                                                      "LU_Fcross");
  /* get the sidereal time since the initial data point % sidereal day */
  sidDayFrac = *(REAL8Vector**)LALInferenceGetVariable( data->dataParams,
                                                        "siderealDay" );
  
  sin_cos_LUT( &sinphi, &cosphi, pars.phi0 );
  
  /************************* CREATE MODEL *************************************/
  /* This model is a complex heterodyned time series for a triaxial neutron
     star emitting at twice its rotation frequency (as defined in Dupuis and
     Woan, PRD, 2005):
       real = (h0/2) * ((1/2)*F+*(1+cos(iota)^2)*cos(phi0) 
         + Fx*cos(iota)*sin(phi0))
       imag = (h0/2) * ((1/2)*F+*(1+cos(iota)^2)*sin(phi0)
         - Fx*cos(iota)*cos(phi0))
   ****************************************************************************/
  
  Xplus = 0.25*(1.+pars.cosiota*pars.cosiota)*pars.h0;
  Xcross = 0.5*pars.cosiota*pars.h0;
  Xpsinphi = Xplus*sinphi;
  Xcsinphi = Xcross*sinphi;
  Xpcosphi = Xplus*cosphi;
  Xccosphi = Xcross*cosphi;
  
  /* set the psi bin for the lookup table */
  psv = LAL_PI_2 / ( psteps - 1. );
  psibinMin = (INT4)floor( ( pars.psi + LAL_PI/4. )/psv );
  psiMin = -(LAL_PI/4.) + psibinMin*psv;
  psibinMax = psibinMin + 1;
  psiMax = psiMin + psv;
  
  /* rescale psi for bilinear interpolation on a unit square */
  psiScaled = (pars.psi - psiMin)/(psiMax - psiMin);
  
  tsv = LAL_DAYSID_SI / tsteps;
  
  for( i=0; i<length; i++ ){
    /* set the time bin for the lookup table */
    /* sidereal day in secs*/
    T = sidDayFrac->data[i];
    timebinMin = (INT4)fmod( floor(T / tsv), tsteps );
    timeMin = timebinMin*tsv;
    timebinMax = (INT4)fmod( timebinMin + 1, tsteps );
    timeMax = timeMin + tsv;
    
    /* get values of matrix for bilinear interpolation */
    plus00 = gsl_matrix_get( LU_Fplus, psibinMin, timebinMin );
    plus01 = gsl_matrix_get( LU_Fplus, psibinMin, timebinMax );
    plus10 = gsl_matrix_get( LU_Fplus, psibinMax, timebinMin );
    plus11 = gsl_matrix_get( LU_Fplus, psibinMax, timebinMax );
    
    cross00 = gsl_matrix_get( LU_Fcross, psibinMin, timebinMin );
    cross01 = gsl_matrix_get( LU_Fcross, psibinMin, timebinMax );
    cross10 = gsl_matrix_get( LU_Fcross, psibinMax, timebinMin );
    cross11 = gsl_matrix_get( LU_Fcross, psibinMax, timebinMax );
    
    /* rescale time for bilinear interpolation on a unit square */
    timeScaled = (T - timeMin)/(timeMax - timeMin);
    
    plus = plus00*(1. - psiScaled)*(1. - timeScaled) + 
      plus10*psiScaled*(1. - timeScaled) + plus01*(1. - psiScaled)*timeScaled +
      plus11*psiScaled*timeScaled;
    cross = cross00*(1. - psiScaled)*(1. - timeScaled) + 
      cross10*psiScaled*(1. - timeScaled) + cross01*(1. - psiScaled)*timeScaled
      + cross11*psiScaled*timeScaled;
    
    /* create the complex signal amplitude model */
    data->compModelData->data->data[i].re = plus*Xpcosphi + cross*Xcsinphi;
    data->compModelData->data->data[i].im = plus*Xpsinphi - cross*Xccosphi;
  }
}


void get_pinsf_amplitude_model( BinaryPulsarParams pars, LALInferenceIFOData
*data ){
  INT4 i = 0, length;
  
  REAL8 psteps, tsteps, psv, tsv;
  INT4 psibinMin, psibinMax, timebinMin, timebinMax;
  REAL8 plus00, plus01, plus10, plus11, cross00, cross01, cross10, cross11;
  REAL8 psiScaled, timeScaled;
  REAL8 psiMin, psiMax, timeMin, timeMax;
  REAL8 plus, cross;
  REAL8 T;
  REAL8 Xplusf, Xcrossf, Xplus2f, Xcross2f;
  REAL8 A1, A2, B1, B2;
  REAL4 sinphi, cosphi, sin2phi, cos2phi;
  
  gsl_matrix *LU_Fplus, *LU_Fcross;
  REAL8Vector *sidDayFrac = NULL;
  
  /* set lookup table parameters */
  psteps = *(INT4*)LALInferenceGetVariable( data->dataParams, "psiSteps" );
  tsteps = *(INT4*)LALInferenceGetVariable( data->dataParams, "timeSteps" );
  
  LU_Fplus = *(gsl_matrix**)LALInferenceGetVariable( data->dataParams,
"LU_Fplus");
  LU_Fcross = *(gsl_matrix**)LALInferenceGetVariable( data->dataParams,
"LU_Fcross");
  /* get the sidereal time since the initial data point % sidereal day */
  sidDayFrac = *(REAL8Vector**)LALInferenceGetVariable( data->dataParams,
                                                        "siderealDay" );
  
  sin_cos_LUT( &sinphi, &cosphi, 0.5*pars.phi0 );
  sin_cos_LUT( &sin2phi, &cos2phi, pars.phi0 );
  
  /************************* CREATE MODEL *************************************/
  /* This model is a complex heterodyned time series for a pinned superfluid
neutron
     star emitting at its roation frequency and twice its rotation frequency 
     (as defined in Jones 2009):

   ****************************************************************************/
  
  Xplusf = 0.125*sin(acos(pars.cosiota))*pars.cosiota*pars.h0;
  Xcrossf = 0.125*sin(acos(pars.cosiota))*pars.h0;
  Xplus2f = 0.25*(1.+pars.cosiota*pars.cosiota)*pars.h0;
  Xcross2f = 0.5*pars.cosiota*pars.h0;
  A1=( (cos(pars.lambda)*cos(pars.lambda)) - pars.h1 ) * (sin( (2*pars.theta)
));
  A2=sin(2*pars.lambda)*sin(pars.theta);
  B1=( (cos(pars.lambda)*cos(pars.lambda))*(cos(pars.theta)*cos(pars.theta)) ) -
(sin(pars.lambda)*sin(pars.lambda)) 
    + ( pars.h1*(sin(pars.theta)*sin(pars.theta)) );
  B2=sin(2*pars.lambda)*cos(pars.theta);
  
  /* set the psi bin for the lookup table */
  psv = LAL_PI_2 / ( psteps - 1. );
  psibinMin = (INT4)floor( ( pars.psi + LAL_PI/4. )/psv );
  psiMin = -(LAL_PI/4.) + psibinMin*psv;
  psibinMax = psibinMin + 1;
  psiMax = psiMin + psv;
  
  /* rescale psi for bilinear interpolation on a unit square */
  psiScaled = (pars.psi - psiMin)/(psiMax - psiMin);
  
  tsv = LAL_DAYSID_SI / tsteps;
  
  /* set model for 1f component */
  length = data->dataTimes->length;
  
  for( i=0; i<length; i++ ){
    /* set the time bin for the lookup table */
    /* sidereal day in secs*/    
    T = sidDayFrac->data[i];
    timebinMin = (INT4)fmod( floor(T / tsv), tsteps );
    timeMin = timebinMin*tsv;
    timebinMax = (INT4)fmod( timebinMin + 1, tsteps );
    timeMax = timeMin + tsv;
    
    /* get values of matrix for bilinear interpolation */
    plus00 = gsl_matrix_get( LU_Fplus, psibinMin, timebinMin );
    plus01 = gsl_matrix_get( LU_Fplus, psibinMin, timebinMax );
    plus10 = gsl_matrix_get( LU_Fplus, psibinMax, timebinMin );
    plus11 = gsl_matrix_get( LU_Fplus, psibinMax, timebinMax );
    
    cross00 = gsl_matrix_get( LU_Fcross, psibinMin, timebinMin );
    cross01 = gsl_matrix_get( LU_Fcross, psibinMin, timebinMax );
    cross10 = gsl_matrix_get( LU_Fcross, psibinMax, timebinMin );
    cross11 = gsl_matrix_get( LU_Fcross, psibinMax, timebinMax );
    
    /* rescale time for bilinear interpolation on a unit square */
    timeScaled = (T - timeMin)/(timeMax - timeMin);
    
    plus = plus00*(1. - psiScaled)*(1. - timeScaled) + 
      plus10*psiScaled*(1. - timeScaled) + plus01*(1. - psiScaled)*timeScaled +
      plus11*psiScaled*timeScaled;
    cross = cross00*(1. - psiScaled)*(1. - timeScaled) + 
      cross10*psiScaled*(1. - timeScaled) + cross01*(1. - psiScaled)*timeScaled
      + cross11*psiScaled*timeScaled;
    
    /* create the complex signal amplitude model */
    /*at f*/
    data->compModelData->data->data[i].re =
plus*Xplusf*((A1*cosphi)-(A2*sinphi)) + 
    ( cross*Xcrossf*((A2*cosphi)-(A1*sinphi)) );
    
    data->compModelData->data->data[i].im =
plus*Xplusf*((A2*cosphi)+(A1*sinphi)) + 
    ( cross*Xcrossf*((A2*sinphi)-(A1*cosphi)) );

  }
  
  /* set model for 2f component */
  length = data->next->dataTimes->length;
  
  sidDayFrac = *(REAL8Vector**)LALInferenceGetVariable( data->next->dataParams,
                                                        "siderealDay" );
  
  for( i=0; i<length; i++ ){
    /* set the time bin for the lookup table */
    /* sidereal day in secs*/    
    T = sidDayFrac->data[i];
    timebinMin = (INT4)fmod( floor(T / tsv), tsteps );
    timeMin = timebinMin*tsv;
    timebinMax = (INT4)fmod( timebinMin + 1, tsteps );
    timeMax = timeMin + tsv;
    
    /* get values of matrix for bilinear interpolation */
    plus00 = gsl_matrix_get( LU_Fplus, psibinMin, timebinMin );
    plus01 = gsl_matrix_get( LU_Fplus, psibinMin, timebinMax );
    plus10 = gsl_matrix_get( LU_Fplus, psibinMax, timebinMin );
    plus11 = gsl_matrix_get( LU_Fplus, psibinMax, timebinMax );
    
    cross00 = gsl_matrix_get( LU_Fcross, psibinMin, timebinMin );
    cross01 = gsl_matrix_get( LU_Fcross, psibinMin, timebinMax );
    cross10 = gsl_matrix_get( LU_Fcross, psibinMax, timebinMin );
    cross11 = gsl_matrix_get( LU_Fcross, psibinMax, timebinMax );
    
    /* rescale time for bilinear interpolation on a unit square */
    timeScaled = (T - timeMin)/(timeMax - timeMin);
    
    plus = plus00*(1. - psiScaled)*(1. - timeScaled) + 
      plus10*psiScaled*(1. - timeScaled) + plus01*(1. - psiScaled)*timeScaled +
      plus11*psiScaled*timeScaled;
    cross = cross00*(1. - psiScaled)*(1. - timeScaled) + 
      cross10*psiScaled*(1. - timeScaled) + cross01*(1. - psiScaled)*timeScaled
      + cross11*psiScaled*timeScaled;
    
    /* create the complex signal amplitude model at 2f*/
    data->next->compModelData->data->data[i].re =
      plus*Xplus2f*((B1*cos2phi)-(B2*sin2phi)) +
      cross*Xcross2f*((B2*cos2phi)+(B1*sin2phi));
    
    data->next->compModelData->data->data[i].im =
      plus*Xplus2f*((B2*cos2phi)+(B1*sin2phi)) -
      cross*Xcross2f*((B1*cos2phi)-(B2*sin2phi));
  }
  
}


/** \brief Calculate the natural logarithm of the evidence that the data
 * consists of only Gaussian noise
 * 
 * The function will calculate the natural logarithm of the evidence that the
 * data (from one or more detectors) consists of stationary segments/chunks 
 * describe by a Gaussian with zero mean and unknown variance.
 * 
 * The evidence is obtained from the joint likelihood given in \c
 * pulsar_log_likelihood with the model term \f$y\f$ set to zero.
 * 
 * \param data [in] Structure containing detector data
 * 
 * \return The natural logarithm of the noise only evidence
 */
REAL8 noise_only_model( LALInferenceIFOData *data ){
  LALInferenceIFOData *datatemp = data;
  
  REAL8 logL = 0.0;
  UINT4 i = 0;
  
  while ( datatemp ){
    UINT4Vector *chunkLengths = NULL;
    REAL8Vector *sumDat = NULL;
  
    REAL8 chunkLength = 0.;
  
    chunkLengths = *(UINT4Vector **)LALInferenceGetVariable( data->dataParams, 
                                                             "chunkLength" );
    sumDat = *(REAL8Vector **)LALInferenceGetVariable( data->dataParams,
                                                        "sumData" );
  
    for (i=0; i<chunkLengths->length; i++){
      chunkLength = (REAL8)chunkLengths->data[i];
   
      logL -= chunkLength * log(sumDat->data[i]);
    }
  
    datatemp = datatemp->next;
  }
  
  return logL;
}

/*------------------------ END OF MODEL FUNCTIONS ----------------------------*/

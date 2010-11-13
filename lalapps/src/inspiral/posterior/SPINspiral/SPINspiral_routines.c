/* 
   
   SPINspiral:                parameter estimation on binary inspirals detected by LIGO, including spins of the binary members
   SPINspiral_routines.c:     miscellaneous routines
   
   
   Copyright 2007, 2008, 2009 Christian Roever, Marc van der Sluys, Vivien Raymond, Ilya Mandel
   
   
   This file is part of SPINspiral.
   
   SPINspiral is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
   
   SPINspiral is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.
   
   You should have received a copy of the GNU General Public License
   along with SPINspiral.  If not, see <http://www.gnu.org/licenses/>.
   
*/



#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <SPINspiral.h>


/**
 * \file SPINspiral_routines.c
 * \brief Contains various routines
 */



//*** MASSES ***//

// ****************************************************************************************************************************************************  
/**
 * \brief Compute symmetric mass ratio (eta) for given individual masses
 */
double massRatio(double m1, double m2) 
{
  return (m1*m2)/pow(m1+m2,2.0);
}
// ****************************************************************************************************************************************************  



// ****************************************************************************************************************************************************  
/**
 * \brief Compute chirp mass (mM) for given individual masses
 */
double chirpMass(double m1, double m2)
{
  return pow(m1*m2,0.6) / pow(m1+m2,0.2);
}
// ****************************************************************************************************************************************************  



// ****************************************************************************************************************************************************  
/**
 * \brief Compute individual masses (m1,m2) for given chirp mass (Mc) and symmetric mass ratio (eta)
 */
void McEta2masses(double Mc, double eta, double *m1, double *m2)
{
  if(eta<=0.25) {
    double Mtot = Mc*exp(-0.6*log(eta));
    double tvar = sqrt(1.0-4.0*eta);
    *m1 = Mtot/2.0 * (1.0 + tvar);
    *m2 = Mtot/2.0 * (1.0 - tvar);
  } else {                                     //Allow 0.25<eta<0.50 (for eta>0.50, m1<0 in this definition
    double eta1 = 0.5 - eta;
    double Mtot = Mc*exp(-0.6*log(eta1));
    double tvar = sqrt(1.0-4.0*eta1);
    *m1 = Mtot/2.0 * (1.0 - tvar);
    *m2 = Mtot/2.0 * (1.0 + tvar);
  }
}
// ****************************************************************************************************************************************************  



// ****************************************************************************************************************************************************  
/**
 * \brief Compute chirp mass (Mc) and symmetric mass ratio (eta) for given individual masses (m1,m2)
 */
void masses2McEta(double m1, double m2, double *Mc, double *eta)
{
  double Mtot = m1+m2;
  *eta = (m1*m2)/(Mtot*Mtot);
  *Mc = Mtot*exp(0.6*log(*eta));
}
// ****************************************************************************************************************************************************  




//*** GMT and RIGHT ASCENSION ***

// ****************************************************************************************************************************************************  
/**
 * \brief Compute Greenwich Mean Sideral time from GPS time.
 * 
 * Computes the Greenwich Mean Sidereal Time (in radians!) from GPS time (in seconds).
 * See K.R. Lang (1999), p.80sqq.
 */
// ****************************************************************************************************************************************************  
double GMST(double GPSsec)
{
  double GPS_Jan1st2000midnight    = 630720013.0;
  double leapseconds = 32.0;  // at Jan 1st 2000
  double seconds, days, centuries, secCurrentDay, result;
  if(GPSsec < GPS_Jan1st2000midnight) fprintf(stderr, "\n ***  Warning: GMSTs before 1/1/2000 are inaccurate (GPS time: %lf) ***\n\n",GPSsec);
  if(GPSsec > 820108813.0) leapseconds += 1.0; // Leap second after 2005/'06
  if(GPSsec > 914803214.0) leapseconds += 1.0; // Leap second after 2008/'09
  
  // Time since Jan 1st 2000 (0:00h):
  seconds       = (GPSsec - GPS_Jan1st2000midnight) + (leapseconds - 32.0);
  days          = floor(seconds/86400.0) - 0.5;
  secCurrentDay = fmod(seconds, 86400.0);
  centuries     = days / 36525.0;
  result  = 24110.54841+(centuries*(8640184.812866+centuries*(0.093104+centuries*6.2e-6)));
  result += secCurrentDay * 1.002737909350795; // UTC day is 1.002 * MST day
  result  = fmod(result/86400.0,1.0);
  result *= 2.0*pi;
  return result;
}
// ****************************************************************************************************************************************************  



// ****************************************************************************************************************************************************  
/**
 * \brief Compute RA from 'longitude' and GMST
 * 
 * All quantities are in radians
 */
double rightAscension(double longi, double gmst)
{
  return fmod(longi + gmst + mtpi,tpi);  //Bring it between 0 and 2pi
}
// ****************************************************************************************************************************************************  



// ****************************************************************************************************************************************************  
/**
 * \brief Compute 'longitude' from RA and GMST
 * 
 * Computes the 'longitude' from the right ascension and GMST.
 * All quantities are in radians.
 * In fact, 'longitude' is something like the Greenwich hour angle of the corresponding RA.
 */
double longitude(double ra, double gmst)
{
  return fmod(ra - gmst + mtpi,tpi);
}
// ****************************************************************************************************************************************************  






//*** VECTORS ***


// ****************************************************************************************************************************************************  
/**
 * \brief Compute the dot product of two vectors
 */
double dotProduct(double vec1[3], double vec2[3])
{
  return vec1[0]*vec2[0] + vec1[1]*vec2[1] + vec1[2]*vec2[2];
}
// ****************************************************************************************************************************************************  



// ****************************************************************************************************************************************************  
/**
 * \brief Multiply a vector by a factor (scalar): vec2 = fac*vec1
 */
void facVec(double vec1[3], double fac, double vec2[3])
{
  vec2[0] = fac*vec1[0];
  vec2[1] = fac*vec1[1];
  vec2[2] = fac*vec1[2];
}
// ****************************************************************************************************************************************************  




// ****************************************************************************************************************************************************  
/**
 * \brief Add two vectors result = vec1 + vec2
 */
void addVec(double vec1[3], double vec2[3], double result[3])
{
  result[0] = vec1[0] + vec2[0];
  result[1] = vec1[1] + vec2[1];
  result[2] = vec1[2] + vec2[2];
}
// ****************************************************************************************************************************************************  





// ****************************************************************************************************************************************************  
/**
 * \brief Normalise a vector:  nvec = vec / |vec|
 */
void normalise(double vec[3])
{
  double length = sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]);
  vec[0] /= length;
  vec[1] /= length;
  vec[2] /= length;
}
// ****************************************************************************************************************************************************  





// ****************************************************************************************************************************************************  
/**
 * \brief Compute the cross product of two vectors
 */
void crossProduct(double vec1[3], double vec2[3], double result[3])
/* cross product (cartesian p., outer p.) of two vectors: */
/*    -  x*y is orthogonal to x and y.                    */
/*    -  |x*y| = |x| * |y| * sin(angle(x,y)).             */
/*    -  x, y, and x*y form a right-handed system.        */
/* note:  x*y  !=  y*x  (not commutative)                 */
{
  result[0] = vec1[1]*vec2[2] - vec1[2]*vec2[1];
  result[1] = vec1[2]*vec2[0] - vec1[0]*vec2[2];
  result[2] = vec1[0]*vec2[1] - vec1[1]*vec2[0];
}
// ****************************************************************************************************************************************************  





// ****************************************************************************************************************************************************  
/**
 * \brief Rotate the vextor x about angle around the normal vector axis
 * 
 * Rotates vector x clockwise around axis (looking along axis while it is pointing towards you).
 * axis must be a UNIT VECTOR
 */
void rotate(double x[3], double ang, double axis[3])
{
  int i, j;
  double cosa = cos(-ang);
  double sina = sin(-ang);
  double omcosa = 1.0 - cosa;
  double R[3][3] = {{cosa+axis[0]*axis[0]*omcosa, 
                     axis[0]*axis[1]*omcosa-axis[2]*sina,
                     axis[0]*axis[2]*omcosa+axis[1]*sina},
                    {axis[1]*axis[0]*omcosa+axis[2]*sina,
                     cosa+axis[1]*axis[1]*omcosa,
                     axis[1]*axis[2]*omcosa-axis[0]*sina},
                    {axis[2]*axis[0]*omcosa-axis[1]*sina,
                     axis[2]*axis[1]*omcosa+axis[0]*sina,
                     cosa+axis[2]*axis[2]*omcosa}};
  double result[3] = {0.0, 0.0, 0.0};
  for (i=0; i<3; ++i)
    for (j=0; j<3; ++j)
      result[i] += R[i][j]*x[j];
  for (i=0; i<3; ++i) x[i] = result[i];
}
// ****************************************************************************************************************************************************  





// ****************************************************************************************************************************************************  
/**
 * \brief Determine whether the vectors x, y and z constitute a right-handed system
 * 
 * Determines whether vectors x,y & z constitute a right-handed system by checking the sign of the triple product or det(x,y,z).
 */
int rightHanded(double x[3], double y[3], double z[3])
{
  double tripleProduct = x[0]*y[1]*z[2] + y[0]*z[1]*x[2] + z[0]*x[1]*y[2] - z[0]*y[1]*x[2] - x[0]*z[1]*y[2] - y[0]*x[1]*z[2];
  return (tripleProduct >= 0.0) ? 1 : 0;
}
// ****************************************************************************************************************************************************  





// ****************************************************************************************************************************************************  
/**
 * \brief Determines the orthogonal projection of vector x onto the span of the two ORTHONORMAL vectors vec1 and vec2.
 */
void orthoProject(double x[3], double vec1[3], double vec2[3])
{
  double sprod1 = x[0]*vec1[0] + x[1]*vec1[1] + x[2]*vec1[2];
  double sprod2 = x[0]*vec2[0] + x[1]*vec2[1] + x[2]*vec2[2];
  x[0] = sprod1*vec1[0] + sprod2*vec2[0];
  x[1] = sprod1*vec1[1] + sprod2*vec2[1];
  x[2] = sprod1*vec1[2] + sprod2*vec2[2];
}
// ****************************************************************************************************************************************************  





// ****************************************************************************************************************************************************  
/**
 * \brief Determines the angle between vectors x and y.
 */
double angle(double x[3], double y[3])
{
  double sprod = x[0]*y[0] + x[1]*y[1] + x[2]*y[2];
  double absx  = sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
  double absy  = sqrt(y[0]*y[0] + y[1]*y[1] + y[2]*y[2]);
  return acos(sprod/(absx*absy));
}
// ****************************************************************************************************************************************************  





// ****************************************************************************************************************************************************  
/**
 * \brief Convert geographical spherical coordinates to a Cartesian normal vector
 * 
 * Turns geographical coordinates (latitude & longitude) into a vector - 
 * Result is a unit (!) vector referring to the (right-handed) coordinate
 * system spanned by the three vectors pointing from geocenter to:       
 *   x) intersection of greenwich meridian and equator                   
 *   y) intersection of 90 deg. East meridian and equator                
 *   z) north pole                                                       
 */
// ****************************************************************************************************************************************************  
void coord2vec(double sinlati, double longi, double x[3])
{
  double coslati = sqrt(1.0-sinlati*sinlati);
  x[0] = cos(longi) * coslati;  /* 'Greenwich'  */
  x[1] = sin(longi) * coslati;  /* 'Ganges'     */
  x[2] = sinlati;               /* 'North Pole' */
}
// ****************************************************************************************************************************************************  



// ****************************************************************************************************************************************************  
/**
 * \brief Compute geographical spherical coordinates from a Cartesian normal vector
 *
 * Inverse of coord2vec() (see there for more details)
 */
// ****************************************************************************************************************************************************  
void vec2coord(double x[3], double *sinlati, double *longi)
{
  double greenwich[3] = {1.0, 0.0, 0.0};
  double ganges[3]    = {0.0, 1.0, 0.0};
  double northpole[3] = {0.0, 0.0, 1.0};
  double dummy[3]     = {x[0], x[1], x[2]};
  
  *sinlati = sin(0.5*pi - angle(northpole, x));
  orthoProject(dummy, greenwich, ganges);
  *longi = angle(greenwich, dummy);
  if(rightHanded(greenwich,northpole,dummy)) *longi *= -1.0;
}
// ****************************************************************************************************************************************************  







// *** OTHER ***



// ****************************************************************************************************************************************************  
/**
 * \brief Grab a random seed from the system clock
 * 
 * If seed==0, set (randomise) seed using the system clock (hence a random (random seed) :-)
 */
// ****************************************************************************************************************************************************  
void setSeed(int *seed)
{
  struct timeval time_now;
  struct timezone tz;
  if(*seed==0) {
    gettimeofday(&time_now, &tz);
    *seed = time_now.tv_usec;
  }
}
// ****************************************************************************************************************************************************  



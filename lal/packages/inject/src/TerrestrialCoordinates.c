/*
*  Copyright (C) 2007 Jolien Creighton, Reinhard Prix, Teviet Creighton, John Whelan
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

/** \file
 * \ingroup SkyCoordinates
 * \author Creighton, T. D.
 * \date  $Date$
 * \brief Converts among equatorial, geographic, and horizon coordinates.
 *
 * $Id$
 *

\par Description

The functions LALEquatorialToGeographic() and
LALGeographicToEquatorial() convert between equatorial and
geographic coordinate systems, reading coordinates in the first system
from <tt>*input</tt> and storing the new coordinates in <tt>*output</tt>.
The two pointers may point to the same object, in which case the
conversion is done in place.  The functions will also check
<tt>input->system</tt> and set <tt>output->system</tt> as appropriate.
Because the geographic coordinate system is not fixed, one must also
specify the time of the transformation in <tt>*gpsTime</tt>.

The function LALSystemToHorizon() transforms coordinates from
either celestial equatorial coordinates or geographic coordinates to a
horizon coordinate system, reading coordinates in the first system
from <tt>*input</tt> and storing the horizon coordinates in
<tt>*output</tt>, as above.  The parameter <tt>*zenith</tt> specifies the
direction of the vertical axis <em>in the original coordinate
system</em>; the routine checks to see that <tt>input->system</tt> and
<tt>zenith->system</tt> agree.  Normally this routine is used to convert
from <em>geographic</em> latitude and longitude to a horizon system, in
which case <tt>*zenith</tt> simply stores the geographic (geodetic)
coordinates of the observer; if converting from equatorial
coordinates, <tt>zenith->longitude</tt> should store the local mean
sidereal time of the horizon system.

The function LALHorizonToSystem() does the reverse of the
above, transforming coordinates from horizon coordinates to either
equatorial or geographic coordinates as specified by
<tt>zenith->system</tt>; the value of <tt>output->system</tt> is set to
agree with <tt>zenith->system</tt>.

Although it is conventional to specify an observation location by its
<em>geodetic</em> coordinates, some routines may provide or require
<em>geocentric</em> coordinates.  The routines
LALGeocentricToGeodetic() and LALGeodeticToGeocentric()
perform this computation, reading and writing to the variable
parameter structure <tt>*location</tt>.  The function
LALGeocentricToGeodetic() reads the fields <tt>location->x</tt>,
<tt>y</tt>, <tt>z</tt>, and computes <tt>location->zenith</tt> and
<tt>location->altitude</tt>.  The function
LALGeodeticToGeocentric() does the reverse, and also sets the
fields <tt>location->position</tt> and <tt>location->radius</tt>.

\par Algorithm

These routines follow the formulae in Sec.5.1 of \ref Lang_K1999,
which we reproduce below.

\par Geographic coordinates

Since geographic and equatorial
coordinates share the same \f$z\f$-axis, the geographic latitude \f$\phi\f$ of
a direction in space is the same as its declination \f$\delta\f$, and
longitude \f$\lambda\f$ and right ascension \f$\alpha\f$ differ only through
the rotation of the Earth:
\f{equation}
\label{eq:lambda-geographic}
\lambda = \alpha - \left(\frac{2\pi\,\mathrm{radians}}
	{24\,\mathrm{hours}}\right)\times\mathrm{GMST} \; ,
\f}
where GMST is Greenwich mean sidereal time.  The conversion routines
here simply use the functions in the <tt>date</tt> package to compute
GMST for a given GPS time, and add it to the longitude.  While this is
simple enough, it does involve several function calls, so it is
convenient to collect these into one routine.

\par Horizon coordinates
We correct a typographical
error on the second line of Eq.(5.45) of \ref Lang_K1999 (it should
have \f$\cos A\f$, not \f$\sin A\f$).  We also note that while our latitudinal
coordinate is just the altitude \f$a\f$ in this system, our longitudinal
coordinate increases counterclockwise, and thus corresponds to the
<em>negative</em> of the azimuth \f$A\f$ as defined by \ref Lang_K1999.
So we have:
\f{eqnarray}
\label{eq:altitude-horizon}
a & = & \arcsin(\sin\delta\sin\phi + \cos\delta\cos\phi\cos h) \; , \\
\label{eq:azimuth-horizon}
-A & = & \arctan\!2(\cos\delta\sin h, \sin\delta\cos\phi -
		\cos\delta\sin\phi\cos h) \; ,
\f}
where \f$\delta\f$ is the declination (geographic latitude) of the
direction being transformed, \f$\phi\f$ is the geographic latitude of the
observer's zenith (i.e.\ the observer's <em>geodetic</em> latitude), and
\f$h\f$ is the <em>hour angle</em> of the direction being transformed.  This
is defined as:
\f{eqnarray}
h & = & \lambda_\mathrm{zenith} - \lambda \nonumber\\
  & = & \mathrm{LMST} - \alpha \nonumber
\f}
where LMST is the local mean sidereal time at the point of
observation.  The inverse transformation is:
\f{eqnarray}
\label{eq:delta-horizon}
\delta & = & \arcsin(\sin a\sin\phi + \cos a\cos A\cos\phi) \; , \\
\label{eq:h-horizon}
h & = & \arctan\!2[\cos a\sin(-A), \sin a\cos\phi -
		\cos a\cos A\sin\phi] \; .
\f}
As explained in <tt>CelestialCoordinates.c</tt>, the function
\f$\arctan\!2(y,x)\f$ returns the argument of the complex number \f$x+iy\f$.

\image html inject_geodetic.png "Fig. 1: The difference between geodetic and geocentric latitude."
\latexonly
\newpage
\begin{wrapfigure}{r}{0.35\textwidth}
\vspace{-2ex}
\begin{center}
\resizebox{0.3\textwidth}{!}{\includegraphics{inject_geodetic}} \\
\parbox{0.3\textwidth}{\caption{\label{fig:geodetic} The difference
between geodetic and geocentric latitude.}}
\end{center}
\vspace{-2ex}
\end{wrapfigure}
\paragraph{Geocentric coordinates:}
\endlatexonly \htmlonly
<b>Geocentric coordinates:</b>\endhtmlonly
As shown in
\latexonly Fig.~\ref{fig:geodetic},\endlatexonly
\htmlonly Fig. 1,\endhtmlonly
the ellipticity of the Earth means that the
vertical axis of a point on the Earth's surface does not pass through
the geometric centre of the Earth.  This means that the geodetic
latitude of a location (defined as the latitude angle
\f$\phi_\mathrm{geodetic}\f$ of that location's zenith direction) is
typically some 10 arcminutes larger than its geocentric latitude
(defined as the latitude angle \f$\phi_\mathrm{geographic}\f$ of the
position vector from the geocentre through the location).
Cartographers traditionally refer to locations by their geodetic
coordinates, since these can be determined locally; however,
geocentric coordinates are required if one wants to construct a
uniform Cartesian system for the Earth as a whole.

To transform from geodetic to geocentric coordinates, one first
defines a ``reference ellipsoid'', the best-fit ellipsoid to the
surface of the Earth.  This is specified by the polar and equatorial
radii of the Earth \f$r_p\f$ and \f$r_e\f$, or equivalently by \f$r_e\f$ and a
flattening factor:
\f[
f \equiv 1 - \frac{r_p}{r_e} = 0.00335281 \; .
\f]
(This constant will eventually migrate into <tt>LALConstants.h</tt>.)
The surface of the ellipsoid is then specified by the equation
\f[
r = r_e ( 1 - f\sin^2\phi ) \; ,
\f]
where \f$\phi=\phi_\mathrm{geodetic}\f$ is the geodetic latitude.  For
points off of the reference ellipsoid, the transformation from
geodetic coordinates \f$\lambda\f$, \f$\phi\f$ to geocentric Cartesian
coordinates is:
\f{eqnarray}
x & = & ( r_e C + h ) \cos\phi\cos\lambda \; , \\
y & = & ( r_e C + h ) \cos\phi\sin\lambda \; , \\
z & = & ( r_e S + h ) \sin\phi \; ,
\f}
where
\f{eqnarray}
C & = & \frac{1}{\sqrt{\cos^2\phi + (1-f)^2\sin^2\phi}} \; , \\
S & = & (1-f)^2 C \; ,
\f}
and \f$h\f$ is the perpendicular elevation of the location above the
reference ellipsoid.  The geocentric spherical coordinates are given
simply by:
\f{eqnarray}
r & = & \sqrt{ x^2 + y^2 + z^2 } \; , \\
\lambda_\mathrm{geocentric} & = & \lambda \quad = \quad
	\lambda_\mathrm{geodetic} \; , \\
\phi_\mathrm{geocentric} & = & \arcsin( z/r ) \; .
\f}
When computing \f$r\f$ we are careful to factor out the largest component
before computing the sum of squares, to avoid floating-point overflow;
however this should be unnecessary for radii near the surface of the
Earth.

The inverse transformation is somewhat trickier.  Eq.(5.29)
of \ref Lang_K1999 conveniently gives the transformation in terms
of a sequence of intermediate variables, but unfortunately these
variables are not particularly computer-friendly, in that they are
prone to underflow or overflow errors.  The following equations
essentially reproduce this sequence using better-behaved methods of
calculation.

Given geocentric Cartesian coordinates
\f$x=r\cos\phi_\mathrm{geocentric}\cos\lambda\f$,
\f$y=r\cos\phi_\mathrm{geocentric}\sin\lambda\f$, and
\f$z=r\sin\phi_\mathrm{geocentric}\f$, one computes the following:
\f{eqnarray}
\varpi & = & \sqrt{ \left(\frac{x}{r_e}\right)^2
	+ \left(\frac{y}{r_e}\right)^2 } \;,\nonumber\\
E & = & (1-f)\left|\frac{z}{r_e}\right| - f(2-f) \;,\nonumber\\
F & = & (1-f)\left|\frac{z}{r_e}\right| + f(2-f) \;,\nonumber\\
P & = & \frac{4}{3}\left( EF + \varpi^2 \right) \quad = \quad
	\frac{4}{3}\left[ \varpi^2 + (1-f)^2\left(\frac{z}{r_e}\right)^2
		- f^2(2-f)^2 \right] \;,\nonumber\\
Q & = & 2(F^2 - E^2) \quad = \quad
	8f(1-f)(2-f)\left|\frac{z}{r_e}\right| \;,\nonumber\\
D & = & P^3 + \varpi^2 Q^2 \;,\nonumber\\
v & = & \left\{\begin{array}{lr}
	\left(\sqrt{D}+\varpi Q\right)^{1/3}
		- \left(\sqrt{D}-\varpi Q\right)^{1/3} &
		D\geq0 \\
	2\sqrt{-P}\cos\left(\frac{1}{3}
		\arccos\left[\frac{Q\varpi}{P\sqrt{-P}}\right]\right) &
		D\leq0 \end{array}\right.\nonumber\\
G & = & \mbox{$\frac{1}{2}$}\left(E+\sqrt{E^2 + \varpi v}\right)\;,\nonumber\\
H & = & \frac{\varpi^2 F - \varpi vG}{G^2(2G-E)} \;,\nonumber\\
t & = & G\left(\sqrt{1+H}-1\right) \;.\nonumber
\f}
Once we have \f$t\f$ and \f$\varpi\f$, we can compute the geodetic longitude
\f$\lambda\f$, latitude \f$\phi\f$, and elevation \f$h\f$:
\f{eqnarray}
\lambda & = & \arctan\!2(y,x) \; , \\
\phi & = & \mathrm{sgn}({z})\arctan\left[\frac{2}{1-f}
	\left(\frac{(\varpi-t)(\varpi+t)}{\varpi t}\right)\right] \; , \\
h & = & r_e(\varpi-t/\varpi)\cos\phi
	+ [z-\mathrm{sgn}({z})r_e(1-f)]\sin\phi \; .
\f}
These formulae, however, introduce certain concerns of numerical
precision that have been only partially dealt with in this code.
Specifically:

- There is a coordinate singularity at \f$\varpi=0\f$, which we deal
with by setting \f$\phi=\pm90^\circ\f$ and \f$\lambda\f$ arbitrarily to
\f$0^\circ\f$.  When \f$z=0\f$ as well, we arbitrarily choose the positive
sign for \f$\phi\f$.  However, the computation of \f$h\f$ in particular has
tricky cancelations as \f$\varpi\rightarrow0\f$, which may give rise to
numerical errors.  These have not yet been thoroughly explored.

- There is another coordinate singularity when \f$D\rightarrow0\f$,
which defines an ellipsoid with equatorial radius
\f$r_0=r_ef(2-f)=42.6977\f$km and axial height \f$z_0=r_e/(1-f)=42.8413\f$km.
Within this ellipsoid, lines of constant latitude begin to cross one
another.  The listed solution is an analytic continuation of the
exterior solution which assigns these points a unique, if arbitrary,
geodetic latitude.  This solution has some peculiar behaviour, such as
giving points in the equatorial plane a positive latitude.  In
practice, however, users will rarely be interested coordinate
transformations deep within the Earth's core.

- The equations for \f$v\f$ and \f$G\f$ have square and cube roots of
expressions involving squares and cubes of numbers.  For formal
robustness one should factor out the leading-order dependence, so that
one is assured of taking squares and cubes of numbers near unity.
However, we are using <tt>REAL8</tt> precision, and have already
normalized our distances by the Earth's radius \f$r_e\f$, so the point is
almost certainly irrelevant.

- The expression for \f$H\f$ may go to zero, leading to precision
errors in \f$t\f$; the number of digits of precision lost is on the order
of the number of leading zeros after the decimal place in \f$H\f$.  I
arbitrarily state that we should not lose more than 4 of our 16
decimal places of precision, meaning that we should series-expand the
square root for \f$H<10^{-4}\f$.  To get our 12 places of precision back,
we need an expansion to \f$H^3\f$:
\f[
t \approx G\left(\frac{1}{2}H - \frac{3}{8}H^2
	+ \frac{5}{16}H^3\right) \; .
\f]

When computing \f$\phi\f$, we first compute \f$t-\varpi\f$, \f$t+\varpi\f$,
\f$t^{-1}\f$, and \f$\varpi^{-1}\f$, sort them by order of magnitude, and
alternately multiply large and small terms.  We note that if the
argument of the \f$\arctan\f$ function is large we have
\f[
\arctan(x) = \mathrm{sgn}(x)\frac{\pi}{2}
	- \arctan\left(\frac{1}{x}\right) \; ,
\f]
but the <tt>atan()</tt> function in the C math library should be smart
enough to do this itself.


<b>Ellipsoidal vs. orthometric elevation:</b> In this module it
is assumed that all elevations refer heights above the reference
ellipsoid.  This is the elevation computed by such techniques as GPS
triangulation.  However, the ``true'' orthometric elevation refers to
the height above the mean sea level or <em>geoid</em>, a level surface
in the Earth's gravitational potential.  Thus, even if two points have
the same ellipsoidal elevation, water will still flow from the higher
to the lower orthometric elevation.

The difference between the geoid and reference ellipsoid is called the
``undulation of the geoid'', and can vary by over a hundred metres
over the Earth's surface.  However, it can only be determined through
painstaking measurements of local variations in the Earth's
gravitational field.  For this reason we will ignore the undulation of
the geoid.

\par Uses
\code
XLALGreenwichMeanSiderealTime()
LALDHeapSort()
\endcode

*/


/*---------- laldoc-documentation follows ---------- */

/*********************** <lalVerbatim file="TerrestrialCoordinatesCV">
Author: Creighton, T. D.
$Id$
**************************************************** </lalVerbatim> */

/********************************************************** <lalLaTeX>

\subsection{Module \texttt{TerrestrialCoordinates.c}}
\label{ss:TerrestrialCoordinates.c}

Converts among equatorial, geographic, and horizon coordinates.

\subsubsection*{Prototypes}
\vspace{0.1in}
\input{TerrestrialCoordinatesCP}
\idx{LALEquatorialToGeographic()}
\idx{LALGeographicToEquatorial()}
\idx{LALSystemToHorizon()}
\idx{LALHorizonToSystem()}

\subsubsection*{Description}

The functions \verb@LALEquatorialToGeographic()@ and
\verb@LALGeographicToEquatorial()@ convert between equatorial and
geographic coordinate systems, reading coordinates in the first system
from \verb@*input@ and storing the new coordinates in \verb@*output@.
The two pointers may point to the same object, in which case the
conversion is done in place.  The functions will also check
\verb@input->system@ and set \verb@output->system@ as appropriate.
Because the geographic coordinate system is not fixed, one must also
specify the time of the transformation in \verb@*gpsTime@.

The function \verb@LALSystemToHorizon()@ transforms coordinates from
either celestial equatorial coordinates or geographic coordinates to a
horizon coordinate system, reading coordinates in the first system
from \verb@*input@ and storing the horizon coordinates in
\verb@*output@, as above.  The parameter \verb@*zenith@ specifies the
direction of the vertical axis \emph{in the original coordinate
system}; the routine checks to see that \verb@input->system@ and
\verb@zenith->system@ agree.  Normally this routine is used to convert
from \emph{geographic} latitude and longitude to a horizon system, in
which case \verb@*zenith@ simply stores the geographic (geodetic)
coordinates of the observer; if converting from equatorial
coordinates, \verb@zenith->longitude@ should store the local mean
sidereal time of the horizon system.

The function \verb@LALHorizonToSystem()@ does the reverse of the
above, transforming coordinates from horizon coordinates to either
equatorial or geographic coordinates as specified by
\verb@zenith->system@; the value of \verb@output->system@ is set to
agree with \verb@zenith->system@.

Although it is conventional to specify an observation location by its
\emph{geodetic} coordinates, some routines may provide or require
\emph{geocentric} coordinates.  The routines
\verb@LALGeocentricToGeodetic()@ and \verb@LALGeodeticToGeocentric()@
perform this computation, reading and writing to the variable
parameter structure \verb@*location@.  The function
\verb@LALGeocentricToGeodetic()@ reads the fields \verb@location->x@,
\verb@y@, \verb@z@, and computes \verb@location->zenith@ and
\verb@location->altitude@.  The function
\verb@LALGeodeticToGeocentric()@ does the reverse, and also sets the
fields \verb@location->position@ and \verb@location->radius@.

\subsubsection*{Algorithm}

These routines follow the formulae in Sec.~5.1 of~\cite{Lang_K:1999},
which we reproduce below.

\paragraph{Geographic coordinates:} Since geographic and equatorial
coordinates share the same $z$-axis, the geographic latitude $\phi$ of
a direction in space is the same as its declination $\delta$, and
longitude $\lambda$ and right ascension $\alpha$ differ only through
the rotation of the Earth:
\begin{equation}
\label{eq:lambda-geographic}
\lambda = \alpha - \left(\frac{2\pi\,\mathrm{radians}}
	{24\,\mathrm{hours}}\right)\times\mathrm{GMST} \; ,
\end{equation}
where GMST is Greenwich mean sidereal time.  The conversion routines
here simply use the functions in the \verb@date@ package to compute
GMST for a given GPS time, and add it to the longitude.  While this is
simple enough, it does involve several function calls, so it is
convenient to collect these into one routine.

\paragraph{Horizon coordinates:} We correct a typographical
error on the second line of Eq.~5.45 of~\cite{Lang_K:1999} (it should
have $\cos A$, not $\sin A$).  We also note that while our latitudinal
coordinate is just the altitude $a$ in this system, our longitudinal
coordinate increases counterclockwise, and thus corresponds to the
\emph{negative} of the azimuth $A$ as defined by~\cite{Lang_K:1999}.
So we have:
\begin{eqnarray}
\label{eq:altitude-horizon}
a & = & \arcsin(\sin\delta\sin\phi + \cos\delta\cos\phi\cos h) \; , \\
\label{eq:azimuth-horizon}
-A & = & \arctan\!2(\cos\delta\sin h, \sin\delta\cos\phi -
		\cos\delta\sin\phi\cos h) \; ,
\end{eqnarray}
where $\delta$ is the declination (geographic latitude) of the
direction being transformed, $\phi$ is the geographic latitude of the
observer's zenith (i.e.\ the observer's \emph{geodetic} latitude), and
$h$ is the \emph{hour angle} of the direction being transformed.  This
is defined as:
\begin{eqnarray}
h & = & \lambda_\mathrm{zenith} - \lambda \nonumber\\
  & = & \mathrm{LMST} - \alpha \nonumber
\end{eqnarray}
where LMST is the local mean sidereal time at the point of
observation.  The inverse transformation is:
\begin{eqnarray}
\label{eq:delta-horizon}
\delta & = & \arcsin(\sin a\sin\phi + \cos a\cos A\cos\phi) \; , \\
\label{eq:h-horizon}
h & = & \arctan\!2[\cos a\sin(-A), \sin a\cos\phi -
		\cos a\cos A\sin\phi] \; .
\end{eqnarray}
As explained in \verb@CelestialCoordinates.c@, the function
$\arctan\!2(y,x)$ returns the argument of the complex number $x+iy$.

\newpage

\begin{wrapfigure}{r}{0.35\textwidth}
\vspace{-2ex}
\begin{center}
\resizebox{0.3\textwidth}{!}{\includegraphics{inject_geodetic}} \\
\parbox{0.3\textwidth}{\caption{\label{fig:geodetic} The difference
between geodetic and geocentric latitude.}}
\end{center}
\vspace{-2ex}
\end{wrapfigure}
\paragraph{Geocentric coordinates:} As shown in
Fig.~\ref{fig:geodetic}, the ellipticity of the Earth means that the
vertical axis of a point on the Earth's surface does not pass through
the geometric centre of the Earth.  This means that the geodetic
latitude of a location (defined as the latitude angle
$\phi_\mathrm{geodetic}$ of that location's zenith direction) is
typically some 10~arcminutes larger than its geocentric latitude
(defined as the latitude angle $\phi_\mathrm{geographic}$ of the
position vector from the geocentre through the location).
Cartographers traditionally refer to locations by their geodetic
coordinates, since these can be determined locally; however,
geocentric coordinates are required if one wants to construct a
uniform Cartesian system for the Earth as a whole.

To transform from geodetic to geocentric coordinates, one first
defines a ``reference ellipsoid'', the best-fit ellipsoid to the
surface of the Earth.  This is specified by the polar and equatorial
radii of the Earth $r_p$ and $r_e$, or equivalently by $r_e$ and a
flattening factor:
$$
f \equiv 1 - \frac{r_p}{r_e} = 0.00335281 \; .
$$
(This constant will eventually migrate into \verb@LALConstants.h@.)
The surface of the ellipsoid is then specified by the equation
$$
r = r_e ( 1 - f\sin^2\phi ) \; ,
$$
where $\phi=\phi_\mathrm{geodetic}$ is the geodetic latitude.  For
points off of the reference ellipsoid, the transformation from
geodetic coordinates $\lambda$, $\phi$ to geocentric Cartesian
coordinates is:
\begin{eqnarray}
x & = & ( r_e C + h ) \cos\phi\cos\lambda \; , \\
y & = & ( r_e C + h ) \cos\phi\sin\lambda \; , \\
z & = & ( r_e S + h ) \sin\phi \; ,
\end{eqnarray}
where
\begin{eqnarray}
C & = & \frac{1}{\sqrt{\cos^2\phi + (1-f)^2\sin^2\phi}} \; , \\
S & = & (1-f)^2 C \; ,
\end{eqnarray}
and $h$ is the perpendicular elevation of the location above the
reference ellipsoid.  The geocentric spherical coordinates are given
simply by:
\begin{eqnarray}
r & = & \sqrt{ x^2 + y^2 + z^2 } \; , \\
\lambda_\mathrm{geocentric} & = & \lambda \quad = \quad
	\lambda_\mathrm{geodetic} \; , \\
\phi_\mathrm{geocentric} & = & \arcsin( z/r ) \; .
\end{eqnarray}
When computing $r$ we are careful to factor out the largest component
before computing the sum of squares, to avoid floating-point overflow;
however this should be unnecessary for radii near the surface of the
Earth.

The inverse transformation is somewhat trickier.  Eq.~5.29
of~\cite{Lang_K:1999} conveniently gives the transformation in terms
of a sequence of intermediate variables, but unfortunately these
variables are not particularly computer-friendly, in that they are
prone to underflow or overflow errors.  The following equations
essentially reproduce this sequence using better-behaved methods of
calculation.

Given geocentric Cartesian coordinates
$x=r\cos\phi_\mathrm{geocentric}\cos\lambda$,
$y=r\cos\phi_\mathrm{geocentric}\sin\lambda$, and
$z=r\sin\phi_\mathrm{geocentric}$, one computes the following:
\begin{eqnarray}
\varpi & = & \sqrt{ \left(\frac{x}{r_e}\right)^2
	+ \left(\frac{y}{r_e}\right)^2 } \;,\nonumber\\
E & = & (1-f)\left|\frac{z}{r_e}\right| - f(2-f) \;,\nonumber\\
F & = & (1-f)\left|\frac{z}{r_e}\right| + f(2-f) \;,\nonumber\\
P & = & \frac{4}{3}\left( EF + \varpi^2 \right) \quad = \quad
	\frac{4}{3}\left[ \varpi^2 + (1-f)^2\left(\frac{z}{r_e}\right)^2
		- f^2(2-f)^2 \right] \;,\nonumber\\
Q & = & 2\varpi(F^2 - E^2) \quad = \quad
	8\varpi f(1-f)(2-f)\left|\frac{z}{r_e}\right| \;,\nonumber\\
D & = & P^3 + Q^2 \;,\nonumber\\
v & = & \left\{\begin{array}{lr}
	\left(\sqrt{D}+Q\right)^{1/3}
		- \left(\sqrt{D}-Q\right)^{1/3} &
		D\geq0 \\
	2\sqrt{-P}\cos\left(\frac{1}{3}
		\arccos\left[\frac{Q}{-P\sqrt{-P}}\right]\right) &
		D\leq0 \end{array}\right.\nonumber\\
W & = & \sqrt{E^2 + \varpi v} \nonumber\\[1ex]
G & = & \mbox{$\frac{1}{2}$}\left(E+W\right)\;,\nonumber\\
t & = & \sqrt{G^2+\frac{\varpi^2 F - \varpi vG}{W}}-G \;.\nonumber
\end{eqnarray}
Once we have $t$ and $\varpi$, we can compute the geodetic longitude
$\lambda$, latitude $\phi$, and elevation $h$:
\begin{eqnarray}
\lambda & = & \arctan\!2(y,x) \; , \\
\phi & = & \mathrm{sgn}({z})\arctan\left[\frac{1}{2(1-f)}
	\left(\frac{(\varpi-t)(\varpi+t)}{\varpi t}\right)\right] \; , \\
h & = & r_e(\varpi-t/\varpi)\cos\phi
	+ [z-\mathrm{sgn}({z})r_e(1-f)]\sin\phi \; .
\end{eqnarray}

\begin{wrapfigure}{r}{0.47\textwidth}
\vspace{-8ex}
\begin{center}
\resizebox{0.42\textwidth}{!}{\includegraphics{inject_geodeticsing}} \\
\parbox{0.42\textwidth}{\caption{\label{fig:geodeticsing} Singular
surfaces in the geodetic coordinate system.  The ellipticity of this
spheroid has been exaggerated compared with the Earth.}}
\end{center}
\vspace{-7ex}
\end{wrapfigure}
\noindent These formulae still leave certain areas where coordinate
singularities or numerical cancelations can occur.  Some of these have
been dealt with in the code:
\begin{itemize}
\item There is a coordinate singularity at $\varpi=0$, which we deal
with by setting $\phi=\pm90^\circ$ and $\lambda$ arbitrarily to
$0^\circ$.  When $z=0$ as well, we arbitrarily choose the positive
sign for $\phi$.  As $\varpi\rightarrow0$, there are cancellations in
the computation of $G$ and $t$, which call for special treatment
(below).

\item There is another coordinate singularity when $D\leq0$, where
lines of constant geodetic latitude will cross each other, as shown in
Fig.~\ref{fig:geodeticsing}.  That is, a given point within this
region can be assigned a range of geodetic latitudes.  The
multi-valued region lies within an inner ellipsoid $P\leq0$, which in
the case of the Earth has equatorial radius $r_0=r_ef(2-f)=42.6977$km
and axial height $z_0=r_0/(1-f)=42.8413$km.  The formula for $v$,
above, has an analytic continuation to $D\leq0$, assigning consistent
(though not unique) values of latitute and elevation to these points.
\end{itemize}

\begin{itemize}
\item Near the equator we have $Q\rightarrow0$, and the first
expression for $v$ becomes a difference of nearly-equal numbers,
leading to loss of precision.  To deal with this, we write that
expression for $v$ as:
$$
\begin{array}{rcl@{\qquad}c@{\qquad}l}
  v &=& D^{1/6}\left[(1+B)^{1/3}-(1-B)^{1/3}\right]
  &\mbox{where}& B \;\;=\;\; \displaystyle \frac{Q}{\sqrt{D}} \\
  &\approx& D^{1/6}\left[\frac{2}{3}B+\frac{10}{81}B^3\right]
  &\mbox{as}& B \;\;\rightarrow\;\;0
\end{array}
$$

The switch from the ``exact'' formula to the series expansion is done
for $B<10^{-3}$ (within about $2^\circ$ of the equator).  This was
found by experimentation to be the point where the inaccuracy of the
series expansion is roughly equal to the imprecision in evaluating the
``exact'' formula.  The resulting position errors are of order 1 part
in $10^{12}$ or less; i.e.\ about 3--4 digits loss of precision.

\item In some places we have expressions of the form $\sqrt{a^2+b}-a$,
which becomes a difference of nearly equal numbers for $|b|\ll a^2$,
resulting in loss of precision.  There are three distinct lines or
surfaces where this occurs:
\begin{enumerate}
\item Near the ellipsoidal surface $P\approx0$, the expression $\sqrt{D}-Q$
in the equation for $v$ becomes of this form.
\item Near the polar axis $\varpi\approx0$, the expression for $t$
becomes of this form.
\item Near the polar axis $\varpi\approx0$ within the inner ellipsoid,
we have $E<0$ and the expression for $G$ becomes of this form.
\end{enumerate}
In each case, we expand in the small parameter $H=b/a^2$, giving:
$$
\sqrt{a^2+b}-a \;\;\approx\;\; a\left(\mbox{$\frac{1}{2}$}H
-\mbox{$\frac{1}{8}$}H^2+\mbox{$\frac{1}{16}$}H^3\right)
\qquad\mbox{for}\qquad |H| = \left|\frac{b}{a^2}\right| \ll 1
$$

We switch to the series expansion for $|H|<\times10^{-4}$, which again
is the point where residual errors in the series expansion are about
the same as the loss of precision without the expansion.  This formula
converges much better than the one for $v$ (above), resulting in
negligible loss of precision in the final position.

\item The only remaining known numerical singularities are at the
poles of the inner ellipsoid, shown as red dots in
Fig.~\ref{fig:geodeticsing}.  These points stubbornly resist
high-precision calculation.  However, they are extremely unlikely to
come up in practice.
\end{itemize}

\paragraph{Ellipsoidal vs.\ orthometric elevation:} In this module it
is assumed that all elevations refer heights above the reference
ellipsoid.  This is the elevation computed by such techniques as GPS
triangulation.  However, the ``true'' orthometric elevation refers to
the height above the mean sea level or \emph{geoid}, a level surface
in the Earth's gravitational potential.  Thus, even if two points have
the same ellipsoidal elevation, water will still flow from the higher
to the lower orthometric elevation.

The difference between the geoid and reference ellipsoid is called the
``undulation of the geoid'', and can vary by over a hundred metres
over the Earth's surface.  However, it can only be determined through
painstaking measurements of local variations in the Earth's
gravitational field.  For this reason we will ignore the undulation of
the geoid.

\subsubsection*{Uses}
\begin{verbatim}
XLALGreenwichMeanSiderealTime()
\end{verbatim}

\subsubsection*{Notes}

\vfill{\footnotesize\input{TerrestrialCoordinatesCV}}

******************************************************* </lalLaTeX> */


#include <math.h>
#include <lal/LALStdlib.h>
#include <lal/LALConstants.h>
#include <lal/Date.h>
#include <lal/Sort.h>
#include <lal/SkyCoordinates.h>
#include <string.h>

#define LAL_EARTHFLAT (0.00335281)
#define LAL_HSERIES (0.0001) /* value H below which we expand sqrt(1+H)-1 */
#define LAL_BSERIES (0.001)  /* value B below which we expand v */

NRCSID( TERRESTRIALCOORDINATESC, "$Id$" );

/* <lalVerbatim file="TerrestrialCoordinatesCP"> */
void
LALEquatorialToGeographic( LALStatus   *stat,
			   SkyPosition *output,
			   SkyPosition *input,
			   LIGOTimeGPS *gpsTime )
{ /* </lalVerbatim> */
  REAL8 gmst;            /* siderial time (radians) */

  INITSTATUS( stat, "LALEquatorialToGeographic", TERRESTRIALCOORDINATESC );

  /* Make sure parameter structures exist. */
  ASSERT( input, stat, SKYCOORDINATESH_ENUL, SKYCOORDINATESH_MSGENUL );
  ASSERT( output, stat, SKYCOORDINATESH_ENUL, SKYCOORDINATESH_MSGENUL );
  ASSERT( gpsTime, stat, SKYCOORDINATESH_ENUL, SKYCOORDINATESH_MSGENUL );

  /* Make sure we're given the right coordinate system. */
  if ( input->system != COORDINATESYSTEM_EQUATORIAL ) {
    ABORT( stat, SKYCOORDINATESH_ESYS, SKYCOORDINATESH_MSGESYS );
  }

  /* Compute the Greenwich mean sidereal time. */
  gmst = fmod( XLALGreenwichMeanSiderealTime( gpsTime ), LAL_TWOPI );

  /* Add to longitude, and exit. */
  output->system = COORDINATESYSTEM_GEOGRAPHIC;
  output->longitude = fmod( input->longitude - gmst, LAL_TWOPI );
  if ( output->longitude < 0.0 )
    output->longitude += LAL_TWOPI;
  output->latitude = input->latitude;
  RETURN( stat );
}


/* <lalVerbatim file="TerrestrialCoordinatesCP"> */
void
LALGeographicToEquatorial( LALStatus   *stat,
			   SkyPosition *output,
			   SkyPosition *input,
			   LIGOTimeGPS *gpsTime )
{ /* </lalVerbatim> */
  REAL8 gmst;            /* siderial time (radians) */

  INITSTATUS( stat, "LALEquatorialToGeographic", TERRESTRIALCOORDINATESC );

  /* Make sure parameter structures exist. */
  ASSERT( input, stat, SKYCOORDINATESH_ENUL, SKYCOORDINATESH_MSGENUL );
  ASSERT( output, stat, SKYCOORDINATESH_ENUL, SKYCOORDINATESH_MSGENUL );
  ASSERT( gpsTime, stat, SKYCOORDINATESH_ENUL, SKYCOORDINATESH_MSGENUL );

  /* Make sure we're given the right coordinate system. */
  if ( input->system != COORDINATESYSTEM_GEOGRAPHIC ) {
    ABORT( stat, SKYCOORDINATESH_ESYS, SKYCOORDINATESH_MSGESYS );
  }

  /* Compute the Greenwich mean sidereal time. */
  gmst = fmod( XLALGreenwichMeanSiderealTime( gpsTime ), LAL_TWOPI );

  /* Subtract from longitude, and exit. */
  output->system = COORDINATESYSTEM_EQUATORIAL;
  output->longitude = fmod( input->longitude + gmst, LAL_TWOPI );
  if ( output->longitude < 0.0 )
    output->longitude += LAL_TWOPI;
  output->latitude = input->latitude;
  RETURN( stat );
}


/* <lalVerbatim file="TerrestrialCoordinatesCP"> */
void
LALSystemToHorizon( LALStatus   *stat,
		    SkyPosition *output,
		    SkyPosition *input,
		    const SkyPosition *zenith )
{ /* </lalVerbatim> */
  REAL8 h, sinH, cosH; /* hour angle, and its sine and cosine */
  REAL8 sinP, cosP;    /* sin and cos of zenith latitude */
  REAL8 sinD, cosD;    /* sin and cos of position latitude (declination) */
  REAL8 sina, sinA, cosA; /* sin and cos of altitude and -azimuth */

  INITSTATUS( stat, "LALSystemToHorizon", TERRESTRIALCOORDINATESC );

  /* Make sure parameter structures exist. */
  ASSERT( input, stat, SKYCOORDINATESH_ENUL, SKYCOORDINATESH_MSGENUL );
  ASSERT( output, stat, SKYCOORDINATESH_ENUL, SKYCOORDINATESH_MSGENUL );
  ASSERT( zenith, stat, SKYCOORDINATESH_ENUL, SKYCOORDINATESH_MSGENUL );

  /* Make sure we have consistent input coordinate systems. */
  if ( input->system != zenith->system ) {
    ABORT( stat, SKYCOORDINATESH_ESYS, SKYCOORDINATESH_MSGESYS );
  }
  if ( ( zenith->system != COORDINATESYSTEM_EQUATORIAL ) &&
       ( zenith->system != COORDINATESYSTEM_GEOGRAPHIC ) ) {
    ABORT( stat, SKYCOORDINATESH_ESYS, SKYCOORDINATESH_MSGESYS );
  }

  /* Compute intermediates. */
  h = zenith->longitude - input->longitude;
  sinH = sin( h );
  cosH = cos( h );
  sinP = sin( zenith->latitude );
  cosP = cos( zenith->latitude );
  sinD = sin( input->latitude );
  cosD = cos( input->latitude );

  /* Compute components. */
  sina = sinD*sinP + cosD*cosP*cosH;
  sinA = cosD*sinH;
  cosA = sinD*cosP - cosD*sinP*cosH;

  /* Compute final results. */
  output->system = COORDINATESYSTEM_HORIZON;
  output->latitude = asin( sina );
  output->longitude = atan2( sinA, cosA );

  /* Optional phase correction. */
  if ( output->longitude < 0.0 )
    output->longitude += LAL_TWOPI;

  RETURN( stat );
}


/* <lalVerbatim file="TerrestrialCoordinatesCP"> */
void
LALHorizonToSystem( LALStatus   *stat,
		    SkyPosition *output,
		    SkyPosition *input,
		    const SkyPosition *zenith )
{ /* </lalVerbatim> */
  REAL8 sinP, cosP;       /* sin and cos of zenith latitude */
  REAL8 sina, cosa;       /* sin and cos of altitude */
  REAL8 sinA, cosA;       /* sin and cos of -azimuth */
  REAL8 sinD, sinH, cosH; /* sin and cos of declination and hour angle */

  INITSTATUS( stat, "LALHorizonToSystem", TERRESTRIALCOORDINATESC );

  /* Make sure parameter structures exist. */
  ASSERT( input, stat, SKYCOORDINATESH_ENUL, SKYCOORDINATESH_MSGENUL );
  ASSERT( output, stat, SKYCOORDINATESH_ENUL, SKYCOORDINATESH_MSGENUL );
  ASSERT( zenith, stat, SKYCOORDINATESH_ENUL, SKYCOORDINATESH_MSGENUL );

  /* Make sure we're given the right coordinate system. */
  if ( input->system != COORDINATESYSTEM_HORIZON ) {
    ABORT( stat, SKYCOORDINATESH_ESYS, SKYCOORDINATESH_MSGESYS );
  }
  if ( ( zenith->system != COORDINATESYSTEM_EQUATORIAL ) &&
       ( zenith->system != COORDINATESYSTEM_GEOGRAPHIC ) ) {
    ABORT( stat, SKYCOORDINATESH_ESYS, SKYCOORDINATESH_MSGESYS );
  }

  /* Compute intermediates. */
  sinP = sin( zenith->latitude );
  cosP = cos( zenith->latitude );
  sina = sin( input->latitude );
  cosa = cos( input->latitude );
  sinA = sin( input->longitude );
  cosA = cos( input->longitude );

  /* Compute components. */
  sinD = sina*sinP + cosa*cosA*cosP;
  sinH = cosa*sinA;
  cosH = sina*cosP - cosa*cosA*sinP;

  /* Compute final results. */
  output->system = zenith->system;
  output->latitude = asin( sinD );
  output->longitude = zenith->longitude - atan2( sinH, cosH );

  /* Optional phase correction. */
  if ( output->longitude < 0.0 )
    output->longitude += LAL_TWOPI;

  RETURN( stat );
}


/* <lalVerbatim file="TerrestrialCoordinatesCP"> */
void
LALGeodeticToGeocentric( LALStatus *stat, EarthPosition *location )
{ /* </lalVerbatim> */
  REAL8 c, s; /* position components in and orthogonal to the equator */
  REAL8 cosP, sinP; /* cosine and sine of latitude */
  REAL8 fFac;       /* ( 1 - f )^2 */
  REAL8 x, y;       /* Cartesian coordinates */
  REAL8 maxComp, r; /* max{x,y,z}, and sqrt(x^2+y^2+z^2) */

  INITSTATUS( stat, "LALGeodeticToGeocentric", TERRESTRIALCOORDINATESC );

  /* Make sure parameter structure exists. */
  ASSERT( location, stat, SKYCOORDINATESH_ENUL, SKYCOORDINATESH_MSGENUL );

  /* Make sure the geodetic coordinates are in the right system. */
  if ( location->geodetic.system != COORDINATESYSTEM_GEOGRAPHIC ) {
    ABORT( stat, SKYCOORDINATESH_ESYS, SKYCOORDINATESH_MSGESYS );
  }

  /* Compute intermediates. */
  fFac = 1.0 - LAL_EARTHFLAT;
  fFac *= fFac;
  cosP = cos( location->geodetic.latitude );
  sinP = sin( location->geodetic.latitude );
  c = sqrt( 1.0 / ( cosP*cosP + fFac*sinP*sinP ) );
  s = fFac*c;
  c = ( LAL_REARTH_SI*c + location->elevation )*cosP;
  s = ( LAL_REARTH_SI*s + location->elevation )*sinP;

  /* Compute Cartesian coordinates. */
  location->x = x = c*cos( location->geodetic.longitude );
  location->y = y = c*sin( location->geodetic.longitude );
  location->z = s;

  /* Compute the radius. */
  maxComp = x;
  if ( y > maxComp )
    maxComp = y;
  if ( s > maxComp )
    maxComp = s;
  x /= maxComp;
  y /= maxComp;
  s /= maxComp;
  r = sqrt( x*x + y*y + s*s );

  /* Compute the spherical coordinates, and exit. */
  location->radius = maxComp*r;
  location->geocentric.longitude = location->geodetic.longitude;
  location->geocentric.latitude = asin( s / r );
  location->geocentric.system = COORDINATESYSTEM_GEOGRAPHIC;
  RETURN( stat );
}


/* <lalVerbatim file="TerrestrialCoordinatesCP"> */
void
LALGeocentricToGeodetic( LALStatus *stat, EarthPosition *location )
{ /* </lalVerbatim> */
  REAL8 x, y, z;   /* normalized geocentric coordinates */
  REAL8 pi;        /* axial distance */

  /* Declare some local constants. */
  const REAL8 rInv = 1.0 / LAL_REARTH_SI;
  const REAL8 f1 = 1.0 - LAL_EARTHFLAT;
  const REAL8 f2 = LAL_EARTHFLAT*( 2.0 - LAL_EARTHFLAT );

  INITSTATUS( stat, "LALGeocentricToGeodetic", TERRESTRIALCOORDINATESC );

  /* Make sure parameter structure exists. */
  ASSERT( location, stat, SKYCOORDINATESH_ENUL, SKYCOORDINATESH_MSGENUL );

  /* See if we've been given a special set of coordinates. */
  x = rInv*location->x;
  y = rInv*location->y;
  z = rInv*location->z;
  pi = sqrt( x*x + y*y );
  location->geodetic.system = COORDINATESYSTEM_GEOGRAPHIC;
  location->geodetic.longitude = atan2( y, x );
  location->radius = LAL_REARTH_SI*sqrt( x*x + y*y + z*z );
  if ( pi == 0.0 ) {
    if ( z >= 0.0 ) {
      location->geodetic.latitude = LAL_PI_2;
      location->elevation = z - f1;
    } else {
      location->geodetic.latitude = -LAL_PI_2;
      location->elevation = -z - f1;
    }
    location->elevation *= LAL_REARTH_SI;
  }

  /* Do the general transformation even if z=0. */
  else {
    REAL8 za, e, f, p, p3, q, q2, d, d2, b, v, w, g, h, gh, t, phi, tanP;
    /* intermediate variables */

    /* See if we're inside the singular ellipsoid.
    if ( pi <= 1.01*f2 ) {
      REAL8 z1 = z*f1/f2;
      REAL8 p1 = pi/f2;
      if ( z1*z1 + p1*p1 < 1.02 ) {
	ABORT( stat, SKYCOORDINATESH_ESING, SKYCOORDINATESH_MSGESING );
      }
    } */

    /* Compute intermediates variables. */
    za = f1*fabs( z );
    e = za - f2;
    f = za + f2;
    p = ( 4.0/3.0 )*( pi*pi + za*za - f2*f2 );
    p3 = p*p*p;
    q = 8.0*pi*f2*za;
    q2 = q*q;
    h = p3/q2;
    d = p3 + q2;

    /* Compute v, using series expansion if necessary. */
    if ( d >= 0.0 ) {
      d2 = sqrt( d );
      b = q/d2;
      if ( fabs( h ) < LAL_HSERIES ) {
	if ( h < 0.0 )
	  v = pow( d2 + q, 1.0/3.0 )
	    + pow( -0.5*q*h*( 1.0 - 0.25*h*( 1.0 - 0.5*h ) ), 1.0/3.0 );
	else
	  v = pow( d2 + q, 1.0/3.0 )
	    - pow( 0.5*q*h*( 1.0 - 0.25*h*( 1.0 - 0.5*h ) ), 1.0/3.0 );
      } else if ( b < LAL_BSERIES ) {
	v = pow( d2, 1.0/3.0 )*( b*( 2.0/3.0 + b*b*10.0/81.0 ) );
      } else if ( b < 1.0 ) {
	v = pow( d2, 1.0/3.0 )*( pow( 1.0 + b, 1.0/3.0 ) -
				 pow( 1.0 - b, 1.0/3.0 ) );
      } else {
	v = pow( d2, 1.0/3.0 )*( pow( b + 1.0, 1.0/3.0 ) +
				 pow( b - 1.0, 1.0/3.0 ) );
      }
    } else {
      v = 2.0*sqrt( -p )*cos( acos( q/( -p*sqrt( -p ) ) )/3.0 );
    }

    /* Compute t, using series expansion if necessary. */
    h = v*pi/( e*e );
    w = fabs( e )*sqrt( 1.0 + h );
    if ( e > 0.0 || h > LAL_HSERIES )
      g = 0.5*( e + w );
    else
      g = -0.25*e*h*( 1.0 - 0.25*h*( 1.0 - 0.5*h ) );
    gh = fabs( pi*( f*pi - v*g )/w );
    h = gh/( g*g );
    if ( fabs( h ) > LAL_HSERIES )
      t = g*( sqrt( 1.0 + h ) - 1.0 );
    else
      t = 0.5*g*h*( 1.0 - 0.25*h*( 1.0 - 0.5*h ) );

    /* Compute latitude, longitude, and elevation. */
    tanP = ( pi - t )*( pi + t )/( 2.0*f1*pi*t );
    phi = atan( tanP );
    location->geodetic.latitude = phi;
    if ( z < 0.0 )
      location->geodetic.latitude *= -1.0;
    location->elevation = ( pi - t/pi )*cos( phi );
    location->elevation += ( fabs( z ) - f1 )*sin( phi );
    location->elevation *= LAL_REARTH_SI;
  }

  /* Transformation complete. */
  RETURN( stat );
}

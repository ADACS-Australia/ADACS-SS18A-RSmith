#include <string.h>
#include <lal/LALStdlib.h>
#include <lal/LALCache.h>
#include <lal/FileIO.h>

extern int lalDebugLevel;
int main( int argc, char *argv[] )
{
	LALFILE *outfile = NULL;
	LALCache *cache;
	int arg = 1;
	lalDebugLevel = 7;
	XLALSetErrorHandler( XLALExitErrorHandler );
	if ( argc > 1 && ! strcmp(argv[1],"-o") ) {
		outfile = XLALFileOpen( argv[2], "w" );
		arg += 2;
	}
	cache = XLALCacheGlob( NULL, argc == 1 ? NULL : argv[arg] );
	for ( ; arg < argc; ++arg ) {
		LALCache *tmp = cache;
		LALCache *add;
		add = XLALCacheGlob( NULL, argv[arg] );
		cache = XLALCacheMerge( tmp, add );
		XLALDestroyCache( add );
		XLALDestroyCache( tmp );
	}
	XLALCacheFileWrite( outfile ? outfile : lalstdout, cache );
	XLALFileClose( outfile );
	XLALDestroyCache( cache );
	LALCheckMemoryLeaks();
	return 0;
}

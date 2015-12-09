/*
*  Copyright (C) 2007 Vladimir Dergachev
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
#include <zlib.h>
#include <png.h>
#include "rastermagic.h"
#include "hookup.h"

extern char *output_dir;

void RGBPic_dump_png(char *filename, RGBPic *p)
{
FILE *fout;
long i,j;
char s[20000];
png_structp png_ptr;
png_infop info_ptr;
png_byte *row,*tmp;
unsigned char *r,*g,*b;

if(!clear_name_png(filename))return;

snprintf(s,20000,"%s%s", output_dir, filename);
fout=fopen(s, "wb");
if(fout==NULL){
	fprintf(stderr,"Error dumping %dx%d picture to \"%s\" in PNG format:",
		p->width, p->height, filename);
	perror("");
	return;
	}

png_ptr=png_create_write_struct(PNG_LIBPNG_VER_STRING, 
	(png_voidp)NULL, NULL, NULL);
if(!png_ptr){
	fclose(fout);
	return;
	}

info_ptr=png_create_info_struct(png_ptr);
if(!info_ptr){
	png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
	fclose(fout);
        return;
    	}
if(setjmp(png_jmpbuf(png_ptr))){
        png_destroy_write_struct(&png_ptr, &info_ptr);
        fclose(fout);
        return;
    	}
png_init_io(png_ptr, fout);
/* save disk space - we can afford to spend extra time compressing.. */
png_set_compression_level(png_ptr, Z_BEST_COMPRESSION);
png_set_IHDR(png_ptr, info_ptr, p->width, p->height,
       8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
       PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
/* png_set_text(png_ptr, info_ptr, text_ptr, num_text); <- set comment */
png_set_text(png_ptr, info_ptr, 
	(png_text[2]){
	{key:"Original filename", text:filename, compression: PNG_TEXT_COMPRESSION_NONE},
	{key:"Software", text:"PowerFlux " VERSION, compression: PNG_TEXT_COMPRESSION_NONE}
	}, 
	2);
png_write_info(png_ptr, info_ptr);

/* write image data here 
   we need to mangle it a bit to conform to format PNG expects.. */

row=aligned_alloca(3*p->width*sizeof(*row));
for(i=0;i<p->height;i++){
	tmp=row;
	r=p->red+p->stride*i;
	g=p->green+p->stride*i;
	b=p->blue+p->stride*i;
	for(j=0;j<p->width;j++){
		*tmp=*r;
		tmp++;
		*tmp=*g;
		tmp++;
		*tmp=*b;
		tmp++;
		r+=p->step;
		g+=p->step;
		b+=p->step;
		}
	png_write_rows(png_ptr, &row, 1);
	}

png_write_end(png_ptr, info_ptr);
png_destroy_write_struct(&png_ptr, &info_ptr);
fclose(fout);
}


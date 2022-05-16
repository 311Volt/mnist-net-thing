/*
A quick and dirty program for recognizing handwritten digits
Copyright (C) 2019 github.com/crizer6772

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
*/

#include "init.hpp"
#include <stdio.h>

bool a5wrap::full_init()
{
	char* err = NULL;
	if(!al_init())
		err = (char*)"al_init() failed";
	if(!al_install_keyboard())
		err = (char*)"al_install_keyboard() failed";
	if(!al_install_mouse())
		err = (char*)"al_install_mouse() failed";
	if(!al_install_audio())
		err = (char*)"al_install_audio() failed";
	if(!al_init_image_addon())
		err = (char*)"al_init_image_addon() failed";
	if(!al_init_acodec_addon())
		err = (char*)"al_init_acodec_addon() failed";
	if(!al_init_font_addon())
		err = (char*)"al_init_font_addon() failed";
	if(!al_init_primitives_addon())
		err = (char*)"al_init_primitives_addon() failed";
	if(!al_init_ttf_addon())
		err = (char*)"al_init_ttf_addon() failed";
	if(!al_init_video_addon())
		err = (char*)"al_init_video_addon() failed";
	if(!al_init_native_dialog_addon())
		err = (char*)"al_init_native_dialog_addon() failed";
	if(err)
    {
        puts(err);
		al_show_native_message_box(NULL, "a5 init failed", err, " ", NULL, ALLEGRO_MESSAGEBOX_ERROR);
    }
	return !err;
}

void a5wrap::clear_bitmap_to_color(ALLEGRO_BITMAP* bmp, ALLEGRO_COLOR col)
{
    ALLEGRO_BITMAP* tb = al_get_target_bitmap();
    al_set_target_bitmap(bmp);
    al_clear_to_color(col);
    al_set_target_bitmap(tb);
}

ALLEGRO_BITMAP* a5wrap::create_black_bitmap(int w, int h)
{
    ALLEGRO_BITMAP* out = al_create_bitmap(w,h);
    clear_bitmap_to_color(out, al_map_rgb(0,0,0));
    return out;
}

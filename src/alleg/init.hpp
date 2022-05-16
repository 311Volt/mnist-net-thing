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
#ifndef ALLEG_INIT
#define ALLEG_INIT

#include "../incl/a5.hpp"

namespace a5wrap
{
	bool full_init();
	void clear_bitmap_to_color(ALLEGRO_BITMAP* bmp, ALLEGRO_COLOR col);
	ALLEGRO_BITMAP* create_black_bitmap(int w, int h);
};

#endif // ALLEG_INIT

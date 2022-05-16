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
#ifndef APP_H
#define APP_H

#include "../alleg/init.hpp"
#include "../incl/full.hpp"
#include "../net/neuralnet.hpp"
#include "fftw3/fftw3.h"

namespace app
{

	class instance
	{
		class button
		{
		public:
			std::string label;
			int x,y,w,h;
			int state; //0-idle, 1-hover, 2-holding lmb
			float color, color_vis;
			bool unhandled_press;
			bool check_hover(int x, int y);
			button();
			button(int x, int y, int w, int h, const char* label);
			~button(){}
		};
		struct box
		{
			int x,y,w,h;
			box();
			box(int x, int y, int w, int h);
			~box(){}
		};
		ALLEGRO_EVENT_QUEUE *mouse_q, *keyb_q;
		ALLEGRO_MOUSE_STATE prev_mouse_state;
		ALLEGRO_FONT* font;
		ALLEGRO_FONT* font_s;
		std::string fontfn;
		std::map<std::string, ALLEGRO_BITMAP*> img;
		std::map<std::string, button> btn;
		std::valarray<float> net_res, net_res_vis;
		std::valarray<float> gaussian;
		ALLEGRO_DISPLAY *disp;
		neural::net digit_net;
		std::string digit_net_name;

		std::vector<std::string> aa_method_names;
		int aa_method;
		int neur_view_id;
		bool is_drawing;

		float *dct_in, *dct_out;
		float frame_pos, frame_pos_vis;
		double last_logic_tick_time, last_draft_downscale;
		fftwf_plan filter_plan, filter_reverse_plan;

		void neural_network_file_prompt();

		std::valarray<float> img2floatarr(ALLEGRO_BITMAP* bmp);
		std::pair<int,int> img_mass_center(ALLEGRO_BITMAP* bmp);
		box img_bounding_box(ALLEGRO_BITMAP* bmp);

		ALLEGRO_COLOR nv_map_float_to_color(float x);
		void render_neuron_view();

		void filter_canvas_sized_bitmap(ALLEGRO_BITMAP* bmp);

		void draw_line_on_canvas(int x0, int y0, int x1, int y1,
								ALLEGRO_COLOR col, float r);
		void mnist_downscale_high_quality();
		void mnist_downscale_draft();
		void mnistcanvas2net();

		void input_loop_tick();
		void logic_tick();
		void render_frame();

		void main_loop();

		float line_thiccness, max_fps;
		bool initialized, running;
	public:
		instance();
		~instance();

		bool init();
		bool deinit();
		bool run();
		bool stop();
	};


};

#endif // APP_H

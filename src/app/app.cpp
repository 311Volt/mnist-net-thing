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
#include "app.hpp"

#undef min
#undef max


app::instance::instance()
{
	initialized = false;
	running = false;
	line_thiccness = 20.0f;
	max_fps = 60.0f;
	net_res = net_res_vis = std::valarray<float>(0.0f, 10);
}
app::instance::~instance()
{
	deinit();
}

app::instance::box::box()
{
	x = y = w = h = 0;
}
app::instance::box::box(int x, int y, int w, int h)
{
	this->x = x;
	this->y = y;
	this->w = w;
	this->h = h;
}

app::instance::button::button()
{
	x = 0;
	y = 0;
	w = 0;
	h = 0;
	state = 0;
	label = std::string();
	color = color_vis = 0.2;
	unhandled_press = false;
}
app::instance::button::button(int x, int y, int w, int h, const char* label)
{
	this->x = x;
	this->y = y;
	this->w = w;
	this->h = h;
	this->label = std::string(label);
	state = 0;
	color = color_vis = 0.2;
	unhandled_press = false;
}
bool app::instance::button::check_hover(int x, int y)
{
	return(x>=this->x&&x<=this->x+w&&y>=this->y&&y<=this->y+h);
}


bool app::instance::init()
{
	disp = al_create_display(640,480);
	img["canvas"] = a5wrap::create_black_bitmap(308,308);
	img["mnistcanvas"] = a5wrap::create_black_bitmap(28,28);
	img["neuronview"] = a5wrap::create_black_bitmap(28,28);
	frame_pos = frame_pos_vis = -311.0f;
	if(!disp)
	{
		al_show_native_message_box(NULL, "error initializing app",
			"failed to create display"," ",NULL,ALLEGRO_MESSAGEBOX_ERROR);
		return false;
	}
	al_clear_to_color(al_map_rgb(0,0,0));
	al_flip_display();
	mouse_q = al_create_event_queue();
	keyb_q = al_create_event_queue();
	al_register_event_source(mouse_q, al_get_mouse_event_source());
	al_register_event_source(keyb_q, al_get_keyboard_event_source());
	al_get_mouse_state(&prev_mouse_state);

	ALLEGRO_CONFIG* cf = al_load_config_file("config.ini");
	if(cf)
	{
		max_fps = atof(al_get_config_value(cf, nullptr, "FPSLimit"));
		if(max_fps <= 0.0f || max_fps > 10000.0f)
			max_fps = 60.0f;
		font = al_load_font(al_get_config_value(cf, nullptr, "Font"), 15, 0);
		font_s = al_load_font(al_get_config_value(cf, nullptr, "Font"), 10, 0);
		if(!font)
		{
			al_show_native_message_box(NULL, "error initializing app",
				"font file invalid/not found","",NULL,ALLEGRO_MESSAGEBOX_ERROR);
			return false;
		}
		al_destroy_config(cf);
	}
	else
	{
		al_show_native_message_box(NULL, "error initializing app",
			"config.ini invalid/not found"," ",NULL,ALLEGRO_MESSAGEBOX_ERROR);
		return false;
	}
	digit_net = neural::net();
	digit_net_name = "(none)";
	al_get_mouse_state(&prev_mouse_state);

	dct_in = (float*)fftwf_malloc(308*308*sizeof(float));
	dct_out = (float*)fftwf_malloc(308*308*sizeof(float));

	filter_plan = fftwf_plan_r2r_2d(308,308,dct_in,dct_out,FFTW_REDFT10,FFTW_REDFT10,FFTW_MEASURE);
	filter_reverse_plan = fftwf_plan_r2r_2d(308,308,dct_out,dct_in,FFTW_REDFT01,FFTW_REDFT01,FFTW_MEASURE);

	gaussian = std::valarray<float>(308*308);
	for(int y=0; y<308; y++)
	{
		for(int x=0; x<308; x++)
		{
			float d = (x*x + y*y) / (28.0f*28.0f*2.0f);
			gaussian[y*308+x] = std::exp(-d);
		}
	}

	neur_view_id = 0;

	aa_method = 1;
	aa_method_names.push_back("sinc");
	aa_method_names.push_back("Gaussian");


	btn["loadnet"] = button(311, 170, 324, 22, "Load a neural network...");

	btn["aa_alg_l"] = button(311, 200, 30, 22, "<-");
	btn["aa_alg_r"] = button(605, 200, 30, 22, "->");

	btn["nv_l"] = button(450, 130, 30, 22, "<-");
	btn["nv_r"] = button(592, 130, 30, 22, "->");

	btn["bt_minus"] = button(311, 230, 30, 22, "-");
	btn["bt_plus"] = button(605, 230, 30, 22, "+");


	is_drawing = false;
	last_logic_tick_time = al_get_time();
	last_draft_downscale = al_get_time();
	initialized = true;
	return true;
}
bool app::instance::deinit()
{
	if(!initialized)
		return false;
	al_destroy_display(disp);
	for(auto& i : img)
		al_destroy_bitmap(i.second);
	al_destroy_event_queue(mouse_q);
	al_destroy_event_queue(keyb_q);
	initialized = running = false;
	return true;
}

void app::instance::neural_network_file_prompt()
{
	ALLEGRO_FILECHOOSER *fc;
	fc = al_create_native_file_dialog("nets", "Open a neural network", "*.net",
									ALLEGRO_FILECHOOSER_FILE_MUST_EXIST);
	bool succ = al_show_native_file_dialog(disp, fc);
	if(!succ)
		return;
	std::string fn(al_get_native_file_dialog_path(fc,0));
	neural::net tmpnet = neural::create_net_from_file(fn.c_str());
	if(tmpnet.get_num_neurons(0) == 784 &&
		tmpnet.get_num_neurons(tmpnet.get_num_layers()-1) == 10)
	{
		digit_net = tmpnet;
		digit_net_name = fn;
		neur_view_id = 0;
	}
	btn["loadnet"].state = 0;
	btn["loadnet"].color = btn["loadnet"].color_vis = 0.0f;
	last_logic_tick_time = al_get_time();
	render_neuron_view();
}
ALLEGRO_COLOR app::instance::nv_map_float_to_color(float x)
{
	//positive: g->r->b
	//negative: r->b->g
	float rf = 0.0f;
	float gf = 0.0f;
	float bf = 0.0f;
	float third = 1.0f/3.0f;
	uint8_t r=0,g=0,b=0;
	x = neural::sigmoid(x/3.0f) - 0.5f;
	x *= 2.0f;
	if(x >= 0)
	{
		gf = 3.0f*fmod(x, third);
		if(x > third)
		{
			gf = 1.0f;
			x -= third;
			rf = 3.0f*fmod(x, third);
			if(x > third)
			{
				rf = 1.0f;
				x -= third;
				bf = 3.0f*fmod(x, third);
			}
		}
	}
	else
	{
		x = -x;
		rf = 3.0f*fmod(x, third);
		if(x > third)
		{
			rf = 1.0f;
			x -= third;
			bf = 3.0f*fmod(x, third);
			if(x > third)
			{
				bf = 1.0f;
				x -= third;
				gf = 3.0f*fmod(x, third);
			}
		}
	}
	r = rf*255.0f;
	g = gf*255.0f;
	b = bf*255.0f;

	return al_map_rgb(r,g,b);
}
void app::instance::render_neuron_view()
{
	if(digit_net.get_num_layers() < 2 || digit_net.get_num_neurons(0) != 784)
	{
		a5wrap::clear_bitmap_to_color(img["neuronview"], al_map_rgb(0,0,0));
		return;
	}
	al_lock_bitmap(img["neuronview"], ALLEGRO_PIXEL_FORMAT_ANY, ALLEGRO_LOCK_READWRITE);
	ALLEGRO_BITMAP *prev_tb = al_get_target_bitmap();
	al_set_target_bitmap(img["neuronview"]);
	for(int y=0; y<28; y++)
	{
		for(int x=0; x<28; x++)
		{
			float w = digit_net.weights[1][neur_view_id][y*28+x];
			w /= neural::standard_deviation(digit_net.weights[1][neur_view_id]);
			al_put_pixel(x,y,nv_map_float_to_color(w));
		}
	}
	al_set_target_bitmap(prev_tb);
	al_unlock_bitmap(img["neuronview"]);
}
void app::instance::filter_canvas_sized_bitmap(ALLEGRO_BITMAP* bmp)
{
	if(al_get_bitmap_width(bmp) != 308)
		return;
	if(al_get_bitmap_height(bmp) != 308)
		return;
	std::valarray<float> bmpf = img2floatarr(bmp);
	bmpf -= 0.5f;
	bmpf *= 2.0f;
	memcpy(dct_in, &bmpf[0], 308*308*sizeof(float));
	fftwf_execute(filter_plan);
	for(int i=0; i<308; i++)
	{
		for(int j=0; j<308; j++)
		{
			if(aa_method == 0) //sinc
			{
				if(i<28 && j<28)
					continue;
				dct_out[i*308+j] = 0.0f;
			}
			if(aa_method == 1)
			{
				dct_out[i*308+j] *= gaussian[i*308+j];
			}
		}
	}

	fftwf_execute(filter_reverse_plan);
	al_lock_bitmap(bmp, ALLEGRO_PIXEL_FORMAT_ANY, ALLEGRO_LOCK_READWRITE);
	ALLEGRO_BITMAP* tb = al_get_target_bitmap();
	al_set_target_bitmap(bmp);
	for(int y=0; y<308; y++)
	{
		for(int x=0; x<308; x++)
		{
			float px_f = dct_in[y*308+x] / (308.0f*308.0f*4.0f);
			px_f = px_f*0.5f + 0.5f;
			px_f = std::max(px_f, 0.0f);
			px_f = std::min(px_f, 1.0f);
			if(px_f > 1.0f) {
				printf("%f at %d,%d\n",px_f,x,y);
			}
			uint8_t px = px_f * 255.0f;
			al_put_pixel(x, y, al_map_rgb(px,px,px));
		}
	}
	al_set_target_bitmap(tb);
	al_unlock_bitmap(bmp);
}

void app::instance::draw_line_on_canvas(int x0, int y0, int x1, int y1,
										ALLEGRO_COLOR col, float r)
{
	al_set_target_bitmap(img["canvas"]);
	al_draw_line(x0, y0, x1, y1, col, r);
	al_draw_filled_circle(x0, y0, r/2.0f, col);
	al_draw_filled_circle(x1, y1, r/2.0f, col);
	al_set_target_bitmap(al_get_backbuffer(disp));
}

std::valarray<float> app::instance::img2floatarr(ALLEGRO_BITMAP* bmp)
{
	int width = al_get_bitmap_width(bmp);
	int height = al_get_bitmap_height(bmp);
	std::valarray<float> out(width*height);
    al_lock_bitmap(bmp, ALLEGRO_PIXEL_FORMAT_ANY,
				ALLEGRO_LOCK_READONLY);
	for(int y=0; y<height; y++)
	{
		for(int x=0; x<width; x++)
		{
			uint8_t r,g,b;
			float rf,gf,bf,lum;
			al_unmap_rgb(al_get_pixel(bmp,x,y),&r,&g,&b);
			rf = (float)r / 255.0f;
			gf = (float)g / 255.0f;
			bf = (float)b / 255.0f;
			lum = 0.299*rf + 0.587*gf + 0.114*bf;
			out[y*width+x] = lum;
		}
	}
	al_unlock_bitmap(bmp);
	return out;
}


std::pair<int,int> app::instance::img_mass_center(ALLEGRO_BITMAP* bmp)
{
	int width = al_get_bitmap_width(bmp);
	int height = al_get_bitmap_height(bmp);
	float wsum_x = 0.0f;
	float wsum_y = 0.0f;
	std::valarray<float> imgf = img2floatarr(bmp);
	for(int y=0; y<height; y++)
	{
		for(int x=0; x<width; x++)
		{
			wsum_x += imgf[y*width+x]*x;
			wsum_y += imgf[y*width+x]*y;
		}
	}
	float mean_x = wsum_x / std::max(0.01f, imgf.sum());
	float mean_y = wsum_y / std::max(0.01f, imgf.sum());
	return std::pair<int,int>(mean_x,mean_y);
}

app::instance::box app::instance::img_bounding_box(ALLEGRO_BITMAP* bmp)
{
	box r;
	int width = al_get_bitmap_width(bmp);
	int height = al_get_bitmap_height(bmp);
	int x0 = width;
	int y0 = height;
	int x1 = 0;
	int y1 = 0;
	al_lock_bitmap(bmp, ALLEGRO_PIXEL_FORMAT_ANY, ALLEGRO_LOCK_READONLY);
	for(int y=0; y<height; y++)
	{
		for(int x=0; x<width; x++)
		{
			ALLEGRO_COLOR c = al_get_pixel(bmp, x, y);
			uint8_t r,g,b;
			al_unmap_rgb(c, &r, &g, &b);
			if(r || g || b)
			{
				x0 = std::min(x0, x);
				y0 = std::min(y0, y);
				x1 = std::max(x1, x);
				y1 = std::max(y1, y);
			}
		}
	}
	al_unlock_bitmap(bmp);
	if(x0 == width)
		return box(0,0,0,0);
	return box(x0, y0, x1-x0, y1-y0);
}

void app::instance::mnist_downscale_high_quality()
{
	ALLEGRO_BITMAP *canvas_pre = a5wrap::create_black_bitmap(308,308);
	ALLEGRO_BITMAP *canvas_unaligned = a5wrap::create_black_bitmap(220,220);
	ALLEGRO_BITMAP *prev_tb = al_get_target_bitmap();
	box bb = img_bounding_box(img["canvas"]);
	int sq = std::max(bb.w, bb.h);
	al_set_target_bitmap(canvas_unaligned);
	al_draw_scaled_bitmap(img["canvas"], bb.x, bb.y, sq, sq, 0, 0, 220, 220, 0);
	std::pair<int,int> mc = img_mass_center(canvas_unaligned);
	al_set_target_bitmap(canvas_pre);
	al_draw_bitmap(canvas_unaligned, 154-mc.first, 154-mc.second, 0);
	filter_canvas_sized_bitmap(canvas_pre);
	al_set_target_bitmap(img["mnistcanvas"]);
	al_draw_scaled_bitmap(canvas_pre, 0, 0, 308, 308, 0, 0, 28, 28, 0);
	//al_lock_bitmap(canvas_clone, ALLEGRO_PIXEL_FORMAT_ANY, ALLEGRO_LOCK_READONLY);
	//al_unlock_bitmap(canvas_clone);
	al_set_target_bitmap(prev_tb);
	al_destroy_bitmap(canvas_pre);
	al_destroy_bitmap(canvas_unaligned);
}
void app::instance::mnist_downscale_draft()
{
	ALLEGRO_BITMAP *tmp = a5wrap::create_black_bitmap(20,20);
	ALLEGRO_BITMAP *prev_tb = al_get_target_bitmap();
	al_set_target_bitmap(tmp);
	box bb = img_bounding_box(img["canvas"]);
	int sq = std::max(bb.w, bb.h);
	al_draw_scaled_bitmap(img["canvas"], bb.x, bb.y, sq, sq, 0, 0, 20, 20, 0);
	std::pair<int,int> mc = img_mass_center(tmp);
	al_set_target_bitmap(img["mnistcanvas"]);
	al_clear_to_color(al_map_rgb(0,0,0));
	al_draw_bitmap(tmp, 13-mc.first, 13-mc.second, 0);
	al_set_target_bitmap(prev_tb);
	al_destroy_bitmap(tmp);
}
void app::instance::mnistcanvas2net()
{
	if(digit_net.get_num_neurons(0) != 784)
		return;
	std::valarray<float> in = img2floatarr(img["mnistcanvas"]);
	if(in.sum() < 1.0f)
		net_res = 0.0f;
	else
		net_res = digit_net.feed_forward(in);
}

void app::instance::input_loop_tick()
{
	ALLEGRO_MOUSE_STATE cur_mouse_state;
	al_get_mouse_state(&cur_mouse_state);
	int prev_x = prev_mouse_state.x;
	int prev_y = prev_mouse_state.y;

	bool high_quality_downscale = false;
	bool need_redraw = false;

	while(!al_is_event_queue_empty(keyb_q))
	{
		ALLEGRO_EVENT ev;
		al_get_next_event(keyb_q, &ev);
		if(ev.type == ALLEGRO_EVENT_KEY_DOWN)
		{
			switch(ev.keyboard.keycode)
			{
			case ALLEGRO_KEY_ESCAPE:
				running = false;
				break;
			case ALLEGRO_KEY_ENTER:
				mnist_downscale_high_quality();
				mnistcanvas2net();
				a5wrap::clear_bitmap_to_color(img["canvas"], al_map_rgb(0,0,0));
				break;
			case ALLEGRO_KEY_F:
				//filter_canvas_sized_bitmap(img["canvas"]);
				break;
			case ALLEGRO_KEY_C:
				a5wrap::clear_bitmap_to_color(img["canvas"], al_map_rgb(0,0,0));
				a5wrap::clear_bitmap_to_color(img["mnistcanvas"], al_map_rgb(0,0,0));
				break;
			default:
				;
			}
		}
	}
	while(!al_is_event_queue_empty(mouse_q))
	{
		ALLEGRO_EVENT ev;
		al_get_next_event(mouse_q, &ev);

		/// *DRAWING* ///
		if((ev.type == ALLEGRO_EVENT_MOUSE_AXES ||
			ev.type == ALLEGRO_EVENT_MOUSE_BUTTON_DOWN) &&
			ev.mouse.x < 308 && ev.mouse.y < 308)
		{
			if(ev.type == ALLEGRO_EVENT_MOUSE_BUTTON_DOWN)
			{
				is_drawing = true;
			}
			int x0 = prev_x;
			int y0 = prev_y;
			int x1 = ev.mouse.x;
			int y1 = ev.mouse.y;

			if(prev_mouse_state.buttons&0x01 || cur_mouse_state.buttons&0x01)
			{
				need_redraw = true;
				draw_line_on_canvas(x0, y0, x1, y1, al_map_rgb(255,255,255), line_thiccness);
			}
			if(prev_mouse_state.buttons&0x02 || cur_mouse_state.buttons&0x02)
			{
				need_redraw = true;
				draw_line_on_canvas(x0, y0, x1, y1, al_map_rgb(0,0,0), line_thiccness*1.4f);
			}
			prev_x = ev.mouse.x;
			prev_y = ev.mouse.y;
		}
		if(ev.type == ALLEGRO_EVENT_MOUSE_BUTTON_UP)
		{
			if(ev.mouse.button == 1 || ev.mouse.button == 2)
			{
				high_quality_downscale = true;
				if(is_drawing)
					need_redraw = true;
				is_drawing = false;
			}
		}

		/// *BUTTONS* ///
		if(ev.type == ALLEGRO_EVENT_MOUSE_AXES)
		{
			for(auto& bp : btn)
			{
				button &b = bp.second;
				//printf("m_axes, hover=%d\n",b.check_hover(ev.mouse.x,ev.mouse.y));
				if(b.state == 0 && b.check_hover(ev.mouse.x, ev.mouse.y))
					b.state = 1;
				if(b.state == 1 && !b.check_hover(ev.mouse.x, ev.mouse.y))
					b.state = 0;
			}
		}
		if(ev.type == ALLEGRO_EVENT_MOUSE_BUTTON_DOWN)
		{
			for(auto& bp : btn)
			{
				button &b = bp.second;
				if(b.state != 2 && b.check_hover(ev.mouse.x, ev.mouse.y) && ev.mouse.button == 1)
					b.state = 2;
			}
		}
		if(ev.type == ALLEGRO_EVENT_MOUSE_BUTTON_UP)
		{
			for(auto& bp : btn)
			{
				button &b = bp.second;
				if(b.state == 2)
				{
					b.state = 0;
					if(b.check_hover(ev.mouse.x, ev.mouse.y) && ev.mouse.button == 1)
					{
						b.unhandled_press = 1;
						b.state = 1;
					}
				}
			}
		}
	}
	if(need_redraw)
	{
		if(high_quality_downscale)
		{
			mnist_downscale_high_quality();
		}
		else if(al_get_time() - last_draft_downscale > 1.0f/30.0f)
		{
			mnist_downscale_draft();
			last_draft_downscale = al_get_time();
		}
	}
	prev_mouse_state = cur_mouse_state;

}
void app::instance::logic_tick()
{

	/// *MOST LIKELY DIGIT* ///
	int mldigit = 0;
	float mloutput = 0.0f;
	for(int i=0; i<10; i++)
	{
		if(net_res[i] > mloutput)
		{
			mloutput = net_res[i];
			mldigit = i;
		}
	}
	frame_pos = 64*mldigit-32;

	/// *BUTTON COLORS* ///

	for(auto& bp : btn)
	{
		button &b = bp.second;
		switch(b.state)
		{
		case 0:
			b.color = 0.2f;
			break;
		case 1:
			b.color = 0.5f;
			break;
		case 2:
			b.color = 0.8f;
			break;
		}
	}

	/// *ANIMATIONS* ///
	float delta_t = (al_get_time()-last_logic_tick_time);
	std::valarray<float> dr = net_res - net_res_vis;
	dr *= delta_t;
	net_res_vis += dr * 16.0f;


	float dp = frame_pos - frame_pos_vis;
	dp *= delta_t;
	frame_pos_vis += dp * 16.0f;
	last_logic_tick_time = al_get_time();

	for(auto& bp : btn)
	{
		button &b = bp.second;
		float dc0 = b.color - b.color_vis;
		dc0 *= delta_t;
		b.color_vis += dc0 * 8.0f;
	}

	/// *BUTTON HANDLING* ///
	if(btn["loadnet"].unhandled_press)
	{
		btn["loadnet"].unhandled_press = 0;
		neural_network_file_prompt();
	}
	if(btn["aa_alg_l"].unhandled_press)
	{
		btn["aa_alg_l"].unhandled_press = 0;
		aa_method--;
		aa_method = (aa_method+int(aa_method_names.size()))%aa_method_names.size();
	}
	if(btn["aa_alg_r"].unhandled_press)
	{
		btn["aa_alg_r"].unhandled_press = 0;
		aa_method++;
		aa_method = (aa_method+int(aa_method_names.size()))%aa_method_names.size();
	}

	if(btn["bt_minus"].unhandled_press)
	{
		btn["bt_minus"].unhandled_press = 0;
		line_thiccness = std::max(1.0f, line_thiccness/1.2f);
	}
	if(btn["bt_plus"].unhandled_press)
	{
		btn["bt_plus"].unhandled_press = 0;
		line_thiccness = std::min(311.0f, line_thiccness*1.2f);
	}

	if(btn["nv_l"].unhandled_press)
	{
		btn["nv_l"].unhandled_press = 0;
		neur_view_id = std::max(0, neur_view_id-1);
		render_neuron_view();
	}
	if(btn["nv_r"].unhandled_press)
	{
		btn["nv_r"].unhandled_press = 0;
		neur_view_id = std::min((int)digit_net.get_num_neurons(1)-1, neur_view_id+1);
		render_neuron_view();
	}

}
void app::instance::render_frame()
{
	al_set_target_bitmap(al_get_backbuffer(disp));
	al_clear_to_color(al_map_rgb(15,15,15));
	al_draw_bitmap(img["canvas"],0,0,0);
	al_draw_rectangle(0, 0, 308, 308, al_map_rgb(160,160,160), 1);
	if(prev_mouse_state.x < 308 && prev_mouse_state.y < 308)
	{
		al_draw_circle(prev_mouse_state.x, prev_mouse_state.y,
						line_thiccness/2.0, al_map_rgb(0,0,255), 1);
	}

	int mgrid_x0 = 320;
	int mgrid_y0 = 15;
	ALLEGRO_COLOR mcgridcol = al_map_rgb(15,15,15);

	al_draw_scaled_bitmap(img["mnistcanvas"], 0, 0, 28, 28, mgrid_x0, mgrid_y0,
						28*4, 28*4, 0);
	for(int i=0; i<29; i++)
	{
		al_draw_line(mgrid_x0+4*i, mgrid_y0, mgrid_x0+4*i, mgrid_y0+28*4,
					mcgridcol, 1);
		al_draw_line(mgrid_x0, mgrid_y0+4*i, mgrid_x0+4*28, mgrid_y0+4*i,
					mcgridcol, 1);
	}
	al_draw_text(font, al_map_rgb(255,255,255), mgrid_x0+6, mgrid_y0+4*28+1,
					ALLEGRO_ALIGN_LEFT, "28x28 preview");

	int nvgrid_x0 = 480;
	int nvgrid_y0 = 15;
	ALLEGRO_COLOR nvgridcol = al_map_rgb(15,15,15);

	al_draw_scaled_bitmap(img["neuronview"], 0, 0, 28, 28, nvgrid_x0, nvgrid_y0,
						28*4, 28*4, 0);
	for(int i=0; i<29; i++)
	{
		al_draw_line(nvgrid_x0+4*i, nvgrid_y0, nvgrid_x0+4*i, nvgrid_y0+28*4,
					nvgridcol, 1);
		al_draw_line(nvgrid_x0, nvgrid_y0+4*i, nvgrid_x0+4*28, nvgrid_y0+4*i,
					nvgridcol, 1);
	}
	al_draw_text(font, al_map_rgb(255,255,255), nvgrid_x0+55, nvgrid_y0+4*28+1,
					ALLEGRO_ALIGN_CENTER, "Neuron view");
	if(digit_net_name != "(none)")
	{
		float sd = neural::standard_deviation(digit_net.bias[1]);
		float b = digit_net.bias[1][neur_view_id];
		float bc = b;
		if(b < 0)
			bc = std::min(-1.0f, b);
		else
			bc = std::max(1.0f, b);
		al_draw_textf(font_s, al_map_rgb(255,255,255), nvgrid_x0+55, nvgrid_y0+4*28+17,
						ALLEGRO_ALIGN_CENTER, "Hidden neuron %d/%d",
						neur_view_id, digit_net.get_num_neurons(1)-1);
		al_draw_textf(font_s, nv_map_float_to_color((3.0f*bc)/sd), nvgrid_x0+55, nvgrid_y0-13,
						ALLEGRO_ALIGN_CENTER, "Bias: %f", b);

	}


	for(int i=0; i<10; i++)
	{
		int sx = 64*i;
		int sy = 414;
		uint8_t c = 235.0*net_res_vis[i] + 20;
		uint8_t f = (c>127)?0:255;
		char digit[2] = {0,0};
		char amt[8] = {0,0,0,0,0,0,0,0};
		digit[0] = i+'0';
		sprintf_s(amt, sizeof(amt), "%.4f", net_res_vis[i]);
		al_draw_filled_rectangle(sx+2,sy+2,sx+62,sy+62,al_map_rgb(c,c,c)); //p
		al_draw_text(font, al_map_rgb(f,f,f), sx+32, sy+5,
					ALLEGRO_ALIGN_CENTER, digit);
		al_draw_text(font_s, al_map_rgb(f,f,f), sx+32, sy+35,
					ALLEGRO_ALIGN_CENTER, amt);

	}

	al_draw_rectangle(frame_pos_vis+32, 414, frame_pos_vis+96, 478,
					al_map_rgba(0,80,80,80), 2.0);
	/*al_draw_textf(font, al_map_rgb(255,255,255), mgrid_x0, mgrid_y0+4*28+12,
					ALLEGRO_ALIGN_LEFT, "canvas mass center: %d, %d",
					img_mass_center(img["canvas"]).first,
					img_mass_center(img["canvas"]).second);*/
	al_draw_text(font, al_map_rgba(60,60,60,150), 8, 280,
					ALLEGRO_ALIGN_LEFT,
					"Draw here");

	al_draw_text(font, al_map_rgb(255,255,255), 7, 308+5, ALLEGRO_ALIGN_LEFT,
				"LMB to draw, RMB to erase");
	al_draw_text(font, al_map_rgb(255,255,255), 7, 308+20, ALLEGRO_ALIGN_LEFT,
				"ENTER to submit image");
	al_draw_text(font, al_map_rgb(255,255,255), 7, 308+35, ALLEGRO_ALIGN_LEFT,
				"C to clear the canvas");

	char lsbuf[256];
	memset(lsbuf,0,256);
	for(unsigned i=0; i<digit_net.get_num_layers(); i++)
	{
		char buf1[80];
		memset(buf1,0,80);
		if(i+1 == digit_net.get_num_layers())
			sprintf_s(buf1, sizeof(buf1), "%d", digit_net.get_num_neurons(i));
		else
			sprintf_s(buf1, sizeof(buf1), "%d, ", digit_net.get_num_neurons(i));
		strcat_s(lsbuf, sizeof(buf1), buf1);
	}
	al_draw_textf(font, al_map_rgb(255,255,255), 470, 200, ALLEGRO_ALIGN_CENTER,
				"Downsampling: %s", aa_method_names[aa_method].c_str());
	al_draw_textf(font, al_map_rgb(255,255,255), 470, 230, ALLEGRO_ALIGN_CENTER,
				"Brush size: %.1f", line_thiccness);

	ALLEGRO_COLOR nnn_col = al_map_rgb(200,0,0);
	if(digit_net_name != "(none)")
		nnn_col = al_map_rgb(50,200,50);


	al_draw_textf(font, nnn_col, 7, 308+60, ALLEGRO_ALIGN_LEFT,
				"Currently loaded neural network: %s", digit_net_name.c_str());
	if(digit_net_name != "(none)")
	{
		al_draw_textf(font, al_map_rgb(160,160,160), 7, 308+75,
				ALLEGRO_ALIGN_LEFT,	"Neurons in each layer: %s", lsbuf);
	}




	/// *BUTTONS* ///
	for(auto& bp : btn)
	{
		button b = bp.second;
		uint8_t c = b.color_vis*255.0f;
		al_draw_filled_rectangle(b.x,b.y,b.x+b.w,b.y+b.h,al_map_rgb(c,c,c));
		al_draw_text(font,al_map_rgb(255,255,255),b.x+b.w/2,b.y,ALLEGRO_ALIGN_CENTER, b.label.c_str());
	}

	al_flip_display();
}

void app::instance::main_loop()
{
	if(!initialized)
		return;
	ALLEGRO_TIMER* fpstimer;
	ALLEGRO_EVENT_QUEUE* timer_q;
	fpstimer = al_create_timer(1.0/max_fps);
	timer_q = al_create_event_queue();
	al_register_event_source(timer_q, al_get_timer_event_source(fpstimer));
	al_start_timer(fpstimer);
	while(running)
	{
		ALLEGRO_EVENT x;
		while(!al_is_event_queue_empty(timer_q))
			al_drop_next_event(timer_q);
		al_wait_for_event(timer_q, &x);
		render_frame();
		logic_tick();
		input_loop_tick();
	}
	al_destroy_event_queue(timer_q);
	al_destroy_timer(fpstimer);
}

bool app::instance::run()
{
	if(!initialized)
		init();
	running = true;
	main_loop();
	return true;
}

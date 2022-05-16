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
#include "reader.hpp"

#define SWAP_UINT32(x) (((x) >> 24) | (((x) & 0x00FF0000) >> 8) | \
						 (((x) & 0x0000FF00) << 8) | ((x) << 24))
mnist::reader::reader()
{
	is_open = false;
}
mnist::reader::~reader()
{

}
bool mnist::reader::open(const char* img_fn, const char* lbl_fn)
{
	if(is_open)
	{
		images = std::vector<std::vector<uint8_t>>();
		labels = std::vector<uint8_t>();
	}
	is_open = true;
	file_img.open(img_fn, std::ios::binary);
	file_lbl.open(lbl_fn, std::ios::binary);
	if(!file_img.good() || !file_lbl.good())
		return false;
	unsigned mn_i, mn_l;
	file_img.read((char*)&mn_i, sizeof(unsigned));
	file_lbl.read((char*)&mn_l, sizeof(unsigned));
	file_img.read((char*)&num_examples, sizeof(unsigned));

	mn_i = SWAP_UINT32(mn_i);
	mn_l = SWAP_UINT32(mn_l);
	num_examples = SWAP_UINT32(num_examples);

	if(mn_i != 2051 || mn_l != 2049)
		return false;

	images = std::vector<std::vector<uint8_t>>(num_examples);
	for(auto& i : images)
		i = std::vector<uint8_t>(784);
	labels = std::vector<uint8_t>(num_examples);
	file_img.seekg(16);
	file_lbl.seekg(8);
	for(unsigned i=0; i<num_examples; i++)
	{
		file_img.read((char*)&images[i][0], 784);
		file_lbl.read((char*)&labels[i], 1);
	}
	file_img.close();
	file_lbl.close();
	return true;
}
bool mnist::reader::close()
{
	images = std::vector<std::vector<uint8_t>>();
	labels = std::vector<uint8_t>();
	is_open = false;
	num_examples = 0;
	return true;
}
std::valarray<float> mnist::reader::get_image(unsigned id)
{
	if(id>=num_examples || !is_open)
		return std::valarray<float>();
	float r = 1.0f / 255.0f;
	std::valarray<float> out(784);
	for(int i=0; i<784; i++)
		out[i] = (float)images[id][i] * r;
	return out;
}
unsigned mnist::reader::get_label(unsigned id)
{
	if(id>=num_examples || !is_open)
		return 311;
	return (unsigned)labels[id];
}

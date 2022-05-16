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
#ifndef MNIST_READER_H
#define MNIST_READER_H

#include "../incl/full.hpp"

namespace mnist
{
	class reader
	{
	private:
		bool is_open;
		unsigned num_examples;
		//std::string img_filename;
		//std::string lbl_filename;
		std::vector<std::vector<uint8_t>> images;
		std::vector<uint8_t> labels;
		std::ifstream file_img, file_lbl;
	public:
		reader();
		~reader();
		bool open(const char* img_fn, const char* lbl_fn);
		bool close();
		std::valarray<float> get_image(unsigned id);
		unsigned get_label(unsigned id);
	};
}

#endif // MNIST_READER_H

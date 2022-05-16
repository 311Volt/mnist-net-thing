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
#ifndef NET_NEURALNET
#define NET_NEURALNET

#include "../incl/c_cpp.hpp"

namespace neural
{
	class net
	{
	private:
		class backprop_layer_result
		{
		public:
			std::vector<std::valarray<float>> dc_dw; // cost/weight
			std::valarray<float> dc_dz; // cost/weighted_sum
		};
		class backprop_result
		{
		public:
			std::vector<backprop_layer_result> layer;
			float output_error;
			backprop_result();
			~backprop_result();
		};
		float feed_neuron(uint32_t layer,uint32_t num,std::valarray<float>& in);
		backprop_result empty_backprop_result();
		void reset_backprop_result(backprop_result& r);
		backprop_result r0,r1;
	public:
		std::vector<std::valarray<float>> weighted_sum, activation, bias;
		std::vector<std::vector<std::valarray<float>>> weights;

		net();
		net(std::vector<int> layer_sizes);
		~net();

		bool init(std::vector<int> layer_sizes);
		void randomize_neurons();
		uint32_t get_num_layers();
		uint32_t get_num_neurons(uint32_t layer);

		void print_net_state(FILE* f);

		bool save_to_file(const char* filename);
		bool load_from_file(const char* filename);

		std::valarray<float> feed_forward(std::valarray<float>& inputs);

		void run_with_backpropagation(
			backprop_result& out,
			std::valarray<float>& training_example,
			std::valarray<float>& desired_output);

		void adjust_neurons(backprop_result& amt,
							float factor);

		float train(std::vector<std::valarray<float>>& training_set,
					std::vector<std::valarray<float>>& desired_outputs,
					float learning_factor);
	};

	net create_net_from_file(const char* filename);

	double exp_custom(double x);
	float exp_custom_f(float x);
	float sigmoid(float x);
	float sigmoid_derivative(float x);

	std::valarray<float> sigmoid(std::valarray<float>& x);
	std::valarray<float> sigmoid_derivative(std::valarray<float>& x);
	std::valarray<float> random_valarray(uint32_t sz);
	std::valarray<float> random_valarray(uint32_t sz, float mean, float dev);

	float standard_deviation(std::valarray<float> x);

	template<class T>
	std::valarray<T> vec2valarray(std::vector<T> v);

	template<class T>
	std::vector<T> valarray2vec(std::valarray<T> v);

	void print_valarray_f(std::valarray<float> v);
};

#endif // NET_NEURALNET

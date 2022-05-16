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
#include "neuralnet.hpp"

uint32_t neural::net::get_num_layers() //for readability
{
	return activation.size();
}
uint32_t neural::net::get_num_neurons(uint32_t layer)
{
	if(layer >= get_num_layers())
		return 0;
	return activation[layer].size();
}

neural::net::net()
{
	//default: initializes a network of 2 layers with 1 neuron each
	init(std::vector<int>(2,1));
}
neural::net::net(std::vector<int> layer_sizes)
{
	init(layer_sizes);
}
neural::net::~net()
{

}

neural::net::backprop_result::backprop_result()
{
	output_error = 0.0f;
}
neural::net::backprop_result::~backprop_result()
{

}

float neural::net::feed_neuron(uint32_t layer,uint32_t num,
							std::valarray<float>& in)
{
	if(layer == 0)
		return 0;
	if(in.size() != get_num_neurons(layer-1))
		return 0;
	std::valarray<float> ws = in * weights[layer][num];
	weighted_sum[layer][num] = ws.sum() + bias[layer][num];
	activation[layer][num] = sigmoid(weighted_sum[layer][num]);
	return activation[layer][num];
}

bool neural::net::init(std::vector<int> layer_sizes)
{
	if(layer_sizes.size() < 2)
	{
		std::vector<int> x(2,1);
		init(x);
		return false;
	}
	bias = std::vector<std::valarray<float>>(layer_sizes.size());
	for(uint32_t i=0; i<layer_sizes.size(); i++)
		bias[i] = std::valarray<float>(0.0f, layer_sizes[i]);
	activation = bias;
	weighted_sum = bias;
	weights = std::vector<std::vector<std::valarray<float>>>(get_num_layers());
	for(uint32_t l=0; l<get_num_layers(); l++)
	{
		weights[l] = std::vector<std::valarray<float>>(get_num_neurons(l));
		if(l == 0)
			continue;
		for(uint32_t k=0; k<get_num_neurons(l); k++)
			weights[l][k] = std::valarray<float>(0.0f, get_num_neurons(l-1));
	}
	randomize_neurons();
	r0 = r1 = empty_backprop_result();
	return true;
}

void neural::net::randomize_neurons()
{
	for(uint32_t l=0; l<get_num_layers(); l++)
	{
		if(l > 0)
			for(uint32_t k=0; k<get_num_neurons(l); k++)
				weights[l][k] = random_valarray(get_num_neurons(l-1),
								0.0f, 1.0f/std::sqrt(get_num_neurons(l)));
		weighted_sum[l] = random_valarray(get_num_neurons(l));
		activation[l] = random_valarray(get_num_neurons(l));
		bias[l] = random_valarray(get_num_neurons(l));
	}
}

neural::net::backprop_result neural::net::empty_backprop_result()
{
	backprop_result r;
	r.output_error = 0.0f;
	r.layer.resize(get_num_layers());
	for(uint32_t l=0; l<get_num_layers(); l++)
	{
		r.layer[l].dc_dz = std::valarray<float>(0.0f, get_num_neurons(l));
		if(l == get_num_layers()-1)
			continue;
		r.layer[l].dc_dw=std::vector<std::valarray<float>>(get_num_neurons(l));
		for(uint32_t k=0; k<get_num_neurons(l); k++)
			r.layer[l].dc_dw[k]=std::valarray<float>(0.0f,get_num_neurons(l+1));
	}
	return r;
}

void neural::net::reset_backprop_result(neural::net::backprop_result& r)
{
	r.output_error = 0.0f;
	for(uint32_t l=0; l<get_num_layers(); l++)
	{
		r.layer[l].dc_dz = 0.0f;
		if(l == get_num_layers()-1)
			continue;
		for(uint32_t k=0; k<get_num_neurons(l); k++)
			r.layer[l].dc_dw[k]=0.0f;
	}
}

std::valarray<float> neural::net::feed_forward(std::valarray<float>& inputs)
{
	if(inputs.size() != get_num_neurons(0))
		return std::valarray<float>(0);
	for(uint32_t i=0; i<inputs.size(); i++)
		activation[0][i] = inputs[i];
	for(uint32_t l=1; l<get_num_layers(); l++)
		for(uint32_t k=0; k<get_num_neurons(l); k++)
			feed_neuron(l, k, activation[l-1]);
	uint32_t out_n = get_num_layers()-1;
	std::valarray<float> out(get_num_neurons(out_n));
	for(uint32_t i=0; i<get_num_neurons(out_n); i++)
		out[i] = activation[out_n][i];
	return out;
}

void neural::net::run_with_backpropagation(
			neural::net::backprop_result& out,
			std::valarray<float>& training_example,
			std::valarray<float>& desired_output)
{
	feed_forward(training_example);
	//backprop_result out = empty_backprop_result();
	uint32_t ll = get_num_layers()-1; //ll = output layer index
	std::valarray<float> mse = activation[ll] - desired_output;
	mse *= mse;
	out.output_error = mse.sum();
	out.layer[ll].dc_dz = activation[ll] - desired_output;
	for(uint32_t l=ll; l>0; l--)
	{
		auto& pl = out.layer[l-1]; 	//previous layer reference
		auto& cl = out.layer[l]; 	//current layer reference
		for(uint32_t k=0; k<get_num_neurons(l-1); k++)
		{
			for(uint32_t j=0; j<get_num_neurons(l); j++)
				pl.dc_dz[k] += weights[l][j][k]*cl.dc_dz[j];
			pl.dc_dw[k] = cl.dc_dz*activation[l-1][k];
			for(uint32_t j=0; j<get_num_neurons(l); j++)
				pl.dc_dw[k][j] += 0.00005f*weights[l][j][k];
		}
		pl.dc_dz *= sigmoid_derivative(weighted_sum[l-1]);
	}
	//return out;
}

void neural::net::adjust_neurons(backprop_result& amt, float factor)
{
	for(uint32_t l=1; l<get_num_layers(); l++)
		bias[l] -= amt.layer[l].dc_dz * factor;
	for(uint32_t l=1; l<get_num_layers(); l++)
		for(uint32_t k=0; k<get_num_neurons(l); k++)
			for(uint32_t j=0; j<get_num_neurons(l-1); j++)
				weights[l][k][j] -= amt.layer[l-1].dc_dw[j][k] * factor;
}

float neural::net::train(std::vector<std::valarray<float>>& training_set,
						std::vector<std::valarray<float>>& desired_outputs,
						float learning_factor)
{
	reset_backprop_result(r0);

	for(uint32_t i=0; i<training_set.size(); i++)
	{
		reset_backprop_result(r1);
		run_with_backpropagation(r1, training_set[i], desired_outputs[i]);
		for(uint32_t l=0; l<get_num_layers(); l++)
		{
			r0.layer[l].dc_dz += r1.layer[l].dc_dz;
			r0.output_error += r1.output_error / float(training_set.size());
			if(l==get_num_layers()-1)
				continue;
			for(uint32_t k=0; k<get_num_neurons(l); k++)
				r0.layer[l].dc_dw[k] += r1.layer[l].dc_dw[k];
		}
	}
	adjust_neurons(r0, learning_factor / float(training_set.size()));
	return r0.output_error;
}

double neural::exp_custom(double x)
{
	bool neg = x<0;
	if(neg) x = -x;
	double r = 1.0, x1, ep=2.7182818284590452;
	int fx = x;
	x -= fx;
	while(fx)
	{
		if(fx&1)
			r *= ep;
		ep *= ep;
		fx >>= 1;
	}
	if(x >= 0.5)
	{
		r *= 1.6487212707001281;
		x -= 0.5;
	}
	if(x >= 0.25)
	{
		r *= 1.2840254166877414;
		x -= 0.25;
	}
	if(x >= 0.125)
	{
		r *= 1.1331484530668263;
		x -= 0.125;
	}
	if(x >= 0.0625)
	{
		r *= 1.0644944589178594;
		x -= 0.0625;
	}
	x1 = x;
	double rf = 1.0+x1;
	x1 *= x;
	rf += x1*0.5;
	x1 *= x;
	rf += x1*0.1666666666666666;
	r *= rf;
	return (!neg) ? (r) : (1.0/r);
}

void neural::net::print_net_state(FILE* f)
{
	fprintf(f, "WEIGHTS:\n");
	for(uint32_t i=1; i<get_num_layers(); i++)
		for(uint32_t j=0; j<get_num_neurons(i); j++)
			for(uint32_t k=0; k<get_num_neurons(i-1); k++)
				fprintf(f, "l%d, %d->%d: %f\n",i,k,j,weights[i][j][k]);

	fprintf(f, "ACTIVATIONS:\n");
	for(uint32_t i=0; i<get_num_layers(); i++)
	{
		fprintf(f, "layer %d:\n", i);
		for(uint32_t j=0; j<get_num_neurons(i); j++)
			fprintf(f, "%.3f\t", activation[i][j]);
		fprintf(f, "\n\n");
	}

	fprintf(f, "BIASES:\n");
	for(uint32_t i=0; i<get_num_layers(); i++)
	{
		fprintf(f, "layer %d:\n", i);
		for(uint32_t j=0; j<get_num_neurons(i); j++)
			fprintf(f, "%.3f\t", bias[i][j]);
		fprintf(f, "\n\n");
	}

	fprintf(f, "WEIGHTED SUMS:\n");
	for(uint32_t i=0; i<get_num_layers(); i++)
	{
		fprintf(f, "layer %d:\n", i);
		for(uint32_t j=0; j<get_num_neurons(i); j++)
			fprintf(f, "%.3f\t", weighted_sum[i][j]);
		fprintf(f, "\n\n");
	}

}

bool neural::net::save_to_file(const char* filename)
{
	/** file format: (everything is little endian)
	  * 4 bytes: magic number (0xA113)
	  * 4 bytes: number of layers
	  * 4xL bytes: numbers of neurons in each layer
	  * then all the biases and weights (float32 little endian),
	  * in the order you'd expect them to be
	 **/
	std::ofstream out(filename, std::ios::binary);
	if(!out.good())
		return false;
	uint32_t magic_number = 0xA113, num_layers=get_num_layers();
	out.write((char*)&magic_number, sizeof(uint32_t));
	out.write((char*)&num_layers, sizeof(uint32_t));
	for(uint32_t i=0; i<num_layers; i++)
	{
		uint32_t num_neurons = get_num_neurons(i);
		out.write((char*)&num_neurons, sizeof(uint32_t));
	}
	for(uint32_t i=0; i<num_layers; i++)
		out.write((char*)&bias[i][0], get_num_neurons(i)*sizeof(float));
	for(uint32_t i=1; i<num_layers; i++)
		for(uint32_t j=0; j<get_num_neurons(i); j++)
			out.write((char*)&weights[i][j][0],get_num_neurons(i-1)*4);
	return true;
}

bool neural::net::load_from_file(const char* filename)
{
	std::ifstream in(filename, std::ios::binary);
	if(!in.good())
		return false;
	uint32_t mn, num_layers;
	in.read((char*)&mn, sizeof(uint32_t));
	in.read((char*)&num_layers, sizeof(uint32_t));
	if(mn != 0xA113)
		return false;
	std::vector<int> layer_sizes;
	layer_sizes.reserve(num_layers);
	for(uint32_t i=0; i<num_layers; i++)
	{
		uint32_t n;
		in.read((char*)&n, sizeof(uint32_t));
		layer_sizes.push_back(n);
	}
	init(layer_sizes);
	for(uint32_t i=0; i<num_layers; i++)
		in.read((char*)&bias[i][0], layer_sizes[i]*sizeof(float));
	for(uint32_t i=1; i<num_layers; i++)
		for(uint32_t j=0; j<(uint32_t)layer_sizes[i]; j++)
			in.read((char*)&weights[i][j][0], layer_sizes[i-1]*sizeof(float));
	return true;
}

neural::net neural::create_net_from_file(const char* filename)
{
	net out;
	out.load_from_file(filename);
	return out;
}

float neural::sigmoid(float x)
{
	return 1.0f/(1.0f+exp_custom(-x));
}

float neural::sigmoid_derivative(float x)
{
	float s = sigmoid(x);
	return s*(1.0f-s);
}

std::valarray<float> neural::sigmoid(std::valarray<float>& x)
{
	return x.apply(sigmoid);
}
std::valarray<float> neural::sigmoid_derivative(std::valarray<float>& x)
{
	std::valarray<float> out = sigmoid(x);
	return out*(1.0f-out);
}


std::valarray<float> neural::random_valarray(uint32_t sz, float mean, float dev)
{
    std::mt19937 gen;
    gen.seed(rand());
    std::normal_distribution<double> dist{mean,dev};
    std::valarray<float> out(sz);
    for(auto& i : out)
		i = dist(gen);
	return out;
}
std::valarray<float> neural::random_valarray(uint32_t sz)
{
    return random_valarray(sz, 0.0f, 1.0f);
}

float neural::standard_deviation(std::valarray<float> x)
{
	if(!x.size())
		return 0.0f;
	float mean = x.sum() / float(x.size());
	std::valarray<float> x1 = x - mean;
	x1 *= x1;
	return std::sqrt(x1.sum() / float (x1.size()));
}

template<class T>
std::valarray<T> neural::vec2valarray(std::vector<T> v)
{
	std::valarray<T> out(v.size());
	for(uint32_t i=0; i<v.size(); i++)
		out[i] = v[i];
	return out;
}

template<class T>
std::vector<T> neural::valarray2vec(std::valarray<T> v)
{
	std::vector<T> out(v.size());
	for(uint32_t i=0; i<v.size(); i++)
		out[i] = v[i];
	return out;
}

void neural::print_valarray_f(std::valarray<float> v)
{
	for(auto& i : v)
		printf("%f\t",i);
	puts("");
}

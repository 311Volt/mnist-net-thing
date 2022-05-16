#include "incl/full.hpp"
#include "alleg/init.hpp"
#include "net/neuralnet.hpp"
#include "mnist/reader.hpp"
#include "app/app.hpp"

int old_main(int argc, char** argv);
int real_main(int argc, char** argv);

int main(int argc, char** argv)
{
	return real_main(argc, argv);
}

int real_main(int argc, char** argv)
{
	/*  this function exists because all objects of classes that make use of
		allegro5 structures should not be created in the actual main function,
		as the library may destroy bitmaps, samples etc before the destructors
		have a chance to do so, causing a segfault */
	srand(time(NULL));
	if(!a5wrap::full_init())
		return 0;
	app::instance inst;
	inst.run();
	inst.deinit();
	//return 0;
	return old_main(argc, argv);
}

int old_main(int argc, char** argv)
{

	neural::net digit_net(std::vector<int>{784,60,10});
	mnist::reader train, t10k;
	train.open("mnist/train-images.idx3-ubyte", "mnist/train-labels.idx1-ubyte");
	t10k.open("mnist/t10k-images.idx3-ubyte", "mnist/t10k-labels.idx1-ubyte");

	//digit_net = neural::create_net_from_file("digitnet201906020511-9725.net");

	int batch_size = 20;
	std::mt19937 gen;
	gen.seed(rand());
	std::uniform_int_distribution<int> dist{0,59999};

	for(int epoch=0; epoch<100; epoch++)
	{

		std::vector<std::valarray<float>> training_set(batch_size);
		std::vector<std::valarray<float>> desired_outputs(batch_size);
		std::vector<int> selections(60000);
		for(int i=0; i<60000; i++)
			selections[i] = i;
		std::shuffle(selections.begin(), selections.end(), gen);
		int numBatches = 100;
		std::valarray<float> err(numBatches);
		for(int i=0; i<numBatches; i++)
		{
			for(int j=0; j<batch_size; j++)
			{
				int id = selections[i*batch_size+j];
				training_set[j] = train.get_image(id);
				desired_outputs[j] = std::valarray<float>(0.0f,10);
				desired_outputs[j][train.get_label(id)] = 1.0f;
			}
			err[i] = digit_net.train(training_set, desired_outputs,
										0.01f + 1.5f / (1.0f + epoch/10.0f));

		}
		float error = err.sum() / float(numBatches);
		int correct = 0;
		for(int i=0; i<10000; i++)
		{
			std::valarray<float> img = t10k.get_image(i);
			auto r = digit_net.feed_forward(img);
			float m=0;
			unsigned mn=0;
			for(unsigned j=0; j<10; j++)
			{
				if(r[j] > m) {
					m = r[j];
					mn = j;
				}
			}
			if(mn == t10k.get_label(i))
				correct++;
		}
		printf("epoch: %d, MSE %f (%d/%d)\n", epoch, error, correct, 10000);
	}
	digit_net.save_to_file("digitnet.net");


	return 0;
}

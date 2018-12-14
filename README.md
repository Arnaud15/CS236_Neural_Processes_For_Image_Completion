### CS236 Deep Generative Processes
## Neural Processes for Image Completion
#### Amaury Sabran, Arnaud Autef, Benjamin Petit

In this project, we develop image completion techniques based on Neural Processes
, a recently proposed class of models that uses neural networks to describe
distributions over functions. We show that the Neural Process model seamlessly
applies to the problem of image completion, explore different approaches to this
task and discuss their performance on two well-known datasets: MNIST and
CIFAR10.

For more details about Neural Processes, see https://arxiv.org/abs/1807.01613 and https://arxiv.org/abs/1807.01622.
For more details about the project, see the [project report](report.pdf).

To train a neural process on MNIST, run `python NP.py`. You can visualize the training and sample reconstructed images with tensorboard.
To train on CIFAR10, run `python NP_CIFAR10.py`.

Once you have a saved model in `models_saved/NP_model_epoch_x`, you can run `python complete_image_MNIST.py --resume_file models_saved/NP_model_epoch_x --mask_type upper` to generate image completions with the upper half of the image masked.

The code `test.py` was used to generate metrics in the report and should not be used.

This project was part of the course CS236 Deep Generative Models taught at Stanford University.
https://deepgenerativemodels.github.io/

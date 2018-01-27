Neural Networks in Kotlin
=========================

This project is my Neural Network playground in Kotlin. I'll be porting all my miscellaneous NN code from various codebases to this project.

# Neural Networks

- Autoencoder
- Stacked (Deep) Autoencoder
- Stacked (Deep) Convolution Autoencoder
- Feature Clustering via Autoencoder


# Training Algorithms

- Stochastic Gradient Descent
- Mini-batch size = 1


# MNIST Training Results

### Single Layer Autoencoder
Training Rate: 0.1
Steps: 100,000
Algorithm: Stochastic Gradient Descent
Error: ~3.5 (Squared error of pixel error per image, training data)

<a href="https://raw.githubusercontent.com/kennycason/neural_network_kotlin/master/results/data/mnist_auto_encoder_100k_steps_generated.png" target="new"><img src="https://raw.githubusercontent.com/kennycason/neural_network_kotlin/master/results/data/mnist_auto_encoder_100k_steps_generated_subset.png" width="400px"/></a><img src="https://raw.githubusercontent.com/kennycason/neural_network_kotlin/master/results/data/mnist_auto_encoder_100k_steps_error_graph.png" width="400px"/>

click image to see all 60,000k reconstructions.


### Single Layer Autoencoder
Training Rate: 0.05
Steps: 1,000,000
Algorithm: Stochastic Gradient Descent
Error: ~2.6 (Squared error of pixel error per image, training data)

<a href="https://raw.githubusercontent.com/kennycason/neural_network_kotlin/master/results/data/mnist_auto_encoder_1m_steps_generated.png" target="new"><img src="https://raw.githubusercontent.com/kennycason/neural_network_kotlin/master/results/data/mnist_auto_encoder_1m_steps_generated_subset.png" width="400px"/></a><img src="https://raw.githubusercontent.com/kennycason/neural_network_kotlin/master/results/data/mnist_auto_encoder_1m_steps_error_graph.png" width="400px"/>

click image to see all 60,000k reconstructions.

### Deep Autoencoder
Training Rate: 0.1
Steps: 250,000
Algorithm: Stochastic Gradient Descent
Error: ~5.6 (Squared error of pixel error per image, training data)

<a href="https://raw.githubusercontent.com/kennycason/neural_network_kotlin/master/results/data/mnist_deep_autoencoder_250k_steps_generated.png" target="new">
<img src="https://raw.githubusercontent.com/kennycason/neural_network_kotlin/master/results/data/mnist_deep_autoencoder_250k_steps_generated_subset.png" width="400px"/>
</a>

click image to see all 60,000k reconstructions.

# Random Vectors

Below are a few error graphs of a single layer autoencoder learning random vectors.

<img src="https://raw.githubusercontent.com/kennycason/neural_network_kotlin/master/results/data/100d_random_vector_auto_encoder_error_graph.png" width="400px"/>


# Feature Clustering

A byproduct of an autoencoder learning to encode features, is that through the encoding/compression process, feature clustering also occurs.

Below are examples of a deep autoencoder learning to encode input vectors.
The deepest most layer maps to a small 2-dimensional feature vector so that we can easily visualize the encoding on a x-y plot.

### Republican and Democrat Voting History Clustering

The deepest most layer maps to a small 2-dimensional feature vector so that we can easily visualize the encoding on a x-y plot.
There is a clear trend showing democratic votes clustered to the top right, and republican to the bottom left.
Unsurprisingly, there is also some overlap demonstrating cases where democrats and republicans share overlapping votes.

<img src="https://raw.githubusercontent.com/kennycason/neural_network_kotlin/master/results/data/deep_auto_encoder_voter_feature_clustering.png" width="450px"/>

### Non-Coding RNA Gene Clustering

<img src="https://raw.githubusercontent.com/kennycason/neural_network_kotlin/master/results/data/cod_rna_feature_clustering.png" width="450px"/>


### Deep Convolutional Autoencoder

Vanila Autoencoders work great for low dimensional data.
However, high dimensional data, such as images require a few tricks to effectively train.
By using a single autoencoder with a single fully-connected encode/decode weight matrix, we are effectively saying that every single pixel is directly correlated to every other pixel.
While this may be true, the relation of pixels of opposite corners, or large distances apart in an image are likely more *abstractly* related.
Asking a neural network to encode that many correlations in a single layer is a very steep request.
You may also discover that the weight matrices are too large to fit in memory, or too costly to compute.
A solution to this is to break each layer into pieces and train smaller networks on subsections of the image.
The separate networks are then connected together in subsequent layers.
This allows the network to learn simpler features first before combining in other layers to form more abstract features.
The idea is that pixels closer to each other are likely more immediately related to each other.
Additionally, it helps that this also results in smaller matrices which means faster computation and potential parellization.
Note that extra work was done to ensure spatial relations when processing 2d data is preserved when splitting/merging data between convolution layers.

Learning of a single small image.

<img src="https://raw.githubusercontent.com/kennycason/neural_network_kotlin/master/results/data/deep_convoluted_auto_encoder_jet.gif" width="300px"/>

Learning a batch of Pokemon Images. 
This may just be an illusion, but I find it interesting that the network appears to be learning the shape and structure before color. 
Note, I'm using a small network and a low number of training cycles. With more training this image will become more clear.

<img src="https://raw.githubusercontent.com/kennycason/neural_network_kotlin/master/results/data/deep_convoluted_auto_encoder_pokemon.png" /> <img src="https://raw.githubusercontent.com/kennycason/neural_network_kotlin/master/results/data/deep_convoluted_auto_encoder_pokemon.gif" />

Learning all 151 pokemon images. Note that the color model hasn't been fully learned. Also there are some strange clipping/white spots. These are largely due to editing issues with the training image on my part. :)

<img src="https://raw.githubusercontent.com/kennycason/neural_network_kotlin/master/results/data/deep_convoluted_auto_encoder_pokemon_151.png" width="400px" /> <img src="https://raw.githubusercontent.com/kennycason/neural_network_kotlin/master/results/data/deep_convoluted_auto_encoder_pokemon_151.gif" width="400px"/>




# Notes

- Since SGD, mini batch = 1, is noisy, the error signal will spike up and down wildly as it converages. A trick I employ is to take the raw errors and group them into batches and compute the average errors after the fact.
```awk '{sum+=$1} (NR%10)==0{print sum/10; sum=0;}' /tmp/raw_errors.log > /tmp/avg_erros.log```
You can also batch up groups and average in code.

- Converting everything from Double to Float was an instant 1.5x speedup. I don't suspect the GPU is actually being used and that is a pure Java speedup.

- I also don't recommend running running `mvn test` as unfortunately most of my code's entry points are via `@Test` annotations. :)

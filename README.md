Neural Networks in Kotlin
=========================

This project is my Neural Network playground in Kotlin. I'll be porting all my miscellaneous NN code from various codebases to this project.

# Neural Networks

- Autoencoder
- Stacked (Deep) Autoencoder
- Feature Clustering via Autoencoder
- Convolution (Coming within 1-2 days)


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

Below are a few error graphs of a single layer autoencoder learing random vectors.

<img src="https://raw.githubusercontent.com/kennycason/neural_network_kotlin/master/results/data/100d_random_vector_auto_encoder_error_graph.png" width="400px"/>


# Feature Clustering

A byproduct of an autoencoder learning to encode features, is that through the encoding/compression process, feature clustering also occurs.

Below are examples of a deep autoencoder learning to encode input vectors.
The deepest most layer maps to a small 2-dimensional feature vector so that we can easily visualize the encoding on a x-y plot.

#### Republican and Democrat Voting History Clustering

The deepest most layer maps to a small 2-dimensional feature vector so that we can easily visualize the encoding on a x-y plot.
There is a clear trend showing democratic votes clustered to the top right, and republican to the bottom left.
Unsurprisingly, there is also some overlap demonstrating cases where democrats and republicans share overlapping votes.

<img src="https://raw.githubusercontent.com/kennycason/neural_network_kotlin/master/results/data/deep_auto_encoder_voter_feature_clustering.png" width="450px"/>

#### Non-Coding RNA Gene Clustering

<img src="https://raw.githubusercontent.com/kennycason/neural_network_kotlin/master/results/data/cod_rna_feature_clustering.png" width="450px"/>

# Notes

- Since SGD, mini batch = 1, is noisy, the error signal will spike up and down wildly as it converages. A trick I employ is to take the raw errors and group them into batches and compute the average errors after the fact.
```awk '{sum+=$1} (NR%10)==0{print sum/10; sum=0;}' /tmp/raw_errors.log > /tmp/avg_erros.log```
You can also batch up groups and average in code.

- Converting everything from Double to Float was an instant 1.5x speedup. I don't suspect the GPU is actually being used and that is a pure Java speedup.

- I also don't recommend running running `mvn test` as unfortunately most of my code's entry points are view `@Test` annotations. :)

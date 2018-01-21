Neural Networks in Kotlin
=========================

This project is my Neural Network playground in Kotlin. I'll be porting all my miscellaneous NN code from various codebases to this project.

# Neural Networks

- Autoencoder
- Stacked (Deep) Autoencoder
- Hebbian Learning


# Training Algorithms

- Stochastic Gradient Descent
- Mini-batch size = 1



# MNIST Training Results

Training Rate: 0.1
Steps: 100,000
Algorithm: Stochastic Gradient Descent
Error: ~3.5 (Squared error of pixel error per image, training data)

Sample generated ouptut: (click [here](https://raw.githubusercontent.com/kennycason/neural_network_kotlin/master/results/data/mnist_auto_encoder_generated.png) to see all 60,000k reconstructions).

![MNIST Learned Subset](https://raw.githubusercontent.com/kennycason/neural_network_kotlin/master/results/data/mnist_auto_encoder_generated_subset.png)

![MNIST Error Graph](https://raw.githubusercontent.com/kennycason/neural_network_kotlin/master/results/data/mnist_auto_encoder_error_graph.png)

![MNIST Error Graph](https://raw.githubusercontent.com/kennycason/neural_network_kotlin/master/results/data/mnist_auto_encoder_1m_error_graph.png)

# Random Vectors

Below are a few error graphs of a single layer Auto Encoder learing random vectors.

![100d Random Vector Error Graph](https://raw.githubusercontent.com/kennycason/neural_network_kotlin/master/results/data/100d_random_vector_auto_encoder_error_graph.png)

![1000d Random Vector Error Graph](https://raw.githubusercontent.com/kennycason/neural_network_kotlin/master/results/data/1000d_random_vector_auto_encoder_error_graph.png)


# Notes

Since SGD, mini batch = 1, is noisy, the error signal will spike up and down wildly as it converages. A trick I employ is to take the raw errors and group them into batches and compute the average errors after the fact.

```awk '{sum+=$1} (NR%10)==0{print sum/10; sum=0;}' /tmp/raw_errors.log > /tmp/avg_erros.log```

You can also batch up groups and average in code.

I also don't recommend running running `mvn test` as unfortunately most of my code's entry points are view `@Test` annotations. :) 

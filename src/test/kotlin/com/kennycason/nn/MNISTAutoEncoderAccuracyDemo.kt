package com.kennycason.nn

import com.kennycason.nn.convolution.ConvolutedAutoEncoder
import com.kennycason.nn.convolution.Dim
import com.kennycason.nn.data.image.MNISTDataLoader
import com.kennycason.nn.learning_rate.FixedLearningRate
import com.kennycason.nn.math.Errors
import com.kennycason.nn.math.Functions
import org.jblas.FloatMatrix
import org.junit.Test

object MNISTAutoEncoderAccuracyDemo {

    /*
     * 1. Train autoencoder
     * 2. Train back prop nn on autoencoder encoded features
     * 3. Select strongest feature from nn as predicted class (0-9)
     *
     * Results:
     * single layer autoencoder, 2 layer nn = 91% accuracy
     * single layer convolution + fc autoencoder + 2 layer nn = 89.5%
     *
     */
    class experiment {

        @Test
        fun run() {
            val n = 60_000
            val xs = MNISTDataLoader.loadIdx3("/data/mnist/train-images-idx3-ubyte").subList(0, n)
            val labels = MNISTDataLoader.loadIdx1("/data/mnist/train-labels-idx1-ubyte").subList(0, n)
            val autoEncoder = trainAutoEncoder(xs)
            val features = xs.map { x -> autoEncoder.encode(x) }

            val labelVectors = buildLabelVectors(labels)
            val nn = trainBackPropNeuralNetwork(labelVectors, features)

            var errors = 0
            (0 until n).map { i ->
                val targetLabel = labels[i]
                val y = nn.feedForward(autoEncoder.encode(xs[i]))
                val estimatedLabel = selectClass(y)
                // println("$targetLabel, error ${Errors.compute(y, labelVectors[i])}, estimated class: $estimatedLabel")
                if (estimatedLabel != targetLabel) {
                    errors++
                }
            }
            println("errors: $errors")
            println("error %: ${errors.toFloat() / labels.size * 100.0}%")
        }

        private fun trainBackPropNeuralNetwork(labels: List<FloatMatrix>, features: List<FloatMatrix>): BackpropagationNeuralNetwork {
            val nn = BackpropagationNeuralNetwork(
                    learningRate = FixedLearningRate(0.05f),
                    layerSizes = arrayOf(
                            features.first().columns,
                            (features.first().columns * 1.75).toInt(),
                            features.first().columns / 2,
                            10),
                    log = true)

            (0..1000).forEach { i ->
                println("batch $i")
                nn.learn(xs = features, ys = labels, steps = 1000)
            }
            return nn
        }

        private fun trainAutoEncoder(xs: List<FloatMatrix>): DeepAutoEncoder {
            val visibleSize = 28 * 28
            val hiddenSize = (visibleSize * 1.50).toInt()
            val outputSize = (visibleSize * .75).toInt()

//        val layer = DeepAutoEncoder(
//                layerDimensions = arrayOf(
//                        arrayOf(visibleSize, hiddenSize),
//                        arrayOf(hiddenSize, outputSize)
//                ),
//                learningRate = 0.75f,
//                log = true)

//            val layer = AutoEncoder(
//                    visibleSize = visibleSize,
//                    hiddenSize = outputSize,
//                    learningRate = 0.1f,
//                    log = true)

            val layer1 = ConvolutedAutoEncoder(
                    visibleDim = Dim(28, 28),
                    hiddenDim = Dim(21, 21),
                    partitions = Dim(7, 7),
                    learningRate = FixedLearningRate(),
                    log = false
            )
            val layer3 = AutoEncoder(
                    visibleSize = 21 * 21,
                    hiddenSize = 150,
                    learningRate = FixedLearningRate(),
                    log = false)

            val layer = DeepAutoEncoder(
                    layers = arrayOf<AbstractAutoEncoder>(layer1, layer3),
                    log = true)

            layer.learn(xs, 100_000)

            val start = System.currentTimeMillis()
            println("${System.currentTimeMillis() - start}ms")
            return layer
        }

    }


    /*
     * Trivial clustering
     * 1. Train an auto encoder on MNIST data
     * 2. Generate hidden features for each of the trained data.
     * 3. Generate an average vector for each of the labeled training data hidden features
     * 4. Measure accuracy of each label based on which average vector it's closest to
     *
     * Result: 82% accuracy
     */
    object Experiment1 {

        @Test
        fun run() {
            val n = 60_000
            val xs = MNISTDataLoader.loadIdx3("/data/mnist/train-images-idx3-ubyte").subList(0, n)
            val labels = MNISTDataLoader.loadIdx1("/data/mnist/train-labels-idx1-ubyte").subList(0, n)
            println(labels.joinToString(","))

            val autoEncoder = trainAutoEncoder(xs)

            val features = xs.map { x -> autoEncoder.encode(x) }

            // this method only yields about 18% error
            val labelFeatureMap = partitionLabelsAndFeatures(labels, features)
            val labelAverageFeatureMap = calculateLabelAverageFeatures(labelFeatureMap)
            calculateFeatureError(labels, features, labelAverageFeatureMap)

        }

        private fun calculateFeatureError(labels: List<Int>, features: List<FloatMatrix>, labelAverageFeatureMap: Map<Int, FloatMatrix>) {
            var errors = 0
            (0 until labels.size).forEach { i ->
                val targetLabel = labels[i]
                val feature = features[i]
                // find most similar feature

                var closestLabel = -1
                var minError = Double.MAX_VALUE
                labelAverageFeatureMap.forEach { label, averageFeature ->
                    //  println("$label -> $averageFeature")
                    val error = Errors.compute(feature, averageFeature)
                    if (error < minError) {
                        closestLabel = label
                        minError = error
                    }
                }
                if (closestLabel != targetLabel) {
                    errors++
                    println("err, feature: $i -> label: $closestLabel was classified as $closestLabel")
                } else {
                    println("suc, feature: $i -> label: $closestLabel")
                }
            }
            println("errors: $errors")
            println("error %: ${errors.toFloat() / labels.size * 100.0}%")
        }

        private fun calculateLabelAverageFeatures(labelFeatureMap: Map<Int, List<FloatMatrix>>): Map<Int, FloatMatrix> {
            val labelAverageFeatureMap = mutableMapOf<Int, FloatMatrix>()
            labelFeatureMap.forEach { label, features ->
                labelAverageFeatureMap.put(label, features
                        .reduce { a, b -> a.add(b) }
                        .div(features.size.toFloat()))
            }

            return labelAverageFeatureMap
        }

        private fun partitionLabelsAndFeatures(labels: List<Int>, features: List<FloatMatrix>): Map<Int, List<FloatMatrix>> {
            val labelFeatureMap = mutableMapOf<Int, MutableList<FloatMatrix>>()
            (0 until labels.size).forEach { i ->
                if (!labelFeatureMap.containsKey(labels[i])) {
                    labelFeatureMap.put(labels[i], mutableListOf())
                }
                labelFeatureMap.get(labels[i])!!.add(features[i])
            }
            return labelFeatureMap
        }

        private fun trainAutoEncoder(xs: List<FloatMatrix>): AutoEncoder {
            val visibleSize = 28 * 28
            val hiddenSize = (visibleSize * .75).toInt()
            val layer = AutoEncoder(
                    visibleSize = visibleSize,
                    hiddenSize = hiddenSize,
                    learningRate = FixedLearningRate(),
                    log = true)

            val start = System.currentTimeMillis()
            layer.learn(xs, 100_000)
            println("${System.currentTimeMillis() - start}ms")
            return layer
        }
    }

    fun selectClass(label: FloatMatrix): Int {
        var bestClass = -1
        var bestError = 1.0f
        (0 until 10).forEach { i ->
            if (1.0f - label[i] < bestError) {
                bestClass = i
                bestError = 1.0f - label[i]
            }
        }
        return bestClass
    }

    fun buildLabelVectors(labels: List<Int>): List<FloatMatrix> {
        return labels.map { label ->
            val vector = FloatMatrix(1, 10)
            vector.put(label, 1.0f)
        }
    }

}
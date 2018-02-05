package com.kennycason.nn

import com.kennycason.nn.convolution.ConvolutedAutoEncoder
import com.kennycason.nn.convolution.Dim
import com.kennycason.nn.data.image.CompositeImageReader
import com.kennycason.nn.data.image.MatrixRGBImageEncoder
import com.kennycason.nn.learning_rate.FixedLearningRate
import com.kennycason.nn.math.Functions
import org.jblas.FloatMatrix
import org.junit.Test

class PokemonAutoEncoderAccuracyDemo {

    /*
     * 1. Train autoencoder
     * 2. Train back prop nn on autoencoder encoded features
     * 3. Select strongest feature from nn as predicted class (0-150)
     *
     * Results:
     *
     */
    @Test
    fun run() {
        val n = 151
        val xs = CompositeImageReader.read(
                file = "/data/image/pokemon_151_dark_bg.png",
                matrixImageEncoder = MatrixRGBImageEncoder(),
                rows = 11,
                cols = 15,
                n = n)
        val labels = (0 until n).toList()

        val labelVectors = buildLabelVectors(labels)
        val autoEncoder = trainAutoEncoder(xs)


        val features = xs.map { x -> autoEncoder.encode(x) }
        val nn = BackpropagationNeuralNetwork(
                learningRate = FixedLearningRate(),
                layerSizes = arrayOf(
                        features.first().columns,
                        features.first().columns / 2,
                        n),
                hiddenActivation = Functions.Sigmoid,
                outputActivation = Functions.Sigmoid,
                log = true)

        (0..100).forEach { i ->
            println("batch $i")
            nn.learn(xs = features, ys = labelVectors, steps = 100)
        }

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


    private fun trainAutoEncoder(xs: List<FloatMatrix>): DeepAutoEncoder {
        val layer1 = ConvolutedAutoEncoder(
                visibleDim = Dim(60 * 3, 60),
                hiddenDim = Dim(360, 120),
                partitions = Dim(60, 60),
                learningRate = FixedLearningRate(),
                log = false
        )
        val layer2 = ConvolutedAutoEncoder(
                visibleDim = Dim(360, 120),
                hiddenDim = Dim(180, 120),
                partitions = Dim(30, 30),
                learningRate = FixedLearningRate(),
                log = false
        )
        val layer3 = ConvolutedAutoEncoder(
                visibleDim = Dim(180, 120),
                hiddenDim = Dim(120, 60),
                partitions = Dim(20, 20),
                learningRate = FixedLearningRate(),
                log = false
        )

        val layer = DeepAutoEncoder(
                layers = arrayOf<AbstractAutoEncoder>(
                        layer1, layer2, layer3),
                log = true)

        layer.learn(xs, 100_000)

        val start = System.currentTimeMillis()
        println("${System.currentTimeMillis() - start}ms")
        return layer
    }



    private fun selectClass(label: FloatMatrix): Int {
        var bestClass = -1
        var bestError = 1.0f
        (0 until label.columns).forEach { i ->
            if (1.0f - label[i] < bestError) {
                bestClass = i
                bestError = 1.0f - label[i]
            }
        }
        return bestClass
    }

    private fun buildLabelVectors(labels: List<Int>): List<FloatMatrix> {
        return labels.map { label ->
            val vector = FloatMatrix(1, labels.size)
            vector.put(label, 1.0f)
        }
    }

}
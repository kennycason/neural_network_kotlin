package com.kennycason.nn

import com.kennycason.nn.convolution.ConvolutedAutoEncoder
import com.kennycason.nn.convolution.Dim
import com.kennycason.nn.data.image.MNISTDataLoader
import com.kennycason.nn.math.Errors
import com.kennycason.nn.math.Functions
import org.jblas.FloatMatrix

fun main(args: Array<String>) {
    MNISTBackErrorPropagatoinAccuracyTest.run()
}

object MNISTBackErrorPropagatoinAccuracyTest {

    /*
     * 1. Train back prop nn only
     * 2. Select strongest feature from nn as predicted class (0-9)
     *
     * Results:
     * errors: 695/60000 = 1.1583332903683186%
     */
    fun run() {
        val n = 60_000
        val xs = MNISTDataLoader.loadIdx3("/data/mnist/train-images-idx3-ubyte").subList(0, n)
        val labels = MNISTDataLoader.loadIdx1("/data/mnist/train-labels-idx1-ubyte").subList(0, n)


        val labelVectors = buildLabelVectors(labels)
        val nn = BackpropagationNeuralNetwork(
                learningRate = 0.1f,
                layerSizes = arrayOf(
                        28 * 28, // 784
                        300,
                        10),
                hiddenActivation = Functions.Sigmoid,
                outputActivation = Functions.Sigmoid,
                log = true)

        (0..1000).forEach { i ->
            println("batch $i")
            nn.learn(xs = xs, ys = labelVectors, steps = 1000)
        }

        var errors = 0
        (0 until n).map { i ->
            val targetLabel = labels[i]
            val y = nn.feedForward(xs[i])
            val estimatedLabel = selectClass(y)
            if (estimatedLabel != targetLabel) {
                errors++
            }
        }
        println("errors: $errors")
        println("error %: ${errors.toFloat() / labels.size * 100.0}%")
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
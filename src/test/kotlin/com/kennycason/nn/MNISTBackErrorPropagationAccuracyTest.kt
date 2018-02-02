package com.kennycason.nn

import com.kennycason.nn.data.PrintUtils
import com.kennycason.nn.data.image.MNISTDataLoader
import com.kennycason.nn.math.Functions
import org.jblas.FloatMatrix

fun main(args: Array<String>) {
    MNISTBackErrorPropagationAccuracyTest.run()
}

object MNISTBackErrorPropagationAccuracyTest {

    /*
     * 1. Train back prop nn only
     * 2. Select strongest feature from nn as predicted class (0-9)
     *
     * Results:
     * errors: 695/60000 = 1.1583332903683186% (1 million steps)
     * errors: 475/60000  = 0.7916666567325592% (2 million steps)
     */
    fun run() {
        val n = 60_000
        val xs = MNISTDataLoader.loadIdx3("/data/mnist/train-images-idx3-ubyte").subList(0, n)
        val labels = MNISTDataLoader.loadIdx1("/data/mnist/train-labels-idx1-ubyte").subList(0, n)
        val testXs = MNISTDataLoader.loadIdx3("/data/mnist/t10k-images-idx3-ubyte")
        val testLabels = MNISTDataLoader.loadIdx1("/data/mnist/t10k-labels-idx1-ubyte")

        val labelVectors = buildLabelVectors(labels)
        val nn = BackpropagationNeuralNetwork(
                learningRate = 0.15f,
                layerSizes = arrayOf(
                        28 * 28, // 784
                        350,
                        10),
                hiddenActivation = Functions.Sigmoid,
                outputActivation = Functions.Sigmoid,
                log = false)

        (0..1000).forEach { i ->
            println("batch $i")
            nn.learn(xs = xs, ys = labelVectors, steps = 1000)

            if (i % 100 == 0 && i > 0) {
                analyzeResults(nn, xs, labels, "train")
                analyzeResults(nn, testXs, testLabels, "test")
            }
        }
    }

    private fun analyzeResults(nn: BackpropagationNeuralNetwork, xs: List<FloatMatrix>, labels: List<Int>, name: String) {
        var errors = 0
        (0 until labels.size).map { i ->
            val targetLabel = labels[i]
            val y = nn.feedForward(xs[i])
            val estimatedLabel = selectClass(y)
            if (estimatedLabel != targetLabel) {
                errors++
            }
        }
        val error = errors.toFloat() / labels.size * 100.0
        println("$name errors: $errors")
        println("$name error %: $error%")
        println("$name accuracy %: ${100.0 - error}%")
    }

    private fun selectClass(label: FloatMatrix): Int {
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

    private fun buildLabelVectors(labels: List<Int>): List<FloatMatrix> {
        return labels.map { label ->
            val vector = FloatMatrix(1, 10)
            vector.put(label, 1.0f)
        }
    }

}
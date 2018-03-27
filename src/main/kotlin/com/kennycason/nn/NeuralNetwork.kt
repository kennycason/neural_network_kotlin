package com.kennycason.nn

import com.kennycason.nn.learning_rate.FixedLearningRate
import com.kennycason.nn.learning_rate.LearningRate
import com.kennycason.nn.math.ActivationFunction
import com.kennycason.nn.math.Errors
import com.kennycason.nn.math.Functions
import org.jblas.FloatMatrix
import java.util.*

/**
 * An implementation of a multi-layer neural network trained via back-error propagation
 */
class NeuralNetwork(layerSizes: Array<Int>,
                    var learningRate: LearningRate = FixedLearningRate(),
                    private val hiddenActivation: ActivationFunction = Functions.Sigmoid,
                    private val outputActivation: ActivationFunction = Functions.Sigmoid,
                    private val log: Boolean = true) {

    private val inputSize: Int
    private val outputSize: Int

    // layer of weight matrices
    // each row maps to a single neuron (note for matrix math orientation)
    private val layerWeights: Array<FloatMatrix>

    private val random = Random()

    init {
        if (layerSizes.size < 2) {
            throw IllegalArgumentException("Must at least provide input and output layer sizes")
        }
        layerWeights = (0 until layerSizes.size - 1).map { i ->
            val xSize = layerSizes[i]
            val ySize = layerSizes[i + 1]
            FloatMatrix.rand(xSize, ySize).mul(2.0f).sub(1.0f)
        }.toTypedArray()

        inputSize = layerSizes.first()
        outputSize = layerSizes.last()

        if (log) {
            println("layer dims: [${layerSizes.joinToString(",")}]")
        }
    }

    fun learn(xs: List<FloatMatrix>, ys: List<FloatMatrix>, steps: Int = 1000) {
        var totalError = 0.0
        (0..steps).forEach { i ->
            // sgd
            val j = random.nextInt(xs.size)

            val x = xs[j]   // input
            val y = ys[j]   // target
            learn(x, y, 1)

            // report error for current training data
//            if (i % 100 == 0 && log) {
//                val error = Errors.compute(y, feedForward(x))
//                println("$i -> error: $error")
//            }
            totalError += Errors.compute(y, feedForward(x))
        }
        if (log) {
            println("error for batch: ${totalError / steps}")
        }
    }

    fun learn(x: FloatMatrix, y: FloatMatrix, steps: Int = 10) {
        (1..steps).forEach { step ->

            // feed forward
            val yEstimatedWithFeatures = feedForwardWithFeatures(x)
            val yEstimated = yEstimatedWithFeatures.first
            val intermediateFeatures = yEstimatedWithFeatures.second

            // back propagate

            // calculate error from training signal on last layer
            val lastLayerWeights = layerWeights.last()
            // y, is the teacher signal (ideal output), yEstimated is our network's current guess.
            val yError = y.sub(yEstimated)
            val yDelta = yError
                    .mul(yEstimated.apply(outputActivation::df)) // error delta * derivative of activation function (sigmoid factored out)
                    .mul(learningRate.get())

            val contributingOutput = intermediateFeatures[intermediateFeatures.size - 2]
            val errorGradients = contributingOutput.transpose().mmul(yDelta)
            lastLayerWeights.addi(errorGradients)



            if (layerWeights.size == 1) { return@forEach }

            // propagate to previous layers
            var nextLayerDelta = yDelta

            // continue propagating error back to subsequent layers
            // calculate layer contribution to the next layer error, derive from weights (feature estimation error)
            (layerWeights.size - 2 downTo 0).forEach { i ->
                val currentLayerWeights = layerWeights[i]

                // determine how much layer contributed to the error in next layer).
                // TODO figure out why i have to alternate transposes...
                //val layerError = layerWeights[i + 1].mmul(nextLayerDelta).transpose()
                val layerError = when (layerWeights[i + 1].columns == nextLayerDelta.rows) {
                    true -> layerWeights[i + 1].mmul(nextLayerDelta).transpose()
                    false -> layerWeights[i + 1].mmul(nextLayerDelta.transpose()).transpose()
                }
                val layerOutput = intermediateFeatures[i + 1]
                val layerDelta = layerError
                        .mul(layerOutput.apply(hiddenActivation::df))

                // multiply errors by the contributing input
                val contributingOutput = intermediateFeatures[i]
                val layerErrorGradients = contributingOutput
                        .transpose()
                        .mmul(layerDelta)

                // update weights
                currentLayerWeights.addi(layerErrorGradients)

                nextLayerDelta = layerDelta
            }

        }
    }

    // only feed-forward to hidden (encoded) layer, return encoded feature
    fun feedForward(x: FloatMatrix): FloatMatrix {
        var currentFeature = x
        (0 until layerWeights.size).forEach { i ->
            val activationFunction = if (i == layerWeights.size - 1) { outputActivation::f } else { hiddenActivation::f }
            currentFeature = currentFeature.mmul(layerWeights[i]).apply(activationFunction)

        }
        return currentFeature
    }

    private fun feedForwardWithFeatures(x: FloatMatrix): Pair<FloatMatrix, List<FloatMatrix>> {
        val features = mutableListOf<FloatMatrix>()
        features.add(x)
        var currentFeature = x
        (0 until layerWeights.size).forEach { i ->
            val activationFunction = if (i == layerWeights.size - 1) { outputActivation::f } else { hiddenActivation::f }
            currentFeature = currentFeature.mmul(layerWeights[i]).apply(activationFunction)
            features.add(currentFeature)
        }
        return Pair(currentFeature, features)
    }

}
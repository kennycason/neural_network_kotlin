package com.kennycason.nn

import com.kennycason.nn.math.Errors
import com.kennycason.nn.math.Functions
import org.jblas.FloatMatrix
import java.util.*

/**
 * An implementation of a multi-layer neural network trained via back-error propagation
 */
class BackpropagationNeuralNetwork(inputSize: Int,
                                  // hiddenSize: Int,
                                   outputSize: Int,
                                   private val learningRate: Float = 0.1f,
                                   private val log: Boolean = true) {

    private val random = Random()

    // each row maps to a single neuron (note for matrix math orientation)
    val weights = FloatMatrix.rand(inputSize, outputSize).mul(2.0f).sub(1.0f)

    init {
        if (log) {
            println("input layer $inputSize x $outputSize")
        }
    }

    fun learn(xs: List<FloatMatrix>, ys: List<FloatMatrix>, steps: Int = 1000) {
        (0..steps).forEach { i ->
            // sgd
            val j = random.nextInt(xs.size)

            val x = xs[j]   // input
            val y = ys[j]   // target
            learn(x, y, 1)

            // report error for current training data TODO report rolling avg error
            if (i % 10 == 0 && log) {
                val error = Errors.compute(x, feedForward(x))
                println("$i -> error: $error")
            }
        }
    }

    fun learn(x: FloatMatrix, y: FloatMatrix, steps: Int = 10) {
        (1..steps).forEach { i ->

            // feed forward
            val yEstimated = feedForward(x)

            // back propagate

            // calculate error
            // y, is the teacher signal (ideal output), yEstimated is our network's current guess.
            val yErrors = y
                    .sub(yEstimated)
                    .mul(yEstimated.apply(Functions.sigmoidDerivative)) // error delta * derivative of activation function (sigmoid factored out)
                    .mul(learningRate)

            val errorGradients = yEstimated.transpose().mmul(yErrors)
           // println(errorGradients)
            weights.addi(errorGradients) // update weights

            // will do these on all subsequent layers
//            // calculate layer 2 contribution to the l1 error, derive from weights (feature estimation error)
//            val featureErrors = decode
//                    .mmul(yErrors.transpose())
//                    .transpose()    // two transposes on smaller matrices (yErrors / features errors is cheaper than performing on large decode matrix
//                    .mul(feature.apply(Functions.sigmoidDerivative)) // error delta * derivative of activation function (sigmoid factored out)
//            //  .mul(activationRate)

//            val encodeGradients = x.transpose().mmul(featureErrors)
//            encode.add(encodeGradients) // update weights

            if (log) {
                val error = Errors.compute(y, yEstimated)
                println("$i -> error: $error")
            }
        }
    }

    // only feed-forward to hidden (encoded) layer, return encoded feature
    fun feedForward(x: FloatMatrix) = x.mmul(weights).applyi(Functions.sigmoid)

}
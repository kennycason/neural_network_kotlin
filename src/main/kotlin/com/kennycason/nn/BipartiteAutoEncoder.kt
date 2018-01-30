package com.kennycason.nn

import com.kennycason.nn.math.ActivationFunction
import com.kennycason.nn.math.Errors
import com.kennycason.nn.math.Functions
import org.jblas.FloatMatrix
import java.util.*

/**
 * Attempt to train a single bipartite graph to be both discriminative and generative via SGD
 *
 * In progress, do not use
 */
class BipartiteAutoEncoder(visibleSize: Int,
                           hiddenSize: Int,
                           private val learningRate: Float = 0.1f,
                           private val activation: ActivationFunction = Functions.Sigmoid,
                           private val log: Boolean = true) : AbstractAutoEncoder() {

    private val random = Random()
    val weights: FloatMatrix

    init {
        if (visibleSize * hiddenSize <= 0) { // also checks for integer overflow
            throw RuntimeException("rows x columns exceeds integer size [$visibleSize x $hiddenSize = ${visibleSize * hiddenSize}]")
        }
        // each row maps to a single neuron (note for matrix math orientation)
        weights = FloatMatrix.rand(visibleSize, hiddenSize).mul(2.0f).sub(1.0f) // scale values between -1 and 1

        if (log) {
            println("weight dimensions $visibleSize x $hiddenSize")
        }
    }

    override fun learn(xs: List<FloatMatrix>, steps: Int) {
        var currentFeatures = xs

        (0..steps).forEach { i ->
            // sgd
            val x = currentFeatures[random.nextInt(currentFeatures.size)]
            learn(x, 1)

            // report error for current training data TODO report rolling avg error
            if (i % 100 == 0 && log) {
                val error = Errors.compute(x, feedForward(x))
                println("$i -> error: $error")
            }
        }
    }

    override fun learn(x: FloatMatrix, steps: Int) {
        (1.. steps).forEach { i ->
            // feed-forward
            val feature = encode(x)
            val y = decode(feature)

            // back propagate

            // calculate error from last layer
            // x, the input is also the teacher signal (ideal output), y is generated output
            val yErrors = x
                    .sub(y)
                    .mul(y.apply(activation::df)) // error delta * derivative of activation function (sigmoid factored out)
                    .mul(learningRate)

            val decodeGradients = feature.transpose().mmul(yErrors)
            weights.addi(decodeGradients) // update weights

            val decode = weights.transpose()
            // calculate layer 2 contribution to the l1 error, derive from weights (feature estimation error)
            val featureErrors = decode.mmul(yErrors.transpose())
                    .transpose()
                    .mul(feature.apply(activation::df)) // error delta * derivative of activation function (sigmoid factored out)

            val encodeGradients = x.transpose().mmul(featureErrors)
            decode.add(encodeGradients) // update weights

            if (i % 100 == 0 && log) {
                val error = Errors.compute(x, y)
                println("$i -> error: $error")
            }
        }
    }

    // only feed-forward to hidden (encoded) layer, return encoded feature
    override fun encode(x: FloatMatrix) = x.mmul(weights).apply(activation::f)

    // given an encoded feature, feed-forward through decoding weights to generate data
    override fun decode(feature: FloatMatrix) = feature.mmul(weights.transpose()).apply(activation::f)

    // full forward propagation
    override fun feedForward(x: FloatMatrix) = decode(encode(x))

}
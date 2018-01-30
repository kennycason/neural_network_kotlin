package com.kennycason.nn

import com.kennycason.nn.math.ActivationFunction
import com.kennycason.nn.math.Errors
import com.kennycason.nn.math.Functions
import org.jblas.FloatMatrix
import java.util.*

class AutoEncoder(visibleSize: Int,
                  hiddenSize: Int,
                  private val learningRate: Float = 0.1f,
                  private val hiddenActivation: ActivationFunction = Functions.Sigmoid,
                  private val visibleActivation: ActivationFunction = Functions.Sigmoid,
                  private val acktivation: ActivationFunction = Functions.Sigmoid,
                  private val log: Boolean = true) : AbstractAutoEncoder() {

    private val random = Random()
    val encode: FloatMatrix    // weight matrix that learns one level of encoding
    val decode: FloatMatrix    // weight matrix that learns one level of decoding

    init {
        if (visibleSize * hiddenSize <= 0) { // also checks for integer overflow
            throw RuntimeException("rows x columns exceeds integer size [$visibleSize x $hiddenSize = ${visibleSize * hiddenSize}]")
        }
        // each row maps to a single neuron (note for matrix math orientation)
        encode = FloatMatrix.rand(visibleSize, hiddenSize).mul(2.0f).sub(1.0f) // scale values between -1 and 1
        decode = FloatMatrix.rand(hiddenSize, visibleSize).mul(2.0f).sub(1.0f) // scale values between -1 and 1

        if (log) {
            println("encode layer $visibleSize x $hiddenSize, decode layer $hiddenSize x $visibleSize")
        }
    }

    override fun learn(xs: List<FloatMatrix>, steps: Int) {
        var currentFeatures = xs

        (0.. steps).forEach { i ->
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
                    .mul(y.apply(visibleActivation::df)) // error delta * derivative of hiddenActivation function (sigmoid factored out)
                    .mul(learningRate)

            val decodeGradients = feature.transpose().mmul(yErrors)
            decode.addi(decodeGradients) // update weights

            // calculate layer 2 contribution to the l1 error, derive from weights (feature estimation error)
            val featureErrors = decode
                    .mmul(yErrors.transpose())
                    .transpose()    // two transposes on smaller matrices (yErrors / features errors is cheaper than performing on large decode matrix
                    .mul(feature.apply(hiddenActivation::df)) // error delta * derivative of hiddenActivation function (sigmoid factored out)

            val encodeGradients = x.transpose().mmul(featureErrors)
            encode.add(encodeGradients) // update weights

            if (i % 100 == 0 && log) {
                val error = Errors.compute(x, y)
                println("$i -> error: $error")
            }
        }
    }

    // only feed-forward to hidden (encoded) layer, return encoded feature
    override fun encode(x: FloatMatrix) = x.mmul(encode).apply(hiddenActivation::f)

    // given an encoded feature, feed-forward through decoding weights to generate data
    override fun decode(feature: FloatMatrix) = feature.mmul(decode).apply(visibleActivation::f)

    // full forward propagation
    override fun feedForward(x: FloatMatrix) = decode(encode(x))

}
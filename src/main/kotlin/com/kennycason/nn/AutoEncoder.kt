package com.kennycason.nn

import com.kennycason.nn.math.Functions
import org.jblas.FloatMatrix

class AutoEncoder(visibleSize: Int,
                  hiddenSize: Int,
                  private val learningRate: Float = 0.1f,
                  private val log: Boolean = true) {

    val encode: FloatMatrix    // weight matrix that learns one level of encoding
    val decode: FloatMatrix    // weight matrix that learns one level of decoding

    init {
        if (visibleSize * hiddenSize <= 0) { // also checks for integer overflow
            throw RuntimeException("rows x columns exceeds integer size [$visibleSize x $hiddenSize = ${visibleSize * hiddenSize}]")
        }
        // each row maps to a single neuron (note for matrix math orientation)
        encode = FloatMatrix.rand(visibleSize, hiddenSize).mul(2.0f).sub(1.0f) // scale values between -1 and 1
        decode = FloatMatrix.rand(hiddenSize, visibleSize).mul(2.0f).sub(1.0f) // scale values between -1 and 1

        println("encode layer $visibleSize x $hiddenSize")
        println("decode layer $hiddenSize x $visibleSize")
    }

    fun learn(x: FloatMatrix, steps: Int = 10) {

        (1.. steps).forEach {
            // feed-forward
            val feature = encode(x)
            val y = decode(feature)

            // back propagate

            // calculate error from last layer
            // x, the input is also the teacher signal (ideal output), y is generated output
            val yErrors = x
                    .sub(y)
                    .mul(y.apply(Functions.sigmoidDerivative)) // error delta * derivative of activation function (sigmoid factored out)
                    .mul(learningRate)

            val decodeGradients = feature.transpose().mmul(yErrors)
            decode.addi(decodeGradients) // update weights

            // calculate layer 2 contribution to the l1 error, derive from weights (feature estimation error)
            val featureErrors = decode
                    .mmul(yErrors.transpose())
                    .transpose()    // two transposes on smaller matrices (yErrors / features errors is cheaper than performing on large decode matrix
                    .mul(feature.apply(Functions.sigmoidDerivative)) // error delta * derivative of activation function (sigmoid factored out)
                  //  .mul(activationRate)

            val encodeGradients = x.transpose().mmul(featureErrors)
            encode.add(encodeGradients) // update weights

            if (log) {
                val errors = x.sub(y)
                println(Math.sqrt(errors.mul(errors).sum().toDouble()))
            }
        }
    }

    // only feed-forward to hidden (encoded) layer, return encoded feature
    fun encode(x: FloatMatrix) = x.mmul(encode).applyi(Functions.sigmoid)

    // given an encoded feature, feed-forward through decoding weights to generate data
    fun decode(feature: FloatMatrix) = feature.mmul(decode).applyi(Functions.sigmoid)

    // full forward propagation
    fun feedForward(x: FloatMatrix) = decode(encode(x))

}
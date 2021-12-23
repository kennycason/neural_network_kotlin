package com.kennycason.nn.genetic

import com.kennycason.nn.apply
import com.kennycason.nn.applyi
import com.kennycason.nn.learning_rate.FixedLearningRate
import com.kennycason.nn.learning_rate.LearningRate
import com.kennycason.nn.math.ActivationFunction
import com.kennycason.nn.math.Errors
import com.kennycason.nn.math.Functions
import com.kennycason.nn.util.MatrixUtils
import org.jblas.FloatMatrix
import java.util.*

class GeneticAutoEncoder(var visibleSize: Int,
                         var hiddenSize: Int,
                         var learningRate: LearningRate = FixedLearningRate(),
                         private val hiddenActivation: ActivationFunction = Functions.Sigmoid,
                         private val visibleActivation: ActivationFunction = Functions.Sigmoid,
                         private val log: Boolean = true) {

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

    fun mutate(probability: Float): GeneticAutoEncoder {
        var i = 0
        encode.applyi { x ->
            if (random.nextFloat() < probability) {
                i++
                mutateValue(x)
            }
            else { x }
        }
        decode.applyi { x ->
            if (random.nextFloat() < probability) {
                i++
                mutateValue(x)
            }
            else { x }
        }
        // println("mutated $i weights")
        return this
    }

    private fun mutateValue(x: Float): Float {
        val deltaMutate = 0.1f // (random.nex() / 10f)
        if (random.nextBoolean()) {
            return x + deltaMutate
        }
        return x + -deltaMutate
    }

    fun copy(): GeneticAutoEncoder {
        val autoEncoder = GeneticAutoEncoder(visibleSize, hiddenSize, learningRate, hiddenActivation, visibleActivation, log)
        MatrixUtils.copyTo(encode, autoEncoder.encode)
        MatrixUtils.copyTo(decode, autoEncoder.decode)
        return autoEncoder
    }

    fun fitness(x: FloatMatrix) = Errors.compute(x, feedForward(x))

    // only feed-forward to hidden (encoded) layer, return encoded feature
    fun encode(x: FloatMatrix) = x.mmul(encode).apply(hiddenActivation::f)

    // given an encoded feature, feed-forward through decoding weights to generate data
    fun decode(feature: FloatMatrix) = feature.mmul(decode).apply(visibleActivation::f)

    // full forward propagation
    fun feedForward(x: FloatMatrix) = decode(encode(x))

}
package com.kennycason.nn

import com.kennycason.nn.math.Errors
import org.jblas.FloatMatrix
import java.util.*

/**
 * A generic deep auto encoder.
 * Can add both AutoEncoder or ConvolutedAutoEncoder or even other DeepAutoEncoders as
 * in individual layers
 */
class DeepAutoEncoder(private val layers: Array<AbstractAutoEncoder>,
                      private val learningRate: Float = 0.1f,
                      private val log: Boolean = false) : AbstractAutoEncoder() {

    private val random = Random()

    constructor(layerDimensions: Array<Array<Int>>,
                learningRate: Float = 0.1f,
                log: Boolean = false
                ) : this(
            layers = Array(
                    layerDimensions.size,
                    {i ->
                        AutoEncoder(
                                learningRate = learningRate,
                                visibleSize = layerDimensions[i][0],
                                hiddenSize = layerDimensions[i][1],
                                log = false)
                    }),
            learningRate = learningRate,
            log = log
    )

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
        var currentFeature = x
        layers.forEachIndexed { i, layer ->
            (0.. steps).forEach { j ->
                // sgd
                layer.learn(currentFeature, 1)

                // report error for current training data TODO report rolling avg error
//                if (j % 10 == 0 && log) {
//                    val error = Errors.compute(currentFeature, layer.feedForward(currentFeature))
//                    println("$j -> error: $error")
//                }
            }

            // generate encoded features to pass on to next layer
            currentFeature = layer.encode(currentFeature)
        }
    }

    override fun encode(x: FloatMatrix): FloatMatrix {
        var currentFeature = x
        // feed forward use hidden as input to next layer
        for (i in (0 until layers.size)) {
            currentFeature = layers[i].encode(currentFeature)
        }
        return currentFeature
    }

    override fun decode(feature: FloatMatrix): FloatMatrix {
        var currentFeature = feature
        // feed forward use hidden as input to next layer

        for (i in layers.size - 1 downTo 0) {
            currentFeature = layers[i].decode(currentFeature)
        }
        return currentFeature
    }

    override fun feedForward(x: FloatMatrix) = decode(encode(x))

}
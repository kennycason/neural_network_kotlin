package com.kennycason.nn.convolution

import com.kennycason.nn.AbstractAutoEncoder
import com.kennycason.nn.math.Errors
import org.jblas.FloatMatrix
import java.util.*

class DeepConvolutedAutoEncoder(private val layers: Array<ConvolutedAutoEncoder>,
                                private val log: Boolean = true) {

    private val random = Random()

    // learn layer by layer greedily
    fun learn(xs: List<FloatMatrix>, steps: Int) {
        var currentFeatures = xs
        layers.forEachIndexed { i, layer ->
            if (log) {
                println("training layer: ${i + 1}")
            }
            (0.. steps).forEach { j ->
                // sgd
                val x = currentFeatures[random.nextInt(currentFeatures.size)]
                layer.learn(x, 1)

                // report error for current training data
                if (j % 100 == 0 && log) {
                    val error = Errors.compute(x, layer.feedForward(x))
                    println("$j -> error: $error")
                }
            }
            // pass encoded features to next layer
            currentFeatures = currentFeatures.map { f -> layer.encode(f) }
            println(currentFeatures.first().columns)
        }
    }

    // learn all layers continuously
    fun learnContinuously(xs: List<FloatMatrix>, steps: Int) {
        (0.. steps).forEach { step ->
            val x = xs[random.nextInt(xs.size)]

            var currentFeature = x
            layers.forEachIndexed {i, layer ->
                // sgd
                layer.learn(currentFeature, 1)

                // report error for current training data TODO report rolling avg error
                if (step % 100 == 0 && log) {
                    val error = Errors.compute(currentFeature, layer.feedForward(currentFeature))
                    println("$step -> error: $error")
                }

                // generate encoded features to pass on to next layer
                currentFeature = layer.encode(currentFeature)
            }
        }
    }

    fun encode(x: FloatMatrix): FloatMatrix {
        var currentFeature = x
        // feed forward use hidden as input to next layer
        for (i in (0 until layers.size)) {
            currentFeature = layers[i].encode(currentFeature)
        }
        return currentFeature
    }

    fun decode(feature: FloatMatrix): FloatMatrix {
        var currentFeature = feature
        // feed forward use hidden as input to next layer

        for (i in layers.size - 1 downTo 0) {
            currentFeature = layers[i].decode(currentFeature)
        }
        return currentFeature
    }

    fun feedForward(x: FloatMatrix) = decode(encode(x))

}
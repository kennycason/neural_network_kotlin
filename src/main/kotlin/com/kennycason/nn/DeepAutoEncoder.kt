package com.kennycason.nn

import com.kennycason.nn.math.Errors
import org.jblas.FloatMatrix

class DeepAutoEncoder(layerDimensions: Array<Array<Int>>,
                      learningRate: Float = 0.1f,
                      private val log: Boolean = false) {

    private val layers: Array<AutoEncoder> = Array(
            layerDimensions.size,
            {i ->
                AutoEncoder(
                        learningRate = learningRate,
                        visibleSize = layerDimensions[i][0],
                        hiddenSize = layerDimensions[i][1],
                        log = log)
            })

    fun learn(xs: Collection<FloatMatrix>, steps: Int = 1000) {
        var currentFeatures = xs
        layers.forEachIndexed { i, layer ->
            if (log) {
                println("training layer: ${i + 1}")
            }

            (0.. steps).forEach { j ->
                // train each sample once
                currentFeatures.forEach { x ->
                    layer.learn(x, 1)
                }

                // report error
                if (j % 10 == 0) {
                    val error = currentFeatures
                            .map { x -> Errors.compute(x, layer.feedForward(x)) }
                            .sum() / currentFeatures.size
                    println("$j -> error: $error")
                }
            }

            // generate encoded features to pass on to next layer
            currentFeatures = currentFeatures
                    .map { x -> layer.encode(x) }
                    .toList()
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
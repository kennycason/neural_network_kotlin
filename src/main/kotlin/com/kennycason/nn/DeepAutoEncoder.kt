package com.kennycason.nn

import org.jblas.DoubleMatrix

class DeepAutoEncoder(layerDimensions: Array<Array<Int>>,
                      learningRate: Double = 0.1,
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

    fun learn(x: DoubleMatrix, steps: Int = 1000) {
        var currentFeature = x

        layers.forEachIndexed { i, layer ->
            if (log) {
                println("training layer: ${i + 1}")
            }

            layer.learn(currentFeature, steps)

            // feed forward to hidden nodes, use hidden as input to next layer
            currentFeature = layer.encode(currentFeature)
        }
    }

    fun encode(x: DoubleMatrix): DoubleMatrix {
        var currentFeature = x
        // feed forward use hidden as input to next layer
        for (i in (0 until layers.size)) {
            currentFeature = layers[i].encode(currentFeature)
        }
        return currentFeature
    }

    fun decode(feature: DoubleMatrix): DoubleMatrix {
        var currentFeature = feature
        // feed forward use hidden as input to next layer

        for (i in layers.size - 1 downTo 0) {
            currentFeature = layers[i].decode(currentFeature)
        }
        return currentFeature
    }

    fun feedForward(x: DoubleMatrix) = decode(encode(x))

}
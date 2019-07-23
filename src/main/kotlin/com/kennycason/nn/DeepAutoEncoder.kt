package com.kennycason.nn

import com.kennycason.nn.learning_rate.FixedLearningRate
import com.kennycason.nn.learning_rate.LearningRate
import com.kennycason.nn.math.ActivationFunction
import com.kennycason.nn.math.Errors
import com.kennycason.nn.math.Functions
import org.jblas.FloatMatrix
import java.util.*

/**
 * A generic deep auto encoder.
 * Can add both AutoEncoder or ConvolutedAutoEncoder layers
 */
class DeepAutoEncoder(val layers: Array<AbstractAutoEncoder>,
                      private val log: Boolean = false) {

    private val random = Random()

    constructor(layerDimensions: Array<Array<Int>>,
                learningRate: LearningRate = FixedLearningRate(),
                visibleActivation: ActivationFunction = Functions.Sigmoid,
                hiddenActivation: ActivationFunction = Functions.Sigmoid,
                log: Boolean = false
                ) : this(
            layers = Array(
                    layerDimensions.size
            ) { i ->
                AutoEncoder(
                        learningRate = learningRate,
                        visibleSize = layerDimensions[i][0],
                        hiddenSize = layerDimensions[i][1],
                        visibleActivation = visibleActivation,
                        hiddenActivation = hiddenActivation,
                        log = log)
            },
            log = log
    )

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
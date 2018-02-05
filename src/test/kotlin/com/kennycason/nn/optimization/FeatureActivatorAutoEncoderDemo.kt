package com.kennycason.nn.optimization

import com.kennycason.nn.AutoEncoder
import com.kennycason.nn.learning_rate.FixedLearningRate
import com.kennycason.nn.math.Errors
import org.jblas.FloatMatrix
import org.junit.Test

/**
 * a single test to verify that the neural network can still learn while activating inactive features
 */
class FeatureActivatorAutoEncoderDemo {

    @Test
    fun multipleVector() {
        val vectorSize = 1000
        val xs =  listOf(
                FloatMatrix.rand(1, vectorSize),
                FloatMatrix.rand(1, vectorSize),
                FloatMatrix.rand(1, vectorSize),
                FloatMatrix.rand(1, vectorSize),
                FloatMatrix.rand(1, vectorSize),
                FloatMatrix.rand(1, vectorSize)
        )

        val layer = AutoEncoder(
                learningRate = FixedLearningRate(0.2f),
                visibleSize = vectorSize,
                hiddenSize = vectorSize / 2,
                log = false)

        val featureActivator = FeatureActivator(layer)

        (0.. 10000).forEach { i ->
            xs.forEach { x ->
                layer.learn(x, 1)
                featureActivator.sample(x)
            }

            val error = Errors.compute(xs[0], layer.feedForward(xs[0]))
            featureActivator.update()
          //  println("$error")
            println("$error, ${featureActivator.generateInactivityVector().sum()}")
            featureActivator.clear()
        }
    }

}
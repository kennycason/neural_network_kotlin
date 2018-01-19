package com.kennycason.nn

import org.jblas.DoubleMatrix
import org.junit.Test


class FeatureActivatorAutoEncoderTest {

    @Test
    fun multipleVector() {
        val vectorSize = 1000
        val xs =  listOf(
                DoubleMatrix.rand(1, vectorSize),
                DoubleMatrix.rand(1, vectorSize),
                DoubleMatrix.rand(1, vectorSize),
                DoubleMatrix.rand(1, vectorSize),
                DoubleMatrix.rand(1, vectorSize),
                DoubleMatrix.rand(1, vectorSize)
        )

        val layer = AutoEncoder(
                learningRate = 0.2,
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
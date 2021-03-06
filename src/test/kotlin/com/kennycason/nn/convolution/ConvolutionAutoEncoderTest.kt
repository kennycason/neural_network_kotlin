package com.kennycason.nn.convolution

import com.kennycason.nn.apply
import com.kennycason.nn.learning_rate.FixedLearningRate
import org.jblas.FloatMatrix
import org.junit.Assert
import org.junit.Test


class ConvolutionAutoEncoderTest {

    @Test
    fun encodeDecodeTest() {
        val convolutedLayer = ConvolutedAutoEncoder(
                visibleDim = Dim(100, 100),
                hiddenDim = Dim(20, 20),
                partitions = Dim(10, 10),
                learningRate = FixedLearningRate(),
                log = true
        )

        val x = FloatMatrix.rand(100 * 100)

        convolutedLayer.learn(x, 10_000)

        println("\n\n")
        println(x)
        val y = convolutedLayer.decode(convolutedLayer.encode(x))
        println(y)

        val error = x.sub(y)
                .apply { i -> i * i }
                .sum() / x.length
        println("squared error: $error")

        Assert.assertTrue(error < 0.05)
    }

    @Test
    fun distributedTimes() {
        val layer = ConvolutedAutoEncoder(
                visibleDim = Dim(100, 100),
                hiddenDim = Dim(20, 20),
                partitions = Dim(10, 10),
                learningRate = FixedLearningRate(),
                log = false
        )
        val distributedLayer = ConvolutedAutoEncoder(
                visibleDim = Dim(100, 100),
                hiddenDim = Dim(20, 20),
                partitions = Dim(10, 10),
                learningRate = FixedLearningRate(),
                distributed = true,
                log = false
        )
        val start = System.currentTimeMillis()
        train(layer)

        val normalTime = System.currentTimeMillis() - start
        println("time to train normal: $normalTime")

        val start2 = System.currentTimeMillis()
        train(distributedLayer)

        val distributedTime = System.currentTimeMillis() - start2
        println("time to train distributed: ${distributedTime}")

        // sorry if you're running a computer with a single core :)
        Assert.assertTrue(distributedTime < normalTime)
    }

    private fun train(layer: ConvolutedAutoEncoder) {
        val x = FloatMatrix.rand(100 * 100)
        layer.learn(x, 10_000)
        val y = layer.decode(layer.encode(x))
        val error = x.sub(y)
                .apply { i -> i * i }
                .sum() / x.length
        println("squared error: $error")
        Assert.assertTrue(error < 0.1)
    }

}
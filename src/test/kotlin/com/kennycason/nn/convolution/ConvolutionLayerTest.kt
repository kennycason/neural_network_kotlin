package com.kennycason.nn.convolution

import com.kennycason.nn.apply
import org.jblas.FloatMatrix
import org.junit.Test


class ConvolutionLayerTest {

    @Test
    fun encodeDecodeTest() {
        val convolutedLayer = ConvolutedLayer(
                visibleDim = Dim(100, 100),
                hiddenDim = Dim(20, 20),
                paritions = Dim(10, 10),
                learningRate = 0.1f,
                log = true
        )

        val x = FloatMatrix.rand(100 * 100)

        convolutedLayer.learn(x, 10_000)


        println("\n\n")
        println(x)
        val y = convolutedLayer.decode(convolutedLayer.encode(x))
        println(y)

        println("squared error: " +
                x.sub(y)
                 .apply { i -> i * i }
                 .sum() / x.length)
    }

}
package com.kennycason.nn

import com.kennycason.nn.data.*
import org.jblas.DoubleMatrix
import org.junit.Test


class AutoEncoderTest {

    @Test
    fun randomVector() {
        val x = DoubleMatrix.rand(1, 10)

        val layer = AutoEncoder(learningRate = 0.1, visibleSize = 10, hiddenSize = 5, log = true)
        layer.learn(x, steps = 1000)

        println("input: " + x.toString("%f", "[", "]", ", ", "\n"))
        println("output: " + layer.feedForward(x).toString("%f", "[", "]", ", ", "\n"))
        println("error: ${Errors.compute(x, layer.feedForward(x))}")
    }

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

        val start = System.currentTimeMillis()
        (0.. 10000).forEach { i ->
            xs.forEach { x ->
                layer.learn(x, 1)
            }

            val error = Errors.compute(xs[0], layer.feedForward(xs[0]))
            if (i % 10 == 0) {
                println("$i -> error: $error")
            }
        }
        println("${System.currentTimeMillis() - start}ms")

        var errorSum = 0.0
        xs.forEach { x ->
            println("input: " + x.toString("%f", "[", "]", ", ", "\n"))
            println("output: " + layer.feedForward(x).toString("%f", "[", "]", ", ", "\n"))
            val error = Errors.compute(x, layer.feedForward(x))
            println("error: $error")
            errorSum += error
        }
        println("total error: " + (errorSum / xs.size))
    }

    @Test
    fun imageJet() {
        val image = Image("/data/ninja.png")
        val imageData = MatrixRGBImageEncoder().encode(image)

        val height = image.height()
        val width = image.width()
        println("$height x $width")

        val layer = AutoEncoder(learningRate = 0.9, visibleSize = imageData.columns, hiddenSize = imageData.columns / 2)

        val start = System.currentTimeMillis()
        (0.. 1000).forEach {
            layer.learn(imageData, 50)
            println("${System.currentTimeMillis() - start}ms")

            val visual = layer.feedForward(imageData)
            val outImage = MatrixRGBImageDecoder(rows = height).decode(visual)
            outImage.save("/tmp/output3.png")
        }
    }

    @Test
    fun imageMicro() {
        val image = Image("/data/micro.png")
        val imageData = MatrixRGBImageEncoder().encode(image)

        val height = image.height()
        val width = image.width()
        println("$height x $width")

        val layer = AutoEncoder(learningRate = 0.1, visibleSize = imageData.columns, hiddenSize = imageData.columns / 2)

        val start = System.currentTimeMillis()
        (0.. 10000).forEach {
            layer.learn(imageData, 10)
            println("${System.currentTimeMillis() - start}ms")

            val visual = layer.feedForward(imageData)
            val outImage = MatrixRGBImageDecoder(rows = height).decode(visual)
            outImage.save("/tmp/output.png")
            Thread.sleep(2000)
        }
    }

}

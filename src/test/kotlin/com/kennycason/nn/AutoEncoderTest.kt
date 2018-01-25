package com.kennycason.nn

import com.kennycason.nn.data.image.*
import com.kennycason.nn.math.Errors
import org.jblas.FloatMatrix
import org.junit.Test


class AutoEncoderTest {

    @Test
    fun randomVector() {
        val x = FloatMatrix.rand(1, 10)

        val layer = AutoEncoder(learningRate = 0.1f, visibleSize = 10, hiddenSize = 5, log = true)
        layer.learn(x, steps = 1000)

        println("input: " + x.toString("%f", "[", "]", ", ", "\n"))
        println("output: " + layer.feedForward(x).toString("%f", "[", "]", ", ", "\n"))
        println("error: ${Errors.compute(x, layer.feedForward(x))}")
    }

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
                learningRate = 0.1f,
                visibleSize = vectorSize,
                hiddenSize = vectorSize / 2,
                log = false)

        val start = System.currentTimeMillis()
        (0.. 10_000).forEach { i ->
            xs.forEach { x ->
                layer.learn(x, 1)
            }

            if (i % 10 == 0) {
                val error = xs
                        .map { x -> Errors.compute(x, layer.feedForward(x)) }
                        .sum() / xs.size
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
    fun imageNinja() {
        val image = Image("/data/ninja.png")
        val imageData = MatrixGrayScaleImageEncoder().encode(image)

        val height = image.height()
        val width = image.width()
        println("$height x $width")

        val layer = AutoEncoder(
                learningRate = 0.01f,
                visibleSize = imageData.columns,
                hiddenSize = (imageData.columns * 0.75).toInt()
        )

        val start = System.currentTimeMillis()
        (0.. 1000).forEach {
            layer.learn(imageData, 50)
            println("${System.currentTimeMillis() - start}ms")

            val visual = layer.feedForward(imageData)
            val outImage = MatrixGrayScaleImageDecoder(rows = height).decode(visual)
            outImage.save("/tmp/output_ninja2.png")
        }
    }

    @Test
    fun imageMicro() {
        val image = Image("/data/micro.png")
        val imageData = MatrixRGBImageEncoder().encode(image)

        val height = image.height()
        val width = image.width()
        println("$height x $width")

        val layer = AutoEncoder(
                learningRate = 0.1f,
                visibleSize = imageData.columns,
                hiddenSize = imageData.columns / 2)

        val start = System.currentTimeMillis()
        var i = 0
        (0.. 100).forEach {
            layer.learn(imageData, 10)
            println("${System.currentTimeMillis() - start}ms")

            val visual = layer.feedForward(imageData)
            val outImage = MatrixRGBImageDecoder(rows = height).decode(visual)
            outImage.save("/tmp/output${i++}.png")
        }
    }

}

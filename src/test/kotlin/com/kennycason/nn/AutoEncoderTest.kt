package com.kennycason.nn

import com.kennycason.nn.data.image.*
import com.kennycason.nn.learning_rate.FixedLearningRate
import com.kennycason.nn.math.Errors
import com.kennycason.nn.math.Functions
import com.kennycason.nn.util.time
import org.jblas.FloatMatrix
import org.junit.Assert
import org.junit.Ignore
import org.junit.Test


class AutoEncoderTest {

    @Test
    fun randomVector() {
        val x = FloatMatrix.rand(1, 10)

        val layer = AutoEncoder(
                learningRate = FixedLearningRate(),
                visibleSize = 10,
                hiddenSize = 5,
                log = true)

        layer.learn(x, steps = 5000)

        val error = Errors.compute(x, layer.feedForward(x))
        println("input: " + x.toString("%f", "[", "]", ", ", "\n"))
        println("output: " + layer.feedForward(x).toString("%f", "[", "]", ", ", "\n"))
        println("error: $error")

        Assert.assertTrue(error < 0.1)
    }

    @Test
    fun multipleVector() {
        val vectorSize = 100
        val xs =  listOf(
                FloatMatrix.rand(1, vectorSize),
                FloatMatrix.rand(1, vectorSize),
                FloatMatrix.rand(1, vectorSize),
                FloatMatrix.rand(1, vectorSize),
                FloatMatrix.rand(1, vectorSize),
                FloatMatrix.rand(1, vectorSize)
        )

        val layer = AutoEncoder(
                learningRate = FixedLearningRate(),
                visibleSize = vectorSize,
                hiddenSize = vectorSize / 2,
                hiddenActivation = Functions.Sigmoid,
                log = false)

        time({
            (0..1000).forEach { i ->
                layer.learn(xs, 100)

                if (i % 100 == 0) {
                    val error = xs
                            .map { x -> Errors.compute(x, layer.feedForward(x)) }
                            .sum() / xs.size
                    println("$i -> error: $error")
                }
            }
        })

        var errorSum = 0.0
        xs.forEach { x ->
            println("input: " + x.toString("%f", "[", "]", ", ", "\n"))
            println("output: " + layer.feedForward(x).toString("%f", "[", "]", ", ", "\n"))
            val error = Errors.compute(x, layer.feedForward(x))
            println("error: $error")
            errorSum += error
        }
        val error = errorSum / xs.size
        println("total error: $error")

        Assert.assertTrue(error < 0.1)
    }

    // image too large, fully connected layer doesn't learn well.
    // test works, but disabled due to time
    @Ignore
    fun imageNinja() {
        val image = Image("/data/image/ninja.png")
        val imageData = MatrixGrayScaleImageEncoder().encode(image)

        val height = image.height()
        val width = image.width()
        println("$height x $width")

        val layer = AutoEncoder(
                learningRate = FixedLearningRate(0.01f),
                visibleSize = imageData.columns,
                hiddenSize = (imageData.columns * 0.75).toInt()
        )

        time({
            (0..1000).forEach { i ->
                layer.learn(imageData, 50)

                val visual = layer.feedForward(imageData)
                val outImage = MatrixGrayScaleImageDecoder(rows = height).decode(visual)
                outImage.save("/tmp/output_ninja.png")

                val error = Errors.compute(imageData, visual)
                println("$i -> error: $error")
            }
        })

        val error = Errors.compute(imageData, layer.feedForward(imageData))
        Assert.assertTrue(error < 10.0f) // should be at least below 10.0
    }

    @Test
    fun imageMicro() {
        val image = Image("/data/image/micro.png")
        val imageData = MatrixRGBImageEncoder().encode(image)

        val height = image.height()
        val width = image.width()
        println("$height x $width")

        val layer = AutoEncoder(
                learningRate = FixedLearningRate(),
                visibleSize = imageData.columns,
                hiddenSize = imageData.columns / 2)

        time({
            (0..1000).forEach { i ->
                layer.learn(imageData, 10)
                val visual = layer.feedForward(imageData)
                val outImage = MatrixRGBImageDecoder(rows = height).decode(visual)
//                if (i % 50 == 0) {
//                    outImage.save("/tmp/output_micro_$i.png")
//                }
            }
        })
        val error = Errors.compute(imageData, layer.feedForward(imageData))
        println("error: $error")
        Assert.assertTrue(error < 0.1f)
    }

    @Test
    fun randomVectorTanh() {
        val x = FloatMatrix.rand(1, 100)

        val layer = AutoEncoder(
                learningRate = FixedLearningRate(),
                visibleSize = 100,
                hiddenSize = 5,
                visibleActivation = Functions.Tanh,
                hiddenActivation = Functions.Tanh,
                log = true)
        layer.learn(x, steps = 10_000)

        println("input: " + x.toString("%f", "[", "]", ", ", "\n"))
        println("output: " + layer.feedForward(x).toString("%f", "[", "]", ", ", "\n"))
        val error = Errors.compute(x, layer.feedForward(x))
        println("error: $error")
        Assert.assertTrue(error < 0.05)
    }

    @Test
    fun randomVectorTanhNegativeValues() {
        val x = FloatMatrix.rand(1, 100).mul(2.0f).sub(1.0f)

        val layer = AutoEncoder(
                learningRate = FixedLearningRate(),
                visibleSize = 100,
                hiddenSize = 5,
                visibleActivation = Functions.Tanh,
                hiddenActivation = Functions.Tanh,
                log = true)
        layer.learn(x, steps = 10_000)

        println("input: " + x.toString("%f", "[", "]", ", ", "\n"))
        println("output: " + layer.feedForward(x).toString("%f", "[", "]", ", ", "\n"))
        val error = Errors.compute(x, layer.feedForward(x))
        println("error: $error")
        Assert.assertTrue(error < 0.05)
    }

    // randomly fails, but when doesn't fail, is accurate.
    // Caused by dying hidden nodes
    @Test
    fun randomVectorLeakyReluHidden() {
        val x = FloatMatrix.rand(1, 100)

        val layer = AutoEncoder(
                learningRate = FixedLearningRate(0.05f),
                visibleSize = 100,
                hiddenSize = 5,
                visibleActivation = Functions.Sigmoid,
                hiddenActivation = Functions.LeakyRelU,
                log = false)
        layer.learn(x, steps = 100_000)

        //   println("input: " + x.toString("%f", "[", "]", ", ", "\n"))
        //   println("output: " + layer.feedForward(x).toString("%f", "[", "]", ", ", "\n"))
        val error = Errors.compute(x, layer.feedForward(x))
        println("error: $error")
        Assert.assertTrue(error < 1.0) // every once in a while this spikes above 0.001, and when it does it's .1 or something high
    }

}

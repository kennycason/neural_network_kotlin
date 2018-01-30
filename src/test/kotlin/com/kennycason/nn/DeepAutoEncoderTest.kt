package com.kennycason.nn

import com.kennycason.nn.math.Errors
import org.jblas.FloatMatrix
import org.junit.Assert
import org.junit.Test

class DeepAutoEncoderTest {

    @Test
    fun vector() {
        val xs = listOf(
                FloatMatrix.rand(1, 100)
        )

        val layer = DeepAutoEncoder(
                layerDimensions = arrayOf(
                        arrayOf(100, 75),
                        arrayOf(75, 50),
                        arrayOf(50, 25),
                        arrayOf(25, 10)
                ),
                learningRate = 0.1f,
                log = true)
        layer.learn(xs = xs, steps = 10_000)

        val x = xs.first()
        val error = Errors.compute(x, layer.feedForward(x))
        println("input: " + x.toString("%f", "[", "]", ", ", "\n"))
        println("output: " + layer.feedForward(x).toString("%f", "[", "]", ", ", "\n"))
        println("error: $error")
        println("deep feature: " + layer.encode(x).toString("%f", "[", "]", ", ", "\n"))

        Assert.assertTrue(error < 0.09)
    }

    @Test
    fun multipleVectors() {
        val vectorSize = 100
        val xs =  listOf(
                FloatMatrix.rand(1, vectorSize),
                FloatMatrix.rand(1, vectorSize),
                FloatMatrix.rand(1, vectorSize),
                FloatMatrix.rand(1, vectorSize),
                FloatMatrix.rand(1, vectorSize),
                FloatMatrix.rand(1, vectorSize)
        )

        val layer = DeepAutoEncoder(
                layerDimensions = arrayOf(
                        arrayOf(vectorSize, (vectorSize * 1.50).toInt()),
                        arrayOf((vectorSize * 1.50).toInt(), (vectorSize * 2.00).toInt()),
                        arrayOf((vectorSize * 2.00).toInt(), (vectorSize * 1.00).toInt()),
                        arrayOf((vectorSize * 1.00).toInt(), (vectorSize * 0.50).toInt())//,
                       // arrayOf((vectorSize * 0.25).toInt(), (vectorSize * 0.10).toInt())
                ),
                learningRate = 0.1f,
                log = true)

        // train
        val start = System.currentTimeMillis()
        layer.learn(xs = xs, steps = 100_000)
        println("${System.currentTimeMillis() - start}ms")

        // measure error
        val totalError = xs.map { x ->
            println("input: " + x.toString("%f", "[", "]", ", ", "\n"))
            println("deep feature: " + layer.encode(x).toString("%f", "[", "]", ", ", "\n"))
            println("output: " + layer.feedForward(x).toString("%f", "[", "]", ", ", "\n"))
            val error = Errors.compute(x, layer.feedForward(x))
            println("error: $error")
            error
        }.sum() / xs.size
        println("total error: $totalError")

        println()

        xs.map { x ->
            println("deep feature: " + layer.encode(x).toString("%f", "[", "]", ", ", "\n"))
        }
        Assert.assertTrue(totalError < 2.00) // it's typically below this
    }

}

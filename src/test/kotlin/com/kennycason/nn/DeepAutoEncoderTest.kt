package com.kennycason.nn

import com.kennycason.nn.math.Errors
import org.jblas.FloatMatrix
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
        println("input: " + x.toString("%f", "[", "]", ", ", "\n"))
        println("output: " + layer.feedForward(x).toString("%f", "[", "]", ", ", "\n"))
        println("error: ${Errors.compute(x, layer.feedForward(x))}")
        println("deep feature: " + layer.encode(x).toString("%f", "[", "]", ", ", "\n"))
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
                        arrayOf(vectorSize, (vectorSize * 0.75).toInt()),
                        arrayOf((vectorSize * 0.75).toInt(), (vectorSize * 0.50).toInt()),
                        arrayOf((vectorSize * 0.50).toInt(), (vectorSize * 0.25).toInt())//,
                       // arrayOf((vectorSize * 0.25).toInt(), (vectorSize * 0.10).toInt())
                ),
                learningRate = 0.1f,
                log = false)

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
    }

}

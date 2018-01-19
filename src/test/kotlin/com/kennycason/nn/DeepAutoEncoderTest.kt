package com.kennycason.nn

import org.jblas.DoubleMatrix
import org.junit.Test

class DeepAutoEncoderTest {

    @Test
    fun vector() {
        val x = DoubleMatrix.rand(1, 100)

        val layer = DeepAutoEncoder(
                layerDimensions = arrayOf(
                        arrayOf(100, 75),
                        arrayOf(75, 50),
                        arrayOf(50, 25),
                        arrayOf(25, 10),
                        arrayOf(10, 2)
                ),
                learningRate = 0.2,
                log = true)
        layer.learn(x, steps = 10_000)

        println("input: " + x.toString("%f", "[", "]", ", ", "\n"))
        println("deep feature: " + layer.encode(x).toString("%f", "[", "]", ", ", "\n"))
        println("output: " + layer.feedForward(x).toString("%f", "[", "]", ", ", "\n"))
        println("error: ${Errors.compute(x, layer.feedForward(x))}")
    }

    @Test
    fun multipleVectors() {
        val vectorSize = 100
        val xs =  listOf(
                DoubleMatrix.rand(1, vectorSize),
                DoubleMatrix.rand(1, vectorSize),
                DoubleMatrix.rand(1, vectorSize),
                DoubleMatrix.rand(1, vectorSize),
                DoubleMatrix.rand(1, vectorSize),
                DoubleMatrix.rand(1, vectorSize)
        )

        val layer = DeepAutoEncoder(
                layerDimensions = arrayOf(
                        arrayOf(100, 75),
                        arrayOf(75, 50),
                        arrayOf(50, 25)
                        //  arrayOf(25, 10)
                        // arrayOf(10, 2)
                ),
                learningRate = 0.1,
                log = false)

        val start = System.currentTimeMillis()
        (0.. 100_000).forEachIndexed { i, v ->
            xs.forEach { x ->
                //println(x)
                layer.learn(x, 1)
            }


            if (i % 1000 == 0) {
                val error = xs.map { x -> Errors.compute(x, layer.feedForward(x)) }
                        .sum() / xs.size
                println("$i -> error: $error")
            }
        }
        println("${System.currentTimeMillis() - start}ms")

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

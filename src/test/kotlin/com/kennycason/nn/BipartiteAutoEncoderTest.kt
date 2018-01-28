package com.kennycason.nn

import com.kennycason.nn.data.image.*
import com.kennycason.nn.math.Errors
import org.jblas.FloatMatrix
import org.junit.Ignore
import org.junit.Test

class BipartiteAutoEncoderTest {

    @Ignore
    fun randomVector() {
        val x = FloatMatrix.rand(1, 10)

        val layer = BipartiteAutoEncoder(learningRate = 0.1f, visibleSize = 10, hiddenSize = 5, log = true)
        layer.learn(x, steps = 1000)

        println("input: " + x.toString("%f", "[", "]", ", ", "\n"))
        println("output: " + layer.feedForward(x).toString("%f", "[", "]", ", ", "\n"))
        println("error: ${Errors.compute(x, layer.feedForward(x))}")
    }

    @Ignore
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

        val layer = BipartiteAutoEncoder(
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

}
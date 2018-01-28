package com.kennycason.nn

import com.kennycason.nn.data.image.*
import com.kennycason.nn.math.Errors
import org.jblas.FloatMatrix
import org.junit.Test

class BackpropagationNeuralNetworkTest {

    @Test
    fun simple() {
        val x = FloatMatrix(listOf(0.0f, 1.0f)).transpose()
        val y = FloatMatrix(listOf(0.0f))

        val layer = BackpropagationNeuralNetwork(
                learningRate = 0.5f,
                inputSize = 2,
                outputSize = 1,
                log = false)

        val start = System.currentTimeMillis()
        (0.. 1000).forEach { i ->
            layer.learn(x, y, 1)

            if (i % 10 == 0) {
                val error = Errors.compute(x, layer.feedForward(x))
                println("$i -> error: $error")
            }
        }
        println("${System.currentTimeMillis() - start}ms")


        println("$x -> " + layer.feedForward(x))
        val error = Errors.compute(y, layer.feedForward(x))
        println("error: $error")
    }

    @Test
    fun or() {
        val xs =  listOf(
                FloatMatrix(listOf(0.0f, 0.0f)).transpose(),
                FloatMatrix(listOf(0.0f, 1.0f)).transpose(),
                FloatMatrix(listOf(1.0f, 0.0f)).transpose(),
                FloatMatrix(listOf(1.0f, 1.0f)).transpose()
        )
        val ys =  listOf(
                FloatMatrix(listOf(0.0f)),
                FloatMatrix(listOf(1.0f)),
                FloatMatrix(listOf(1.0f)),
                FloatMatrix(listOf(1.0f))
        )

        val layer = BackpropagationNeuralNetwork(
                learningRate = 0.5f,
                inputSize = 2,
                outputSize = 1,
                log = false)

        val start = System.currentTimeMillis()
        (0.. 1000).forEach { i ->
            layer.learn(xs, ys, 1)

            if (i % 10 == 0) {
                val error = xs
                        .map { x -> Errors.compute(x, layer.feedForward(x)) }
                        .sum() / xs.size
                println("$i -> error: $error")
            }
        }
        println("${System.currentTimeMillis() - start}ms")

        var errorSum = 0.0
        xs.forEachIndexed { i, x ->
            println("$x -> " + layer.feedForward(x))
            val error = Errors.compute(ys[i], layer.feedForward(x))
            println("error: $error")
            errorSum += error
        }
        println("total error: " + (errorSum / xs.size))
    }

}
package com.kennycason.nn

import com.kennycason.nn.data.image.*
import com.kennycason.nn.math.Errors
import org.jblas.FloatMatrix
import org.junit.Assert
import org.junit.Test

class BackpropagationNeuralNetworkTest {

    @Test
    fun simple() {
        val x = FloatMatrix(listOf(0.0f, 1.0f)).transpose()
        val y = FloatMatrix(listOf(0.62f)) // a random target variable

        val layer = BackpropagationNeuralNetwork(
                learningRate = 0.5f,
                layerSizes = arrayOf(2, 1),
                log = false)

        val start = System.currentTimeMillis()
        (0.. 1000).forEach { i ->
            layer.learn(x, y, 1)

            if (i % 10 == 0) {
                val error = Errors.compute(y, layer.feedForward(x))
                println("$i -> error: $error")
            }
        }
        println("${System.currentTimeMillis() - start}ms")


        println("$x -> " + layer.feedForward(x))
        val error = Errors.compute(y, layer.feedForward(x))
        println("error: $error")
        Assert.assertTrue(error < 0.05)
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
                learningRate = 0.1f,
                layerSizes = arrayOf(2, 3, 1),
                log = false)

        val start = System.currentTimeMillis()
        (0.. 15_000).forEach { i ->
            layer.learn(xs, ys, 1)

            if (i % 10 == 0) {
                val error = xs
                        .mapIndexed { i, x -> Errors.compute(ys[i], layer.feedForward(x)) }
                        .sum() / xs.size
                println("$i -> error: $error")
            }
        }
        println("${System.currentTimeMillis() - start}ms")

        var errorSum = 0.0
        xs.forEachIndexed { i, x ->
            println("$x -> " + layer.feedForward(x))
            val error = Errors.compute(ys[i], layer.feedForward(x))
            errorSum += error
        }
        val error = errorSum / xs.size
        println("total error: $error")
        Assert.assertTrue(error < 0.09)
    }

    @Test
    fun xor() {
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
                FloatMatrix(listOf(0.0f))
        )

        val layer = BackpropagationNeuralNetwork(
                learningRate = 0.3f,
                layerSizes = arrayOf(2, 10, 1),
                log = false)

        val start = System.currentTimeMillis()
        (0.. 15_000).forEach { i ->
            layer.learn(xs, ys, 1)

            if (i % 10 == 0) {
                val error = xs
                        .mapIndexed { i, x -> Errors.compute(ys[i], layer.feedForward(x)) }
                        .sum() / xs.size
                println("$i -> error: $error")
            }
        }
        println("${System.currentTimeMillis() - start}ms")

        var errorSum = 0.0
        xs.forEachIndexed { i, x ->
            println("$x -> " + layer.feedForward(x))
            val error = Errors.compute(ys[i], layer.feedForward(x))
            errorSum += error
        }
        val error = errorSum / xs.size
        println("total error: $error")
        Assert.assertTrue(error < 0.09)
    }

    @Test
    fun deepRandom() {
        val xLen = 10
        val yLen = 2
        val xs =  listOf(
                FloatMatrix.rand(1, xLen),
                FloatMatrix.rand(1, xLen),
                FloatMatrix.rand(1, xLen),
                FloatMatrix.rand(1, xLen),
                FloatMatrix.rand(1, xLen)
        )
        val ys =  listOf(
                FloatMatrix.rand(1, yLen),
                FloatMatrix.rand(1, yLen),
                FloatMatrix.rand(1, yLen),
                FloatMatrix.rand(1, yLen),
                FloatMatrix.rand(1, yLen)
        )

        val layer = BackpropagationNeuralNetwork(
                learningRate = 0.1f,
                layerSizes = arrayOf(
                        xLen,
                        (xLen * 1.5).toInt(),
                        xLen * 2,
                        xLen,
                        yLen),
                log = false)

        val start = System.currentTimeMillis()
        (0.. 15_000).forEach { i ->
            layer.learn(xs, ys, 1)

            if (i % 10 == 0) {
                val error = xs
                        .mapIndexed { i, x -> Errors.compute(ys[i], layer.feedForward(x)) }
                        .sum() / xs.size
                println("$i -> error: $error")
            }
        }
        println("${System.currentTimeMillis() - start}ms")

        var errorSum = 0.0
        xs.forEachIndexed { i, x ->
            println("$x -> " + layer.feedForward(x))
            val error = Errors.compute(ys[i], layer.feedForward(x))
            errorSum += error
        }
        val error = errorSum / xs.size
        println("total error: $error")
        Assert.assertTrue(error < 0.09)
    }

}
package com.kennycason.nn

fun main(a: Array<String>) {
    val xs = arrayOf(
            doubleArrayOf(1.0, 1.0, 1.0),
            doubleArrayOf(1.0, 1.0, 0.0),
            doubleArrayOf(1.0, 1.0, 0.0),
            doubleArrayOf(1.0, 0.0, 0.0),
            doubleArrayOf(1.0, 0.0, 0.0),
            doubleArrayOf(1.0, 0.0, 0.0)
    )

    val hebbian = Hebbian(3)
    hebbian.learn(xs)
    println("learned weights: " + hebbian.weights.joinToString(", "))

    println(hebbian.evaluate(doubleArrayOf(1.0, 1.0, 1.0)))
    println(hebbian.evaluate(doubleArrayOf(1.0, 1.0, 0.0)))
    println(hebbian.evaluate(doubleArrayOf(1.0, 0.0, 1.0)))
    println(hebbian.evaluate(doubleArrayOf(0.0, 0.0, 1.0)))
}

class Hebbian(size: Int,
              private val l: Double = 0.1) {

    val weights = DoubleArray(size, { 1.0 } )

    fun learn(xs: Array<DoubleArray>) {
        for (x in xs) {
            val output = evaluate(x)
            // update weights
            (0 until weights.size).forEach { i ->
                weights[i] += output * l * x[i]
            }
        }
    }

    fun evaluate(x: DoubleArray) = weights
            .mapIndexed { i, w -> w * x[i] }
            .sum()
}

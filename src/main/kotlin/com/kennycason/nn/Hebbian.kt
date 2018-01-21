package com.kennycason.nn

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

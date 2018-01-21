package com.kennycason.nn

class Hebbian(size: Int,
              private val l: Float = 0.1f) {

    val weights = FloatArray(size, { 1.0f } )

    fun learn(xs: Array<FloatArray>) {
        for (x in xs) {
            val output = evaluate(x)
            // update weights
            (0 until weights.size).forEach { i ->
                weights[i] += output * l * x[i]
            }
        }
    }

    fun evaluate(x: FloatArray) = weights
            .mapIndexed { i, w -> w * x[i] }
            .sum()
}

package com.kennycason.nn

import org.junit.Test


class HebbianTest {

    @Test fun basic() {
        val xs = arrayOf(
                floatArrayOf(1.0f, 1.0f, 1.0f),
                floatArrayOf(1.0f, 1.0f, 0.0f),
                floatArrayOf(1.0f, 1.0f, 0.0f),
                floatArrayOf(1.0f, 0.0f, 0.0f),
                floatArrayOf(1.0f, 0.0f, 0.0f),
                floatArrayOf(1.0f, 0.0f, 0.0f)
        )

        val hebbian = Hebbian(3)
        hebbian.learn(xs)
        println("learned weights: " + hebbian.weights.joinToString(", "))

        println(hebbian.evaluate(floatArrayOf(1.0f, 1.0f, 1.0f)))
        println(hebbian.evaluate(floatArrayOf(1.0f, 1.0f, 0.0f)))
        println(hebbian.evaluate(floatArrayOf(1.0f, 0.0f, 1.0f)))
        println(hebbian.evaluate(floatArrayOf(0.0f, 0.0f, 1.0f)))
    }

}
package com.kennycason.nn

import org.junit.Test


class HebbianTest {

    @Test fun basic() {
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

}
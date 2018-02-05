package com.kennycason.nn

import org.junit.Assert
import org.junit.Test


class HebbianTest {

    @Test
    fun basic() {
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

        Assert.assertEquals(5.66f, hebbian.evaluate(floatArrayOf(1.0f, 1.0f, 1.0f)), 0.01f)
        Assert.assertEquals(4.36f, hebbian.evaluate(floatArrayOf(1.0f, 1.0f, 0.0f)), 0.01f)
        Assert.assertEquals(3.79f, hebbian.evaluate(floatArrayOf(1.0f, 0.0f, 1.0f)), 0.01f)
        Assert.assertEquals(1.3f, hebbian.evaluate(floatArrayOf(0.0f, 0.0f, 1.0f)), 0.01f)

    }

}
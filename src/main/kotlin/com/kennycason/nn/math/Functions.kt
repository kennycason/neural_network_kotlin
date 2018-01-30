package com.kennycason.nn.math

interface ActivationFunction {
    fun f(x: Float) : Float     // f(x)
    fun df(x: Float) : Float    // f'(x) derivative
}

object Functions {

    object Sigmoid : ActivationFunction {
        override fun f(x: Float) = (1.0f / (1.0f + Math.exp(-x.toDouble()).toFloat()))
        // sigmoid is factored out as it's applied default during propagation to output features
        // i.e. output feature = activation_function(feed_forward(x)
        override fun df(x: Float) = x * (1.0f - x)
    }

    object RelU : ActivationFunction {
        override fun f(x: Float) = Math.max(0f, x)
        override fun df(x: Float) = if (x <= 0f) { 0f } else { 1f }
    }

    object LeakyRelU : ActivationFunction {
        override fun f(x: Float) = Math.max(0.01f, x)
        override fun df(x: Float) = if (x <= 0f) { 0.01f } else { 1f }
    }

    object Tanh : ActivationFunction {
        override fun f(x: Float) = Math.tanh(x.toDouble()).toFloat()
        override fun df(x: Float) = 1 - (x * x)
    }

}
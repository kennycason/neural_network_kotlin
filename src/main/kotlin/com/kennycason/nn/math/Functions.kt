package com.kennycason.nn.math


object Functions {

    val sigmoid = fun (x: Float) = (1.0f / (1.0f + Math.exp(-x.toDouble()).toFloat()))

    // sigmoid is factored out as it's applied default during propagation to output features
    // i.e. output feature = activation_function(feed_forward(x)
    val sigmoidDerivative = fun (x: Float) = x * (1.0f - x)

}
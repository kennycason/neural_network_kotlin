package com.kennycason.nn.math


object Functions {

    val sigmoid = fun (x: Double) = 1.0 / (1 + Math.exp(-x))

    // sigmoid is factored out as it's applied default during propagation to output features
    // i.e. output feature = activation_function(feed_forward(x)
    val sigmoidDerivative = fun (x: Double) = x * (1.0 - x)

}
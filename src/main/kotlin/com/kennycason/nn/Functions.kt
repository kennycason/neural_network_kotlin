package com.kennycason.nn


object Functions {

    val sigmoid = fun (x: Double) = 1.0 / (1 + Math.exp(-x))

    val sigmoidDerivative = fun (x: Double) = x * (1.0 - x)
//        val y = sigmoid(x)
//        return y * (1 - y)

}
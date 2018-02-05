package com.kennycason.nn.util

fun <O : Any> time(lambda: () -> O): O {
    val start = System.currentTimeMillis()
    val out = lambda.invoke()
    println("${System.currentTimeMillis() - start}ms")
    return out
}
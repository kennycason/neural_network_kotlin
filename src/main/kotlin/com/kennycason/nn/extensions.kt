package com.kennycason.nn

import org.jblas.FloatMatrix

inline fun <reified INNER> array2d(sizeOuter: Int, sizeInner: Int,
                                   noinline innerInit: (Int)->INNER): Array<Array<INNER>> = Array(sizeOuter) { Array<INNER>(sizeInner, innerInit) }

fun FloatMatrix.apply(fn : (x: Float) -> Float) = applyi(fn, this, FloatMatrix(rows, columns))

fun FloatMatrix.applyi(fn : (x: Float) -> Float) : FloatMatrix {
    return applyi(fn, this, FloatMatrix(rows, columns))
}

fun FloatMatrix.applyi(fn : (x: Float) -> Float, other: FloatMatrix, result: FloatMatrix) : FloatMatrix {
    other.data.forEachIndexed { i, d -> result.data[i] = fn(d) }
    return result
}
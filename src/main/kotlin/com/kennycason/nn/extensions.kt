package com.kennycason.nn

import org.jblas.DoubleMatrix

inline fun <reified INNER> array2d(sizeOuter: Int, sizeInner: Int,
                                   noinline innerInit: (Int)->INNER): Array<Array<INNER>> = Array(sizeOuter) { Array<INNER>(sizeInner, innerInit) }

fun DoubleMatrix.apply(fn : (x: Double) -> Double) = applyi(fn, this, DoubleMatrix(rows, columns))

fun DoubleMatrix.applyi(fn : (x: Double) -> Double) : DoubleMatrix {
    return applyi(fn, this, DoubleMatrix(rows, columns))
}

fun DoubleMatrix.applyi(fn : (x: Double) -> Double, other: DoubleMatrix, result: DoubleMatrix) : DoubleMatrix {
    other.data.forEachIndexed { i, d -> result.data[i] = fn(d) }
    return result
}
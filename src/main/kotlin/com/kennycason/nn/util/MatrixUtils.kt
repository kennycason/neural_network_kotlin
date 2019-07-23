package com.kennycason.nn.util

import org.jblas.FloatMatrix

object MatrixUtils {
    fun copy(m: FloatMatrix) = m.add(0f)

    fun copyTo(from: FloatMatrix, to: FloatMatrix) {
        (0 until from.length).forEach { i -> to.data[i] = from.data[i] }
    }
}
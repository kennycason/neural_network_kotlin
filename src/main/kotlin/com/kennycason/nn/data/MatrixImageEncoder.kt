package com.kennycason.nn.data

import org.jblas.DoubleMatrix

interface MatrixImageEncoder {
    fun encode(image: Image): DoubleMatrix
}
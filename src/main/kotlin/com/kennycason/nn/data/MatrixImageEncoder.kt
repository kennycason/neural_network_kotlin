package com.kennycason.nn.data

import org.jblas.FloatMatrix

interface MatrixImageEncoder {
    fun encode(image: Image): FloatMatrix
}
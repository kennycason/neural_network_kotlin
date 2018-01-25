package com.kennycason.nn.data.image

import org.jblas.FloatMatrix

interface MatrixImageEncoder {
    fun encode(image: Image): FloatMatrix
}
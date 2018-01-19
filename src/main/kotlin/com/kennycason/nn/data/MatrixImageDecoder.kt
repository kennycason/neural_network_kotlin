package com.kennycason.nn.data

import org.jblas.DoubleMatrix


interface MatrixImageDecoder {
    fun decode(data: DoubleMatrix): Image
}
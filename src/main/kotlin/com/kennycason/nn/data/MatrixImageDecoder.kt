package com.kennycason.nn.data

import org.jblas.DoubleMatrix


abstract class MatrixImageDecoder(val rows: Int) {
    abstract fun decode(data: DoubleMatrix): Image
}
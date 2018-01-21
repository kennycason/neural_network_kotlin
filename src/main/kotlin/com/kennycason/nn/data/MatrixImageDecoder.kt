package com.kennycason.nn.data

import org.jblas.FloatMatrix


abstract class MatrixImageDecoder(val rows: Int) {
    abstract fun decode(data: FloatMatrix): Image
}
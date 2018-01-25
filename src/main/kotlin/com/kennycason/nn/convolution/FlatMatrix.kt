package com.kennycason.nn.convolution

import org.jblas.FloatMatrix


/**
 * A 2d matrix represented as a 1d matrix.
 *
 * Often times data may be 2d, but when it gets input into the nn it
 * is a 1d matrix. Knowing the rows/cols of the matrix helps us interpret
 * the 1d matrix in 2d.
 */
data class FlatMatrix(val m: FloatMatrix,
                      val rows: Int,
                      val cols: Int)
package com.kennycason.nn.data

import org.jblas.FloatMatrix


data class LabeledData(val xs: FloatMatrix,
                       val ys: Array<String>)
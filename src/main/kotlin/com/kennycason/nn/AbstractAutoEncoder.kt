package com.kennycason.nn

import org.jblas.FloatMatrix


abstract class AbstractAutoEncoder {

    abstract fun learn(xs: List<FloatMatrix>, steps: Int)

    abstract fun learn(x: FloatMatrix, steps: Int)

    abstract fun encode(x: FloatMatrix): FloatMatrix

    abstract fun decode(y: FloatMatrix): FloatMatrix

    abstract fun feedForward(x: FloatMatrix): FloatMatrix

}

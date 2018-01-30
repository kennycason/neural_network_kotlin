package com.kennycason.nn.math

import org.jblas.FloatMatrix


object Errors {
    fun compute(e: FloatMatrix, a: FloatMatrix): Double {
        val errors = e.sub(a)
        return Math.sqrt(errors.mul(errors).sum().toDouble())
    }
}
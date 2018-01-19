package com.kennycason.nn

import org.jblas.DoubleMatrix


object Errors {
    fun compute(e: DoubleMatrix, a: DoubleMatrix): Double {
        val errors = e.sub(a)
        return Math.sqrt(errors.mul(e.sub(a)).sum())
    }
}
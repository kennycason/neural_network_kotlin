package com.kennycason.nn

import com.kennycason.nn.util.MatrixUtils
import org.jblas.FloatMatrix
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Test

class MatrixUtilsTest {

    @Test
    fun copy() {
        val m = FloatMatrix(2, 2)
        (0 until 4).forEach { i -> m.data[i] = i.toFloat() }
        val m2 = MatrixUtils.copy(m)

        assertFalse(m === m2)
        (0 until 4).forEach { i -> assertEquals(m.data[i], m2.data[i])  }
    }

    @Test
    fun copyTo() {
        val from = FloatMatrix(2, 2)
        val to = FloatMatrix(2, 2)
        (0 until 4).forEach { i -> from.data[i] = i.toFloat() }
        MatrixUtils.copyTo(from, to)

        assertFalse(from === to)
        (0 until 4).forEach { i -> assertEquals(from.data[i], to.data[i])  }
    }
}
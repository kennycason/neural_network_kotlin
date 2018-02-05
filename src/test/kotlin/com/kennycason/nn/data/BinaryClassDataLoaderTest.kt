package com.kennycason.nn.data

import org.junit.Assert
import org.junit.Test


class BinaryClassDataLoaderTest {

    @Test
    fun load() {
        val labeledData = BinaryClassDataLoader.load("/data/cod_rna_short.svm")
        println(labeledData)

        Assert.assertEquals("-1", labeledData.ys[0])
        Assert.assertEquals(-766.0f, labeledData.xs.get(0, 0))
        Assert.assertEquals(128.0f, labeledData.xs.get(0, 1))
        Assert.assertEquals(0.140625f, labeledData.xs.get(0, 2))
    }
}
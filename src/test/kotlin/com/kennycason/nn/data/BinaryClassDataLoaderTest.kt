package com.kennycason.nn.data

import org.junit.Test


class BinaryClassDataLoaderTest {

    @Test
    fun load() {
        val labeledData = BinaryClassDataLoader.load("/data/cod_rna_short.svm")
        println(labeledData)
    }
}
package com.kennycason.nn.data

import org.junit.Test


class MNISTImageLoaderTest {

    @Test
    fun loadIdx3() {
        val mnistImageLoader = MNISTImageLoader()
        val totalDataSet = mnistImageLoader.loadIdx3("/data/mnist/train-images-idx3-ubyte").div(255.0)

        println("loaded ${totalDataSet.rows} samples")
        (0 until 50).forEach { i ->
            println(PrintUtils.toPixelBox(totalDataSet.getRow(i).toArray(), 28, 0.05))
        }

    }
}
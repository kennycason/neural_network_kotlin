package com.kennycason.nn.data

import org.junit.Test


class CompositeImageWriterTest {

    @Test
    fun mnist() {
        val totalDataSet = MNISTImageLoader.loadIdx3("/data/mnist/train-images-idx3-ubyte").div(255.0)
        val grayScaleImageDecoder = MatrixGrayScaleImageDecoder(rows = 28)

        // 60,000 images ~ 245x245

        val image = CompositeImageWriter.write(
                data = totalDataSet,
                matrixImageDecoder = grayScaleImageDecoder,
                rows = 245,
                cols = 245)

        image.save("/tmp/mnist_composite.png")
    }

}
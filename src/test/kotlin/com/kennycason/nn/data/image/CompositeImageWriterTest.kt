package com.kennycason.nn.data.image

import com.kennycason.nn.data.image.helper.ImageHelper
import org.junit.Assert
import org.junit.Test


class CompositeImageWriterTest {

    // TODO replace with smaller example
    @Test
    fun mnist() {
        val totalDataSet = MNISTDataLoader.loadIdx3("/data/mnist/train-images-idx3-ubyte")
        val grayScaleImageDecoder = MatrixGrayScaleImageDecoder(rows = 28)

        // 60,000 images ~ 245x245
        val image = CompositeImageWriter.write(
                data = totalDataSet,
                matrixImageDecoder = grayScaleImageDecoder,
                rows = 245,
                cols = 245)

        image.save("/tmp/mnist_composite.png")

        val testImage = Image("/data/image/mnist_composite.png")
        val savedImage = Image("/tmp/mnist_composite.png")
        Assert.assertTrue(ImageHelper.equals(testImage, savedImage))
    }

}
package com.kennycason.nn.data.image

import com.kennycason.nn.data.image.helper.ImageHelper
import org.junit.Assert
import org.junit.Test

class MatrixGrayScaleImageEncodeDecodeTest {

    @Test
    fun encodeDecode() {
        val imageData = MatrixGrayScaleImageEncoder().encode(Image("/data/image/ninja.png"))
        val image = MatrixGrayScaleImageDecoder(rows = 32).decode(imageData)

        image.save("/tmp/ninja_grayscale.png")

        val testImage = Image("/data/image/ninja_grayscale.png")
        val savedImage = Image("/tmp/ninja_grayscale.png")
        Assert.assertTrue(ImageHelper.equals(testImage, savedImage))
    }

}
package com.kennycason.nn.data.image

import com.kennycason.nn.data.image.helper.ImageHelper
import org.junit.Assert
import org.junit.Test


class MatrixRGBImageEncodeDecodeTest {

    @Test
    fun encodeDecode() {
        val originalImage = Image("/data/image/ninja.png")
        val imageData = MatrixRGBImageEncoder().encode(originalImage)
        val image = MatrixRGBImageDecoder(rows = 32).decode(imageData)
        image.save("/tmp/ninja.png")

        val savedImage = Image("/tmp/ninja.png")
        Assert.assertTrue(ImageHelper.equals(originalImage, savedImage))
    }

}
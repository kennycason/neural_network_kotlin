package com.kennycason.nn.data.image

import com.kennycason.nn.data.image.Image
import com.kennycason.nn.data.image.MatrixGrayScaleImageDecoder
import com.kennycason.nn.data.image.MatrixGrayScaleImageEncoder
import org.junit.Test

class MatrixGrayScaleImageEncodeDecodeTest {

    @Test
    fun encodeDecode() {
        val imageData = MatrixGrayScaleImageEncoder().encode(Image("/data/ninja.png"))
        val image = MatrixGrayScaleImageDecoder(rows = 32).decode(imageData)
        image.save("/tmp/ninja_grayscale.png")
    }

}
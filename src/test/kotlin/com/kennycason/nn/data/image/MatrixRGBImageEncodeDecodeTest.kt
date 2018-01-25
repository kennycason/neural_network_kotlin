package com.kennycason.nn.data.image

import com.kennycason.nn.data.image.Image
import com.kennycason.nn.data.image.MatrixRGBImageDecoder
import com.kennycason.nn.data.image.MatrixRGBImageEncoder
import org.junit.Test


class MatrixRGBImageEncodeDecodeTest {

    @Test
    fun encodeDecode() {
        val imageData = MatrixRGBImageEncoder().encode(Image("/data/ninja.png"))
        val image = MatrixRGBImageDecoder(rows = 32).decode(imageData)
        image.save("/tmp/ninja.png")
    }

}
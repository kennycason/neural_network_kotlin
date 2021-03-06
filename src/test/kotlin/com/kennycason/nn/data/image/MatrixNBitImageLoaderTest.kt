package com.kennycason.nn.data.image

import org.junit.Test


class MatrixNBitImageLoaderTest {

    @Test
    fun encodeDecode3BitTest() {
        val image = Image("/data/image/fighter_jet_small.jpg")
        val imageData = MatrixNBitImageEncoder(bits = 3).encode(image)
        val decoded = MatrixNBitImageDecoder(bits = 3, rows = image.height()).decode(imageData)
        decoded.save("/tmp/decoded_image_test.png")
    }


}
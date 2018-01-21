package com.kennycason.nn.data

import org.jblas.FloatMatrix
import java.awt.image.BufferedImage

class MatrixGrayScaleImageEncoder : MatrixImageEncoder {

    override fun encode(image: Image): FloatMatrix {
        val data = FloatMatrix(1, image.width() * image.height())
        val bi = image.data()
        var i = 0
        for (y in 0 until image.height()) {
            for (x in 0 until image.width()) {

                val rgb = bi.getRGB(x, y) and 0xFFFFFF

                val r = rgb and 0xFF0000 shr 16
                val g = rgb and 0xFF00 shr 8
                val b = rgb and 0xFF

                val grayScale = ((r + g + b) / 3) / 255.0f
                data.put(i++, grayScale)
            }
        }
        return data
    }

}
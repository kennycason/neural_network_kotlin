package com.kennycason.nn.data

import org.jblas.FloatMatrix
import java.awt.image.BufferedImage

class MatrixRGBImageEncoder(private val ignoreRGB: Int = -1) : MatrixImageEncoder {

    override fun encode(image: Image): FloatMatrix {
        val data = FloatMatrix(1, image.width() * 3 * image.height())
        val bi = image.data()
        for (y in 0 until image.height()) {
            for (x in 0 until image.width()) {
                read(data, bi, x, y)
            }
        }
        return data
    }

    private fun read(data: FloatMatrix, bi: BufferedImage, x: Int, y: Int) {
        val rgb = bi.getRGB(x, y) and 0xFFFFFF
        if (rgb == ignoreRGB) { return }

        val r = rgb and 0xFF0000 shr 16
        val g = rgb and 0xFF00 shr 8
        val b = rgb and 0xFF

        val index = y * bi.width * 3 + x * 3
        data.put(index, r / 255.0f)
        data.put(index + 1, g / 255.0f)
        data.put(index + 2, b / 255.0f)
    }

}
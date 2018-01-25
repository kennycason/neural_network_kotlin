package com.kennycason.nn.data.image

import org.jblas.FloatMatrix
import java.awt.image.BufferedImage

class MatrixRGBImageDecoder(rows: Int) : MatrixImageDecoder(rows) {

    override fun decode(data: FloatMatrix): Image {
        val cols = data.columns / 3 / rows

        val bi = BufferedImage(cols, rows, BufferedImage.TYPE_INT_RGB)
        var i = 0
        for (col in 0 until cols) {
            for (row in 0 until rows) {
                val r = (data.get(0, i) * 255.0f).toInt()
                val g = (data.get(0, i + 1) * 255.0f).toInt()
                val b = (data.get(0, i + 2) * 255.0f).toInt()
                val rgb = (r shl 16) + (g shl 8) + b
                bi.setRGB(col, row, rgb)
                i += 3
            }
        }
        return Image(bi)
    }


}
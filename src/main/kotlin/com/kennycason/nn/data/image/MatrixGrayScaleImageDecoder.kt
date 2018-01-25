package com.kennycason.nn.data.image

import org.jblas.FloatMatrix
import java.awt.image.BufferedImage

class MatrixGrayScaleImageDecoder(private val threshold: Double = 0.5,
                                  rows: Int) : MatrixImageDecoder(rows) {

    override fun decode(data: FloatMatrix): Image {
        val cols = data.columns / rows
        val bi = BufferedImage(cols, rows, BufferedImage.TYPE_INT_RGB)
        var i = 0
        for (col in 0 until cols) {
            for (row in 0 until rows) {
                val data = (data.get(i++) * 255.0).toInt()
                val grayScale = data + (data shl 8) + (data shl 16)
                bi.setRGB(col, row, grayScale)
            }
        }
        return Image(bi)
    }

}
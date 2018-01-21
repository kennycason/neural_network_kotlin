package com.kennycason.nn.data

import org.jblas.DoubleMatrix
import java.awt.image.BufferedImage

class MatrixGrayScaleImageDecoder(private val threshold: Double = 0.5,
                                  rows: Int) : MatrixImageDecoder(rows) {

    override fun decode(data: DoubleMatrix): Image {
        val cols = data.columns / rows
        val bi = BufferedImage(cols, rows, BufferedImage.TYPE_INT_RGB)
        for (x in 0 until rows * cols) {
            val data_x = (data.get(x) * 255.0).toInt()
            bi.setRGB(x % cols, x / cols, data_x + (data_x shl 8) + (data_x shl 16))
        }
        return Image(bi)
    }

}
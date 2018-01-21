package com.kennycason.nn.data

import org.jblas.DoubleMatrix
import java.awt.image.BufferedImage

class MatrixRGBImageDecoder(rows: Int) : MatrixImageDecoder(rows) {

    override fun decode(data: DoubleMatrix): Image {
        val cols = data.columns / 3 / rows
        val bi = BufferedImage(cols, rows, BufferedImage.TYPE_INT_RGB)
        var y = 0
        for (x in 0 until cols * rows) {
            read(data, bi, x, y)
            if (x > 0 && x % cols == 0) {
                y++
            }
        }
        return Image(bi)
    }

    private fun read(data: DoubleMatrix, bi: BufferedImage, x: Int, y: Int) {
        val cols = data.columns / 3 / rows
        val offset = x * 3
        val r = (data.get(0, offset) * 255.0).toInt()
        val g = (data.get(0, offset + 1) * 255.0).toInt()
        val b = (data.get(0, offset + 2) * 255.0).toInt()
        val rgb = (r shl 16) + (g shl 8) + b
        bi.setRGB(x % cols, y, rgb)

    }


}
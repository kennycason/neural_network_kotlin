package com.kennycason.nn.data

import org.jblas.DoubleMatrix
import java.awt.image.BufferedImage

class MatrixGrayScaleImageEncoder : MatrixImageEncoder {
    private val BITS = 8

    private val HIGH_BIT_FLAG = 8421504

    override fun encode(image: Image): DoubleMatrix {
        val data = DoubleMatrix(1, image.width() * BITS * image.height())
        val bi = image.data()
        for (y in 0 until image.height()) {
            for (x in 0 until image.width()) {
                read(data, bi, x, y)
            }
        }
        return data
    }

    private fun read(data: DoubleMatrix, bi: BufferedImage, x: Int, y: Int) {
        var flag = HIGH_BIT_FLAG
        var offset = 0
        while (offset < 8) {
            val rgb = bi.getRGB(x, y) and 0xFFFFFF
            val set = rgb and flag == flag
            val index = y * bi.width * BITS + (x * BITS + offset)
            data.data[index] = if (set) 1.0 else 0.0
            offset++
            flag = flag shr 1
        }
    }

}
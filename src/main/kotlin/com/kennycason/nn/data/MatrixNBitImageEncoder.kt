package com.kennycason.nn.data

import org.jblas.FloatMatrix
import java.awt.image.BufferedImage


class MatrixNBitImageEncoder(private val bits: Int = 24) : MatrixImageEncoder {
    private val RGB_BITS = 24

    private val HIGH_BIT_FLAG = 8388608

    override fun encode(image: Image): FloatMatrix {
        val data = FloatMatrix(1, image.width() * bits * image.height())
        val bi = image.data()
        for (y in 0 until image.height()) {
            for (x in 0 until image.width()) {
                read(data, bi, x, y)
            }
        }
        return data
    }

    private fun read(data: FloatMatrix, bi: BufferedImage, x: Int, y: Int) {
        var flag = HIGH_BIT_FLAG
        var offset = 0
        while (flag > 0) {
            val rgb = bi.getRGB(x, y) and 0xFFFFFF
            val set = rgb and flag == flag
            val index = y * bi.width * bits + (x * bits + offset)
            data.data[index] = if (set) 1.0f else 0.0f
            offset++
            flag = flag shr RGB_BITS / bits
        }
    }

}
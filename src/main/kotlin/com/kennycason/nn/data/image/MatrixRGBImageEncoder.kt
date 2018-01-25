package com.kennycason.nn.data.image

import org.jblas.FloatMatrix

class MatrixRGBImageEncoder : MatrixImageEncoder {

    override fun encode(image: Image): FloatMatrix {
        val data = FloatMatrix(1, image.width() * 3 * image.height())
        val bi = image.data()
        var i = 0
        for (col in 0 until image.width()) {
            for (row in 0 until image.height()) {
                val rgb = bi.getRGB(col, row) and 0xFFFFFF

                val r = rgb and 0xFF0000 shr 16
                val g = rgb and 0xFF00 shr 8
                val b = rgb and 0xFF

                data.put(i, r / 255.0f)
                data.put(i + 1, g / 255.0f)
                data.put(i + 2, b / 255.0f)

                i += 3
            }
        }
        return data
    }


}
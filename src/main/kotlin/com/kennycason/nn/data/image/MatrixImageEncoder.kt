package com.kennycason.nn.data.image

import org.jblas.FloatMatrix
import java.awt.image.BufferedImage

interface MatrixImageEncoder {
    fun encode(image: Image): FloatMatrix
    fun encode(bi: BufferedImage): FloatMatrix
}
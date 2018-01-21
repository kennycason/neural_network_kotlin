package com.kennycason.nn.data

import org.jblas.DoubleMatrix
import java.awt.image.BufferedImage

/**
 * write multiple images to a single image
 * useful for displaying bulk of images together or presenting sequences to demonstrate learning
 */
object CompositeImageWriter {

    fun write(data: DoubleMatrix,
              matrixImageDecoder: MatrixImageDecoder,
              rows: Int,
              cols: Int): Image {

        val perImageWidth = data.columns / matrixImageDecoder.rows
        val perImageHeight = matrixImageDecoder.rows
        val compositeImageWidth = cols * perImageWidth
        val compositeImageHeight = rows * perImageHeight

        val bi = BufferedImage(compositeImageWidth, compositeImageHeight, BufferedImage.TYPE_INT_RGB)
        var i = 0

        (0 until cols).forEach { col ->
            (0 until rows).forEach { row ->
                if (i >= data.rows) { return Image(bi) }

                val image = matrixImageDecoder.decode(data.getRow(i++))
                (0 until perImageWidth).forEach { x ->
                    (0 until perImageHeight).forEach { y ->
                        bi.setRGB(
                                col * perImageWidth + x,
                                row * perImageHeight + y,
                                image.get(x, y))
                    }
                }

            }
        }

        return Image(bi)
    }

}
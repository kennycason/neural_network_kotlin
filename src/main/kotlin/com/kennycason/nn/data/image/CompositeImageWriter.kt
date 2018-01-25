package com.kennycason.nn.data.image

import org.jblas.FloatMatrix
import java.awt.image.BufferedImage

/**
 * write multiple images to a single image
 * useful for displaying bulk of images together or presenting sequences to demonstrate learning
 */
object CompositeImageWriter {

    fun write(data: List<FloatMatrix>,
              matrixImageDecoder: MatrixImageDecoder,
              rows: Int,
              cols: Int): Image {
        // calculate to know the exact dimensions, easier than asking user to provide to function
        val sample = matrixImageDecoder.decode(data[0])
        val perImageWidth = sample.width()
        val perImageHeight = sample.height()
        val compositeImageWidth = cols * perImageWidth
        val compositeImageHeight = rows * perImageHeight

        val bi = BufferedImage(compositeImageWidth, compositeImageHeight, BufferedImage.TYPE_INT_RGB)
        var i = 0

        (0 until cols).forEach { col ->
            (0 until rows).forEach { row ->
                if (i >= data.size) {
                    return Image(bi)
                }
                val image = matrixImageDecoder.decode(data.get(i++))
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
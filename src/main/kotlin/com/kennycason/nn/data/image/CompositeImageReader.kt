package com.kennycason.nn.data.image

import org.jblas.FloatMatrix
import java.awt.image.BufferedImage

/**
 * read multiple a tile sheet into individual image matrices
 */
object CompositeImageReader {

    fun read(file: String,
             matrixImageEncoder: MatrixImageEncoder,
             rows: Int,
             cols: Int,
             n: Int): List<FloatMatrix> {

        val image = Image(file)

        val imageWidth = image.width() / cols
        val imageHeight = image.height() / rows
        println("image properties, rows: $imageHeight x cols: $imageWidth")

        val images = mutableListOf<FloatMatrix>()
        (0 until rows).forEach { row ->
            (0 until cols).forEach { col ->
                if (images.size == n) { return images }

                val bi = BufferedImage(imageWidth, imageHeight, BufferedImage.TYPE_INT_RGB)
                (0 until imageWidth).forEach { x ->
                    (0 until imageHeight).forEach { y ->
                        bi.setRGB(
                                x, y,
                                image.get(col * imageWidth + x, row * imageHeight + y))
                    }
                }
                val encoded = matrixImageEncoder.encode(Image(bi))
                images.add(encoded)
            }
        }

        return images
    }

}
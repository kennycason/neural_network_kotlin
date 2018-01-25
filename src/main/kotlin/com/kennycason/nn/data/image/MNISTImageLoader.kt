package com.kennycason.nn.data.image

import org.apache.commons.io.IOUtils
import org.jblas.FloatMatrix
import java.nio.ByteBuffer


object MNISTImageLoader {
    /*
        [offset] [type]          [value]          [description]
        0000     32 bit integer  0x00000803(2051) magic number
        0004     32 bit integer  60000            number of images
        0008     32 bit integer  28               number of rows
        0012     32 bit integer  28               number of columns
        0016     unsigned byte   ??               pixel
        0017     unsigned byte   ??               pixel
        ........
        xxxx     unsigned byte   ??               pixel

        Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).
     */
    fun loadIdx3(file: String): List<FloatMatrix> {
        val byteBuffer = ByteBuffer.wrap(IOUtils.toByteArray(MNISTImageLoader::class.java.getResourceAsStream(file)))
        val magicNumber = byteBuffer.int // 2051
        val numberImages = byteBuffer.int
        val numberRows = byteBuffer.int
        val numberCols = byteBuffer.int

        println("rows: $numberRows x $numberCols")

        val data = mutableListOf<FloatMatrix>()
        for (i in 0 until numberImages) {
            val entry = FloatMatrix(1, numberRows * numberCols)
            data.add(readImage(byteBuffer, numberRows, numberCols).divi(255.0f))
        }
        return data
    }

    private fun readImage(byteBuffer: ByteBuffer, numberRows: Int, numberCols: Int): FloatMatrix {
        val data = FloatMatrix(1, numberRows * numberCols)
        for (i in 0 until numberCols) {
            for (j in 0 until numberRows) {
                data.put(j * numberCols + i, (byteBuffer.get().toInt() and 0xFF).toFloat())
            }
        }
        return data
    }

}
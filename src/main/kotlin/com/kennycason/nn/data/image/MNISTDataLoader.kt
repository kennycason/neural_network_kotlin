package com.kennycason.nn.data.image

import org.apache.commons.io.IOUtils
import org.jblas.FloatMatrix
import java.nio.ByteBuffer


object MNISTDataLoader {
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
        val byteBuffer = ByteBuffer.wrap(IOUtils.toByteArray(MNISTDataLoader::class.java.getResourceAsStream(file)))
        val magicNumber = byteBuffer.int // 2051
        val numberImages = byteBuffer.int
        val numberRows = byteBuffer.int
        val numberCols = byteBuffer.int

        println("rows: $numberRows x $numberCols")

        val data = mutableListOf<FloatMatrix>()
        for (i in 0 until numberImages) {
            data.add(readImage(byteBuffer, numberRows, numberCols).divi(255.0f))
        }
        return data
    }

    /*
        [offset] [type]          [value]          [description]
        0000     32 bit integer  0x00000801(2049) magic number (MSB first)
        0004     32 bit integer  10000            number of items
        0008     unsigned byte   ??               label
        0009     unsigned byte   ??               label
        ........
        xxxx     unsigned byte   ??               label

        The labels values are 0 to 9.
    */
    fun loadIdx1(file: String): List<Int> {
        val byteBuffer = ByteBuffer.wrap(IOUtils.toByteArray(MNISTDataLoader::class.java.getResourceAsStream(file)))
        val magicNumber = byteBuffer.int // 2049
        val numberItems = byteBuffer.int

        val data = mutableListOf<Int>()
        for (i in 0 until numberItems) {
            data.add(byteBuffer.get().toInt() and 0xFF)
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
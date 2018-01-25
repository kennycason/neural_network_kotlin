package com.kennycason.nn.data.image

import com.kennycason.nn.data.PrintUtils
import org.junit.Test


class MNISTImageLoaderTest {

    @Test
    fun loadIdx3() {
        val totalDataSet = MNISTImageLoader.loadIdx3("/data/mnist/train-images-idx3-ubyte")

        println("loaded ${totalDataSet.size} samples")
        (0 until 50).forEach { i ->
            println(PrintUtils.toPixelBox(totalDataSet.get(i).toArray(), 28, 0.05))
        }
    }

    @Test
    fun saveIdx3AsImage() {
        val totalDataSet = MNISTImageLoader.loadIdx3("/data/mnist/train-images-idx3-ubyte")
        val grayScaleImageDecoder = MatrixGrayScaleImageDecoder(rows = 28)

        println("loaded ${totalDataSet.size} samples")
        (0 until 10).forEach { i ->
            grayScaleImageDecoder.decode(totalDataSet.get(i)).save("/tmp/mnist_$i.png")
        }

        // re-read and re-write (sanity test)
        val grayScaleImageEncoder = MatrixGrayScaleImageEncoder()
        (0 until 10).forEach { i ->
            val data = grayScaleImageEncoder.encode(Image("/tmp/mnist_$i.png"))
            grayScaleImageDecoder.decode(data).save("/tmp/mnist_gray_scale_$i.png")
        }
    }

}
package com.kennycason.nn

import com.kennycason.nn.data.PrintUtils
import com.kennycason.nn.data.image.CompositeImageWriter
import com.kennycason.nn.data.image.MNISTImageLoader
import com.kennycason.nn.data.image.MatrixGrayScaleImageDecoder
import com.kennycason.nn.math.Errors
import org.jblas.FloatMatrix
import org.junit.Test

class MNISTDeepAutoEncoderTest {

    // 250k steps = 1h
    @Test
    fun mnistDataSet() {
        val xs = MNISTImageLoader
                .loadIdx3("/data/mnist/train-images-idx3-ubyte")


        val visibleSize = 28 * 28
        val layer = DeepAutoEncoder(
                learningRate = 0.1f,
                layerDimensions = arrayOf(
                        arrayOf(visibleSize, (visibleSize * 0.75).toInt()),                  // 784 -> 588
                        arrayOf((visibleSize * 0.75).toInt(), (visibleSize * 0.50).toInt()), // 588 -> 294
                        arrayOf((visibleSize * 0.50).toInt(), (visibleSize * 0.25).toInt())  // 294 -> 196
                        //  arrayOf((visibleSize * 0.25).toInt(), (visibleSize * 0.10).toInt())  // 196 > 78
                ),
                log = true)

        val start = System.currentTimeMillis()
        layer.learn(xs = xs, steps = 250_000)
        println("${System.currentTimeMillis() - start}ms")

        println("calculate error")
        val generatedOutputs = mutableListOf<FloatMatrix>()
        // calculate total error
        var errorSum = 0.0
        (0 until xs.size).forEach { i ->
            val x = xs.get(i)
            val y = layer.feedForward(x)
            val error = Errors.compute(x, y)
            generatedOutputs.add(y)

            if (i % 1000 == 0) {
                println("input:\n" + PrintUtils.toPixelBox(x.toArray(), 28, 0.5))
                println("output:\n" + PrintUtils.toPixelBox(y.toArray(), 28, 0.7))
                println("error: $error")
                println("$i/${xs.size}")
            }
            errorSum += error
        }
        println("total error: " + (errorSum / (xs.size)))

        // write to composite image
        val image = CompositeImageWriter.write(
                data = generatedOutputs,
                matrixImageDecoder = MatrixGrayScaleImageDecoder(rows = 28),
                rows = 245,
                cols = 245)

        image.save("/tmp/mnist_deep_autoencoder_generated.png")
    }

}
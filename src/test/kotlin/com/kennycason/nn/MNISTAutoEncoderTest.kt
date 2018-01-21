package com.kennycason.nn

import com.kennycason.nn.data.*
import com.kennycason.nn.math.Errors
import org.jblas.FloatMatrix
import org.junit.Test
import java.io.File
import java.util.*

class MNISTAutoEncoderTest {

    @Test
    fun mnistDataSet() {
        val xs = MNISTImageLoader.loadIdx3("/data/mnist/train-images-idx3-ubyte")

        val visibleSize = 28 * 28
        val hiddenSize = (visibleSize * 0.75).toInt()
        val layer = AutoEncoder(
                learningRate = 0.1f,
                visibleSize = visibleSize,
                hiddenSize = hiddenSize,
                log = false)

        val start = System.currentTimeMillis()
        val rand = Random()
        (0.. 1_000_000).forEach { i ->
            val x = xs.getRow(rand.nextInt(xs.rows))
            layer.learn(x, 1)

            val error = Errors.compute(x, layer.feedForward(x))
            if (i % 25 == 0) {
                println("$i -> error: $error")
            }

        }
        println("${System.currentTimeMillis() - start}ms")

        val generatedOutputs = FloatMatrix(xs.rows, xs.columns)

        // calculate error
        var errorSum = 0.0
        (0 until xs.rows).forEach{ i ->
            val x = xs.getRow(i)
            val y = layer.feedForward(x)
            val error = Errors.compute(x, y)
            generatedOutputs.putRow(i, y)

            if (i % 1000 == 0) {
                println("input:\n" + PrintUtils.toPixelBox(x.toArray(), 28, 0.5))
                println("output:\n" + PrintUtils.toPixelBox(y.toArray(), 28, 0.7))
                println("error: $error")
                println("$i/${xs.rows}")
            }
            errorSum += error
        }
        println("total error: " + (errorSum / (xs.rows)))

        // write to composite image
        val image = CompositeImageWriter.write(
                data = generatedOutputs,
                matrixImageDecoder = MatrixGrayScaleImageDecoder(rows = 28),
                rows = 245,
                cols = 245)

        image.save("/tmp/mnist_autoencoder_generated.png")

        val weightFile = File("/tmp/experiment/mnist_auto_encoder_weights.log")
        weightFile.writeText(
                "encode:\n" +
                        layer.encode.toString("%f", "[", "]", ", ", "\n") + "\n" +
                        "decode:\n" +
                        layer.decode.toString("%f", "[", "]", ", ", "\n") + "\n"
        )

    }

}
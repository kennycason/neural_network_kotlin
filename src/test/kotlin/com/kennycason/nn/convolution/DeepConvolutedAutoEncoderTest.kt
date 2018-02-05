package com.kennycason.nn.convolution

import com.kennycason.nn.data.image.*
import com.kennycason.nn.learning_rate.FixedLearningRate
import com.kennycason.nn.math.Errors
import com.kennycason.nn.util.time
import org.jblas.FloatMatrix
import org.junit.Assert
import org.junit.Ignore
import org.junit.Test


class DeepConvolutedAutoEncoderTest {

    @Test
    fun ninjaTurdle() {
        val image = Image("/data/image/ninja.png")
        val imageData = MatrixRGBImageEncoder().encode(image)
        // 32 x 24
        // encode layer 2304 x 1728,
        // decode layer 1728 x 2304

        val height = image.height()
        val width = image.width()

        val layer = ConvolutedAutoEncoderFactory.generateConvolutedAutoencoder(
                initialVisibleDim = Dim(rows = 32 * 3, cols = 24),
                layers = 3,
                learningRate = FixedLearningRate()
        )

        time({
            (0..250).forEach { i ->
                layer.learn(listOf(imageData), 10)

                val visual = layer.feedForward(imageData)
                val outImage = MatrixRGBImageDecoder(rows = height).decode(visual)
                outImage.save("/tmp/output_ninja_$i.png")
            }
        })

        val error = Errors.compute(imageData, layer.feedForward(imageData))

        println("error -> $error")
        Assert.assertTrue(error < 3.0) // typically around 1.5, so this may non-deterministically fail
    }

    @Test
    fun jet() {
        val image = Image("/data/image/fighter_jet_small.jpg")
        val imageData = MatrixRGBImageEncoder().encode(image)
        // 100 x 63
        // encode layer 300 x 63,

        val height = image.height()
        val width = image.width()

        val layer = ConvolutedAutoEncoderFactory.generateConvolutedAutoencoder(
                initialVisibleDim = Dim(rows = 100 * 3, cols = 63),
                layers = 3,
                learningRate = FixedLearningRate(),
                log = false
        )

        time({
            (0..500).forEach { i ->
                layer.learn(listOf(imageData), 10)

                val visual = layer.feedForward(imageData)
                val outImage = MatrixRGBImageDecoder(rows = height).decode(visual)

                if (i % 10 == 0) {
                    outImage.save("/tmp/output_jet_$i.png")
                }

                val error = Errors.compute(imageData, layer.feedForward(imageData))
                println("error -> $error")
            }
        })

        val error = Errors.compute(imageData, layer.feedForward(imageData))
        println("error -> $error")
        Assert.assertTrue(error < 3.0) // typically around 1.5
    }

    @Ignore
    fun pokemon() {
        val xs = CompositeImageReader.read(
                file = "/data/image/pokemon_151_dark_bg.png",
                matrixImageEncoder = MatrixRGBImageEncoder(),
                rows = 11,
                cols = 15,
                n = 151)
        // images are 60x60 (encode layer x3 rgb = 10800)

        val height = 60
        val width = 60

//        val layer = ConvolutedAutoEncoderFactory.generateConvolutedAutoencoder(
//                initialVisibleDim = Dim(rows = 60 * 3, cols = 60),
//                layers = 5,
//                learningRate = FixedLearningRate()
//        )

        val layer1 = ConvolutedAutoEncoder(
                visibleDim = Dim(60 * 3, 60),
                hiddenDim = Dim(360, 120),
                partitions = Dim(60, 60),
                learningRate = FixedLearningRate(),
                log = false
        )
        val layer2 = ConvolutedAutoEncoder(
                visibleDim = Dim(360, 120),
                hiddenDim = Dim(180, 120),
                partitions = Dim(30, 30),
                learningRate = FixedLearningRate(),
                log = false
        )
        val layer3 = ConvolutedAutoEncoder(
                visibleDim = Dim(180, 120),
                hiddenDim = Dim(120, 60),
                partitions = Dim(20, 20),
                learningRate = FixedLearningRate(),
                log = false
        )
        val layer4 = ConvolutedAutoEncoder(
                visibleDim = Dim(120, 60),
                hiddenDim = Dim(60, 30),
                partitions = Dim(10, 10),
                learningRate = FixedLearningRate(),
                log = false
        )
//        val layer5 = ConvolutedLayer(
//                visibleDim = Dim(80, 60),
//                hiddenDim = Dim(40, 40),
//                paritions = Dim(20, 20),
//                learningRate = FixedLearningRate(),
//                log = false
//        )
//        val layer6 = ConvolutedLayer(
//                visibleDim = Dim(40, 40),
//                hiddenDim = Dim(20, 20),
//                paritions = Dim(10, 10),
//                learningRate = FixedLearningRate(),
//                log = false
//        )
//        val layer7 = ConvolutedLayer(
//                visibleDim = Dim(20, 20),
//                hiddenDim = Dim(1, 1),
//                paritions = Dim(1, 1),
//                learningRate = FixedLearningRate(),
//                log = false
//        )

        val layer = DeepConvolutedAutoEncoder(
                layers = arrayOf(layer1, layer2, layer3, layer4/*, layer5, layer6*/))

        val start = System.currentTimeMillis()
        var j = 1
        val m = 0
        val n = 151
        (0.. 100_000).forEach { i ->
            layer.learn(xs.subList(m, n), n * 2)
            println("$i -> ${System.currentTimeMillis() - start}ms")

            if (i > j) {
                val ys = mutableListOf<FloatMatrix>()
                xs.subList(m, n).forEach { x ->
                    ys.add(layer.feedForward(x))
                }
                val image = CompositeImageWriter.write(
                        data = ys,
                        rows = 11,
                        cols = 15,
                        matrixImageDecoder = MatrixRGBImageDecoder(rows = 60))
                image.save("/tmp/deep_generated_pokemon_$i.png")
                j += 2
            }
        }
    }

}
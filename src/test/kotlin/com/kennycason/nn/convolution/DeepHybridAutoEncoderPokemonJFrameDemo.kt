package com.kennycason.nn.convolution

import com.kennycason.nn.AbstractAutoEncoder
import com.kennycason.nn.AutoEncoder
import com.kennycason.nn.DeepAutoEncoder
import com.kennycason.nn.data.image.*
import com.kennycason.nn.learning_rate.FixedLearningRate
import org.jblas.FloatMatrix
import org.junit.Test

import java.awt.Graphics
import java.util.*
import javax.swing.JFrame
import javax.swing.JPanel
import javax.swing.WindowConstants

/**
 * Created by kenny on 12/4/16
 *
 * Any live cell with fewer than two live neighbours dies, as if caused by under-population.
 * Any live cell with two or three live neighbours lives on to the next generation.
 * Any live cell with more than three live neighbours dies, as if by over-population.
 * Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.
 */


class DeepHybridConvolutedAutoEncoderPokemonJFrameDemo {

    @Test
    fun run() {
        val random = Random()
        val screenWidth = 920
        val screenHeight = 660
        val saveImage = false

        val xs = CompositeImageReader.read(
                file = "/data/image/pokemon_151_dark_bg.png",
                matrixImageEncoder = MatrixRGBImageEncoder(),
                rows = 11,
                cols = 15,
                n = 151)

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
        //    val layer3 = AutoEncoder(
//            visibleSize = 180 * 120,
//            hiddenSize = 60 * 60,
//            learningRate = FixedLearningRate(),
//            log = false
//    )
        val layer4 = ConvolutedAutoEncoder(
                visibleDim = Dim(120, 60),
                hiddenDim = Dim(60, 30),
                partitions = Dim(10, 10),
                learningRate = FixedLearningRate(),
                log = false
        )

        //    val layer5 = ConvolutedAutoEncoder(
//            visibleDim = Dim(120, 60),
//            hiddenDim = Dim(60, 30),
//            paritions = Dim(10, 10),
//            learningRate = FixedLearningRate(),
//            log = false
//    )
        val layer5 = AutoEncoder(
                visibleSize = 60 * 30,
                hiddenSize = 100,
                learningRate = FixedLearningRate(),
                log = false
        )


        val layer = DeepAutoEncoder(
                layers = arrayOf<AbstractAutoEncoder>(
                        layer1, layer2/*, layer3, layer4, layer5, layer6*/),
                log = true)

        var i = 0
        val m = 0
        val n = 151

        val frame = JFrame()
        frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE)
        frame.setSize(screenWidth, screenHeight)
        frame.setVisible(true)

        val panel = object: JPanel() {
            override fun paintComponent(g: Graphics) {
                super.paintComponent(g)

                val ys = mutableListOf<FloatMatrix>()
                xs.subList(m, n).forEach { x ->
                    ys.add(layer.feedForward(x))
                }
                val image = CompositeImageWriter.write(
                        data = ys,
                        rows = 11,
                        cols = 14,
                        matrixImageDecoder = MatrixRGBImageDecoder(rows = 60))

                g.drawImage(image.bi, 0, 0, screenWidth, screenHeight - 20, this)

                if (saveImage) {
                    image.save("/tmp/very_deep_generated_pokemon_$i.png")
                }


            }
        }
        frame.add(panel)
        panel.revalidate()

        while (true) {
            layer.learn(xs.subList(m, n), 150)
            panel.repaint()
            i++
        }
    }

}
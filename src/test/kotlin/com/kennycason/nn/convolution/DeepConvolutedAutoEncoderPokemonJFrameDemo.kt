package com.kennycason.nn.convolution

import com.kennycason.nn.data.image.*
import org.jblas.FloatMatrix
import org.junit.Test

import java.awt.Color
import java.awt.Graphics
import java.awt.image.BufferedImage
import java.io.File
import java.util.*
import java.util.concurrent.atomic.AtomicInteger
import javax.imageio.ImageIO
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

fun main(args: Array<String>) {
    ConwaysGameOfLife().run()
}

class ConwaysGameOfLife {
    val random = Random()
    val screenWidth = 920
    val screenHeight = 660
    val saveImage = false

    val xs = CompositeImageReader.read(
            file = "/data/pokemon_151_dark_bg.png",
            matrixImageEncoder = MatrixRGBImageEncoder(),
            rows = 11,
            cols = 15,
            n = 151)

    val layer1 = ConvolutedLayer(
            visibleDim = Dim(60 * 3, 60),
            hiddenDim = Dim(360, 120),
            paritions = Dim(60, 60),
            learningRate = 0.1f,
            log = false
    )
//    val layer2 = ConvolutedLayer(
//            visibleDim = Dim(360, 120),
//            hiddenDim = Dim(180, 120),
//            paritions = Dim(30, 30),
//            learningRate = 0.1f,
//            log = false
//    )
//    val layer3 = ConvolutedLayer(
//            visibleDim = Dim(180, 120),
//            hiddenDim = Dim(120, 60),
//            paritions = Dim(20, 20),
//            learningRate = 0.1f,
//            log = false
//    )
//    val layer4 = ConvolutedLayer(
//            visibleDim = Dim(120, 60),
//            hiddenDim = Dim(60, 30),
//            paritions = Dim(10, 10),
//            learningRate = 0.1f,
//            log = false
//    )

    val layer = DeepConvolutedAutoEncoder(
            layers = arrayOf(layer1/*, layer2, layer3, layer4, layer5, layer6*/))

    var i = 0
    val m = 0
    val n = 151
    fun run() {
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
import com.github.sarxos.webcam.Webcam
import com.kennycason.nn.AbstractAutoEncoder
import com.kennycason.nn.AutoEncoder
import com.kennycason.nn.DeepAutoEncoder
import com.kennycason.nn.convolution.ConvolutedAutoEncoder
import com.kennycason.nn.convolution.Dim
import com.kennycason.nn.data.image.MatrixGrayScaleImageDecoder
import com.kennycason.nn.data.image.MatrixGrayScaleImageEncoder
import com.kennycason.nn.data.image.MatrixRGBImageDecoder
import com.kennycason.nn.data.image.MatrixRGBImageEncoder
import com.kennycason.nn.learning_rate.FixedLearningRate
import java.awt.Color
import java.awt.Graphics
import java.awt.image.BufferedImage
import java.io.File
import java.util.*
import javax.imageio.ImageIO
import javax.swing.JFrame
import javax.swing.JPanel
import javax.swing.WindowConstants

fun main(args: Array<String>) {
    WebCamContinuousLearnDemo().run()
}

class WebCamContinuousLearnDemo {
    val screenWidth = 176 * 3 * 2
    val screenHeight = 144 * 3 * 2
    val saveImage = false

    fun run() {

        val frame = JFrame()
        frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE)
        frame.setSize(screenWidth, screenHeight)
        frame.setVisible(true)

        // 176 x 144
        val webcam = Webcam.getDefault()
        webcam.open()

        // black/white
        val layer1 = ConvolutedAutoEncoder(
                visibleDim = Dim(176, 144),
                hiddenDim = Dim(88, 72),
                partitions = Dim(8, 8),
                learningRate = FixedLearningRate(),
                log = false
        )
        val layer2 =  ConvolutedAutoEncoder(
                visibleDim = Dim(88, 72),
                hiddenDim = Dim(22, 36),
                partitions = Dim(11, 6),
                learningRate = FixedLearningRate(),
                log = false
        )
        val layer3 = AutoEncoder(
                visibleSize = 22 * 36,
                hiddenSize = 250,
                learningRate = FixedLearningRate(),
                log = false
        )

        val autoEncoder = DeepAutoEncoder(
                layers = arrayOf<AbstractAutoEncoder>(layer1, layer2, layer3),
                log = true)

        val encoder = MatrixGrayScaleImageEncoder()
        val decoder = MatrixGrayScaleImageDecoder(rows = 144)

        // Color network
//        val layer1 = ConvolutedAutoEncoder(
//                visibleDim = Dim(176 * 3, 144),
//                hiddenDim = Dim(176, 72),
//                partitions = Dim(16, 8),
//                learningRate = FixedLearningRate(),
//                log = false
//        )
//        val layer2 = ConvolutedAutoEncoder(
//                visibleDim = Dim(176, 72),
//                hiddenDim = Dim(88, 36),
//                partitions = Dim(8, 6),
//                learningRate = FixedLearningRate(),
//                log = false
//        )
//        val layer3 =  ConvolutedAutoEncoder(
//                visibleDim = Dim(88, 36),
//                hiddenDim = Dim(22, 12),
//                partitions = Dim(11, 6),
//                learningRate = FixedLearningRate(),
//                log = false
//        )
//        val layer4 = AutoEncoder(
//                visibleSize = 22 * 12,
//                hiddenSize = 250,
//                learningRate = FixedLearningRate(),
//                log = false
//        )
//
//        val autoEncoder = DeepAutoEncoder(
//                layers = arrayOf<AbstractAutoEncoder>(layer1, layer2, layer3, layer4),
//                log = true)
//        val encoder = MatrixRGBImageEncoder()
//        val decoder = MatrixRGBImageDecoder(rows = 144)

        println("dim ${webcam.image.width}x${webcam.image.height}")
        
        var i = 0
        val panel = object: JPanel() {
            override fun paintComponent(g: Graphics) {
                super.paintComponent(g)

                val x = encoder.encode(webcam.image)
                println(x.length)
                // learn one iteration
                autoEncoder.learn(listOf(x), 1)
                val y = autoEncoder.feedForward(x)
                val image = decoder.decode(y)

                g.drawImage(image.bi, 0, 0, screenWidth, screenHeight, this)

                if (saveImage && i % 1000 == 0) {
                    ImageIO.write(image.bi, "PNG", File("/tmp/webcam-learn-$i.png"))
                }
            }
        }
        frame.add(panel)
        panel.revalidate()

        while (true) {
            panel.repaint()
        }
    }



}

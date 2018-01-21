package com.kennycason.nn

import com.kennycason.nn.data.*
import com.kennycason.nn.math.Errors
import org.jblas.FloatMatrix
import org.junit.Test

class VoterClusteringAutoEncoderTest {

    @Test
    fun voterClustering() {
        val labeledData = VoterDataLoader.load()
        val xs = labeledData.xs
        val ys = labeledData.ys

        val layer = DeepAutoEncoder(
                learningRate = 0.05f,
                layerDimensions = arrayOf(
                        arrayOf(16, 32),
                        arrayOf(32, 20),
                        arrayOf(20, 2)
                ),
                log = true)

        layer.learn(xs.rowsAsList(), 1_000_000)

//        (0 until xs.rows).forEach { i ->
//            println("i: ${xs.getRow(i)}\no: ${layer.feedForward(xs.getRow(i))}")
//        }

        (0 until xs.rows).forEach { i ->
            val encoded = layer.encode(xs.getRow(i))
            println("${encoded[0]}, ${encoded[1]}, ${ys[i].substring(0, 1)}")
        }
    }

}
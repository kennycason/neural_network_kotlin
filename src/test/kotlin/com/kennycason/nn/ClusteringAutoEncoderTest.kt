package com.kennycason.nn

import com.kennycason.nn.data.*
import com.kennycason.nn.math.Errors
import org.jblas.FloatMatrix
import org.junit.Test

/**
 * for these demos you should take the results and plot on a labeled x,y scatter plot
 */
class ClusteringAutoEncoderTest {

    @Test
    fun voterClustering() {
        val labeledData = VoterDataLoader.load()
        val xs = labeledData.xs
        val ys = labeledData.ys

        val layer = DeepAutoEncoder(
                learningRate = 0.05f,
                layerDimensions = arrayOf(
                        arrayOf(16, 32),
                        arrayOf(32, 32),
                        arrayOf(32, 16),
                        arrayOf(16, 2)
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


    // detect non-coding rna genes
    @Test
    fun codRnaShortClustering() {
        val labeledData = BinaryClassDataLoader.load("/data/cod_rna_short.svm")
        val xs = labeledData.xs.applyi { x -> // scale input
            if (Math.abs(x) > 100) {
                x / 1000f
            } else if (Math.abs(x) > 1) {
                x / 100f
            } else {
                x
            }
        }
        val ys = labeledData.ys

        val layer = DeepAutoEncoder(
                learningRate = 0.05f,
                layerDimensions = arrayOf(
                        arrayOf(8, 16),
//                        arrayOf(16, 8),
                        arrayOf(16, 2)
                ),
                log = true)

        layer.learn(xs.rowsAsList(), 1_000_000)

//        (0 until xs.rows).forEach { i ->
//            println("i: ${xs.getRow(i)}\no: ${layer.feedForward(xs.getRow(i))}")
//        }

        (0 until xs.rows).forEach { i ->
            val encoded = layer.encode(xs.getRow(i))
            println("${encoded[0]}, ${encoded[1]}, ${ys[i]}")
        }
    }
}
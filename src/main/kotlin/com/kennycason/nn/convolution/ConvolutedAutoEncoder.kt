package com.kennycason.nn.convolution

import com.kennycason.nn.AbstractAutoEncoder
import com.kennycason.nn.AutoEncoder
import com.kennycason.nn.math.ActivationFunction
import com.kennycason.nn.math.Errors
import com.kennycason.nn.math.Functions
import org.jblas.FloatMatrix
import java.util.*

class ConvolutedAutoEncoder(private val visibleDim: Dim,
                            private val hiddenDim: Dim,
                            private val paritions: Dim,
                            private val learningRate: Float,
                            private val visibleActivation: ActivationFunction = Functions.Sigmoid,
                            private val hiddenActivation: ActivationFunction = Functions.Sigmoid,
                            private val log: Boolean) : AbstractAutoEncoder() {

    private val random = Random()
    private val visibleChunkRows = visibleDim.rows / paritions.rows
    private val visibleChunkCols = visibleDim.cols / paritions.cols
    private val hiddenChunkRows = hiddenDim.rows / paritions.rows
    private val hiddenChunkCols = hiddenDim.cols / paritions.cols

    init {
        if (visibleDim.rows % visibleChunkRows != 0) {
            throw IllegalArgumentException(
                    "visible rows can not be evenly divided by row chunk size. rows: ${visibleDim.rows}, chunk size: $visibleChunkRows")
        }
        if (visibleDim.cols % visibleChunkCols != 0) {
            throw IllegalArgumentException(
                    "visible columns can not be evenly divided by column chunk size. rows: ${visibleDim.cols}, chunk size: $visibleChunkCols")
        }
        if (hiddenDim.rows % hiddenChunkRows != 0) {
            throw IllegalArgumentException(
                    "hidden rows can not be evenly divided by row chunk size. rows: ${hiddenDim.rows}, chunk size: $hiddenChunkRows")
        }
        if (hiddenDim.cols % hiddenChunkCols != 0) {
            throw IllegalArgumentException(
                    "hidden columns can not be evenly divided by column chunk size. rows: ${hiddenDim.cols}, chunk size: $hiddenChunkCols")
        }
    }

    private val autoencoders = Array<AutoEncoder>((visibleDim.rows / visibleChunkRows) * (visibleDim.cols / visibleChunkCols), {
        AutoEncoder(
                visibleSize = visibleChunkRows * visibleChunkCols,
                hiddenSize = hiddenChunkRows * hiddenChunkCols,
                learningRate = learningRate,
                visibleActivation = visibleActivation,
                hiddenActivation = hiddenActivation,
                log = log
        )
    })

    override fun learn(xs: List<FloatMatrix>, steps: Int) {
        var currentFeatures = xs

        (0.. steps).forEach { i ->
            // sgd
            val x = currentFeatures[random.nextInt(currentFeatures.size)]
            learn(x, 1)

            // report error for current training data TODO report rolling avg error
            if (i % 100 == 0 && log) {
                val error = Errors.compute(x, feedForward(x))
                println("$i -> error: $error")
            }
        }
    }

    override fun learn(x: FloatMatrix, steps: Int) {
        val xs = splitInput(x)

        (0 until autoencoders.size).forEach { i ->
            autoencoders[i].learn(xs[i], steps)
        }
    }

    override fun encode(x: FloatMatrix): FloatMatrix {
        val xs = splitInput(x)

        val ys = autoencoders
                .mapIndexed { i, autoEncoder -> autoEncoder.encode(xs[i]) }
                .toTypedArray()

        return MatrixRegions.merge(ys,
                hiddenDim.rows / hiddenChunkRows, hiddenDim.cols / hiddenChunkCols,
                hiddenChunkRows, hiddenChunkCols).m
    }

    override fun decode(y: FloatMatrix): FloatMatrix {
        val ys = splitOutput(y)

        val xs = autoencoders
                .mapIndexed { i, autoEncoder -> autoEncoder.decode(ys[i]) }
                .toTypedArray()

        return MatrixRegions.merge(xs,
                visibleDim.rows / visibleChunkRows, visibleDim.cols / visibleChunkCols,
                visibleChunkRows, visibleChunkCols).m
    }

    override fun feedForward(x: FloatMatrix) = decode(encode(x))

    private fun splitInput(x: FloatMatrix) = MatrixRegions.readRegions(
            FlatMatrix(x, visibleDim.rows, visibleDim.cols),
            visibleChunkRows, visibleChunkCols)

    private fun splitOutput(x: FloatMatrix) = MatrixRegions.readRegions(
            FlatMatrix(x, hiddenDim.rows, hiddenDim.cols),
            hiddenChunkRows, hiddenChunkCols)

}
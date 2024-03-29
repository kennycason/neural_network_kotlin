package com.kennycason.nn.optimization

import com.kennycason.nn.AutoEncoder
import com.kennycason.nn.math.Errors
import com.kennycason.nn.data.image.MNISTDataLoader
import com.kennycason.nn.data.PrintUtils
import com.kennycason.nn.learning_rate.FixedLearningRate
import org.jblas.FloatMatrix
import org.junit.Test
import java.io.File
import java.util.*

/**
 * full experiment of inactive feature generation vs standard SGD
 */
class MNISTAutoEncoderWithFeatureActivatorDemo {

    @Test
    fun mnistDataSet() {
        val xs = MNISTDataLoader.loadIdx3("/data/mnist/train-images-idx3-ubyte")

        val visibleSize = 28 * 28
        val hiddenSize = (visibleSize * 0.75).toInt()
        val learningRate = FixedLearningRate(0.1f)

        (1.. 10).forEach { x ->
            // control
            val layer = AutoEncoder(
                    learningRate = learningRate,
                    visibleSize = visibleSize,
                    hiddenSize = hiddenSize,
                    log = false)

            runExperiment(
                    id = "${x}_standard",
                    layer = layer,
                    xs = xs,
                    featureActivator = FeatureActivator(layer = layer, activationRate = 0.005f),
                    activateFeatures = false
            )

            // experiment
            val layer2 = AutoEncoder(
                    learningRate = learningRate,
                    visibleSize = visibleSize,
                    hiddenSize = hiddenSize,
                    log = false)
            runExperiment(
                    id = "${x}_feature_activation",
                    layer = layer2,
                    xs = xs,
                    featureActivator = FeatureActivator(layer = layer2, activationRate = 0.005f),
                    activateFeatures = true
            )
        }
    }

    private fun runExperiment(id: String,
                              layer: AutoEncoder,
                              xs: List<FloatMatrix>,
                              featureActivator: FeatureActivator,
                              activateFeatures: Boolean) {
        println("running experiment: $id")

        val random = Random()
        val errors = mutableListOf<Double>()
        val inactiveFeatures = mutableListOf<Int>()
        val start = System.currentTimeMillis()
        (0.. 100_000).forEach { i ->
            val x = xs.get(random.nextInt(xs.size))
            layer.learn(x, 1)

            featureActivator.sample(x)

            val error = Errors.compute(x, layer.feedForward(x))
            errors.add(error)

            if (i % 25 == 0) {
                println("$i -> error: $error")
                inactiveFeatures.add(featureActivator.generateInactivityVector().sum().toInt())
                if (activateFeatures) {
                    featureActivator.update()
                }
                featureActivator.clear()
            }
        }
        val trainTime = System.currentTimeMillis() - start
        println("${trainTime}ms")

        logGenerations(id, layer, xs)
        logResults(
                Result(
                        id = id,
                        visibleFeatures = layer.encode.rows,
                        hiddenFeatures = layer.encode.columns,
                        errors = errors,
                        inactiveFeatures = inactiveFeatures,
                        totalError = calculateError(layer, xs),
                        trainTime = trainTime,
                        layer = layer
                ))
    }

    private fun calculateError(layer: AutoEncoder, xs: List<FloatMatrix>): Double  {
        println("calculating error")
        var errorSum = 0.0
        val rowsToCheck = 10_000
        (0 until rowsToCheck).forEach{ i ->
            val x = xs.get(i)
            errorSum += Errors.compute(x, layer.feedForward(x))
            if (i % 100 == 0) {
                println("$i/$rowsToCheck")
            }
        }
        val error = errorSum / rowsToCheck
        println("total error: " + error)
        return error
    }

    private fun logGenerations(id: String, layer: AutoEncoder, xs: List<FloatMatrix>) {
        println("logging random generations")

        val file = File("/tmp/experiment/${id}_generation.log")

        val sb = StringBuilder()
        (0.. 100).forEach { i ->
            val x = xs.get(i)

            val error = Errors.compute(x, layer.feedForward(x))
            sb.append("input:\n" + PrintUtils.toPixelBox(x.toArray(), 28, 0.5) + "\n" +
                      "output:\n" + PrintUtils.toPixelBox(layer.feedForward(x).toArray(), 28, 0.7) + "\n" +
                      "error: $error\n")
        }
        file.writeText(sb.toString())
    }

    private fun logResults(result: Result) {
        println("logging results")
        val file = File("/tmp/experiment/${result.id}.log")
        file.writeText(result.toString())

        val weightFile = File("/tmp/experiment/${result.id}_weights.log")
        weightFile.writeText(
                        "encode:\n" +
                        result.layer.encode.toString("%f", "[", "]", ", ", "\n") + "\n" +
                        "decode:\n" +
                        result.layer.decode.toString("%f", "[", "]", ", ", "\n") + "\n"
        )

    }

    data class Result(val id: String,
                      val visibleFeatures: Int,
                      val hiddenFeatures: Int,
                      val errors: MutableList<Double> = mutableListOf(),
                      val inactiveFeatures: MutableList<Int> = mutableListOf(),
                      val totalError: Double,
                      val trainTime: Long,
                      val layer: AutoEncoder) {

        override fun toString() = "id: $id\n\n" +
                "visible x hidden: ${visibleFeatures}x${hiddenFeatures}\n\n" +
                "totalError: $totalError\n\n" +
                "trainTime: $trainTime\n\n" +
                "errors:\n${errors.joinToString("\n")}\n\n" +
                "inactiveFeatures:\n${inactiveFeatures.joinToString("\n")}"
    }

}
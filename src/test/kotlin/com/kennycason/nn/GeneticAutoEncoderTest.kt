package com.kennycason.nn

import com.kennycason.nn.genetic.Generation
import com.kennycason.nn.genetic.Generation.buildGeneration
import com.kennycason.nn.learning_rate.FixedLearningRate
import com.kennycason.nn.math.Errors
import org.jblas.FloatMatrix
import org.junit.Assert
import org.junit.Test
import toFormattedString

// doesn't work yet WIP
class GeneticAutoEncoderTest {

    private val visibleSize = 100
    private val hiddenSize = 30
    private val generationSize = 100
    private val trainingIterations = 1_000
    private val printIterationInterval = 10
    private val mutationRate = 0.02f

    @Test
    fun randomVector() {
        val x = FloatMatrix.rand(1, visibleSize)

        println("learn x: ${x.toFormattedString()}")

        var generation = buildGeneration(generationSize, visibleSize, hiddenSize)

        (0 until trainingIterations).forEach { i -> // iteration
            generation = Generation.live(
                generation, x,
                generationSize = generationSize,
                mutationRate = mutationRate
            )

            val mostFit = Generation.mostFit(generation, x)
            if (i % printIterationInterval == 0) {
                println("$i, ${mostFit.first}")
//                println("encode: " + mostFit.second.encode.toFormattedString())
//                println("decode: " + mostFit.second.decode.toFormattedString())
            }
        }

        val mostFit = Generation.mostFit(generation, x).second

        val y = mostFit.feedForward(x)
        val error = Errors.compute(x, y)
        println("input: " + x.toFormattedString())
        println("output: " + y.toFormattedString())
        println("error: $error")

        println("encode: " + mostFit.encode.toFormattedString())
        println("decode: " + mostFit.decode.toFormattedString())
    }

    @Test
    fun randomVectorWithGeneticAlgorithmsAndGradientDescent() {
        val x = FloatMatrix.rand(1, visibleSize)

        println("learn x: ${x.toFormattedString()}")

        var generation = buildGeneration(generationSize, visibleSize, hiddenSize)

        (0 until trainingIterations).forEach { i -> // iteration
            generation = Generation.live(generation, x,
                generationSize = generationSize,
                mutationRate = mutationRate
            )

            val mostFit = Generation.mostFit(generation, x)
            if (i % printIterationInterval == 0) {
                println("$i, ${mostFit.first}")
//                println("encode: " + mostFit.second.encode.toFormattedString())
//                println("decode: " + mostFit.second.decode.toFormattedString())
            }
        }

        val mostFit = Generation.mostFit(generation, x).second

        val y = mostFit.feedForward(x)
        val error = Errors.compute(x, y)
        println("input: " + x.toFormattedString())
        println("output: " + y.toFormattedString())
        println("error: $error")

        println("encode: " + mostFit.encode.toFormattedString())
        println("decode: " + mostFit.decode.toFormattedString())


        // now train with auto encoder via gradient descent
        val autoEncoder = AutoEncoder(
            learningRate = FixedLearningRate(0.01f),
            visibleSize = visibleSize,
            hiddenSize = hiddenSize
        )
        autoEncoder.encode = mostFit.encode
        autoEncoder.decode = mostFit.decode

        (0..1000).forEach { i ->
            autoEncoder.learn(x, 50)

            val yi = autoEncoder.feedForward(x)
            val errori = Errors.compute(x, yi)

            //println("$i -> error: $errori")
            println("$i, $errori")
        }

        val y2 = autoEncoder.feedForward(x)
        val error2 = Errors.compute(x, y2)
        println("input: " + x.toFormattedString())
        println("output: " + y2.toFormattedString())
        println("error: $error2")

        println("encode: " + autoEncoder.encode.toFormattedString())
        println("decode: " + autoEncoder.decode.toFormattedString())

        Assert.assertTrue(error2 < 0.1)
    }

}

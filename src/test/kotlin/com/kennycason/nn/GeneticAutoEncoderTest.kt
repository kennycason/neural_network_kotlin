package com.kennycason.nn

import com.kennycason.nn.genetic.Generation
import com.kennycason.nn.genetic.Generation.buildGeneration
import com.kennycason.nn.genetic.GeneticAutoEncoder
import com.kennycason.nn.learning_rate.FixedLearningRate
import com.kennycason.nn.math.Errors
import org.jblas.FloatMatrix
import org.junit.Assert
import org.junit.Test
import toFormattedString
import java.util.*

// doesn't work yet WIP
class GeneticAutoEncoderTest {

    private val visibleSize = 5
    private val hiddenSize = 3
    private val generationSize = 100
    private val trainingIterations = 1_000
    private val printIterationInterval = 10
    private val mutationRate = 0.05f

    @Test
    fun randomVector() {
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

        val error = Errors.compute(x, mostFit.feedForward(x))
        println("input: " + x.toFormattedString())
        println("output: " + mostFit.feedForward(x).toFormattedString())
        println("error: $error")

        println("encode: " + mostFit.encode.toFormattedString())
        println("decode: " + mostFit.decode.toFormattedString())

        Assert.assertTrue(error < 0.01)
    }

}

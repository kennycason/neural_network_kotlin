package com.kennycason.nn

import com.kennycason.nn.genetic.GeneticAutoEncoder
import com.kennycason.nn.learning_rate.FixedLearningRate
import com.kennycason.nn.math.Errors
import org.jblas.FloatMatrix
import org.junit.Assert
import org.junit.Test
import java.util.*

// doesn't work yet WIP
class GeneticAutoEncoderTest {
    private val random = Random()

    private val visibleSize = 10
    private val hiddenSize = 5
    private val generationSize = 20
    private val trainingIterations = 100_000
    private val mutationRate = 0.01f

    @Test
    fun randomVector() {
        val x = FloatMatrix.rand(1, visibleSize)

        val generation = buildGeneration(generationSize)

        (0 until trainingIterations).forEach { i -> // iteration
            val nextGeneration = evaluateGeneration(generation, x)
            generation.clear()
            generation.addAll(nextGeneration)

            val mostFit = mostFit(generation, x)
            if (i % 10_000 == 0) {
                println("$i, ${mostFit.first}")
            }
        }

        val mostFit = mostFit(generation, x).second

        val error = Errors.compute(x, mostFit.feedForward(x))
        println("input: " + x.toString("%f", "[", "]", ", ", "\n"))
        println("output: " + mostFit.feedForward(x).toString("%f", "[", "]", ", ", "\n"))
        println("error: $error")

        Assert.assertTrue(error < 0.1)
    }

    private fun evaluateGeneration(generation: List<GeneticAutoEncoder>, x: FloatMatrix): List<GeneticAutoEncoder> {
        val nextGeneration = mutableListOf<GeneticAutoEncoder>()
        val mostFit = takeTopNFit(generation, x, 5)
        nextGeneration.addAll(mostFit)
        while (nextGeneration.size < generationSize) {
            nextGeneration.add(randomlyBreed(mostFit))
        }
        return nextGeneration
    }

    private fun mostFit(nns: List<GeneticAutoEncoder>, x: FloatMatrix) = sortByFitness(nns, x).first()

    private fun randomlyBreed(mostFit: List<GeneticAutoEncoder>) =
            mostFit[random.nextInt(mostFit.size)]
                    .copy()
                    .mutate(mutationRate)

    private fun takeTopNFit(nns: List<GeneticAutoEncoder>, x: FloatMatrix, n: Int) =
            sortByFitness(nns, x)
                    .take(n)
                    .map { fitnessAndNn -> fitnessAndNn.second }

    private fun sortByFitness(nns: List<GeneticAutoEncoder>, x: FloatMatrix) = nns
            .map { nn -> Pair(nn.fitness(x), nn) }
            .sortedBy { fitnessAndNn -> fitnessAndNn.first }
            .toList()

    private fun buildGeneration(size: Int) = (0 until size)
            .map {
                GeneticAutoEncoder(
                        learningRate = FixedLearningRate(),
                        visibleSize = visibleSize,
                        hiddenSize = hiddenSize,
                        log = false)
            }
            .toMutableList()

}

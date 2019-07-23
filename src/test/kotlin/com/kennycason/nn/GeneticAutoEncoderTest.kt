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

    @Test
    fun randomVector() {
        val x = FloatMatrix.rand(1, 10)

        val nns = buildGeneration(30)

        (0 until 10000).forEach { i -> // iteration
            val nextGeneration = evaluateGeneration(nns, x)
            nns.clear()
            nns.addAll(nextGeneration)

            val mostFit = mostFit(nns, x)
            println("$i, ${mostFit.first}")
        }

        val mostFit = mostFit(nns, x).second

        val error = Errors.compute(x, mostFit.feedForward(x))
        println("input: " + x.toString("%f", "[", "]", ", ", "\n"))
        println("output: " + mostFit.feedForward(x).toString("%f", "[", "]", ", ", "\n"))
        println("error: $error")

        Assert.assertTrue(error < 0.1)
    }

    private fun evaluateGeneration(nns: List<GeneticAutoEncoder>, x: FloatMatrix): List<GeneticAutoEncoder> {
        val nextGeneration = mutableListOf<GeneticAutoEncoder>()
        val mostFit = takeTopNFit(nns, x, 5)
        nextGeneration.addAll(mostFit)
        while (nextGeneration.size < 30) {
            nextGeneration.add(randomlyBreed(mostFit))
        }
        return nextGeneration
    }

    private fun mostFit(nns: List<GeneticAutoEncoder>, x: FloatMatrix) = sortByFitness(nns, x).first()

    private fun randomlyBreed(mostFit: List<GeneticAutoEncoder>) =
            mostFit[random.nextInt(mostFit.size)]
                    .copy()
                    .mutate(0.02f)

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
                        visibleSize = 10,
                        hiddenSize = 5,
                        log = false)
            }
            .toMutableList()

}

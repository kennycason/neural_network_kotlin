package com.kennycason.nn.genetic

import com.kennycason.nn.learning_rate.FixedLearningRate
import org.jblas.FloatMatrix
import java.util.*

object Generation {
    private val random = Random()

    fun live(generation: List<GeneticAutoEncoder>,
             x: FloatMatrix,
             generationSize: Int = 100,
             mutationRate: Float = 0.05f): List<GeneticAutoEncoder> {
        val nextGeneration = mutableListOf<GeneticAutoEncoder>()
        val topNFit = takeTopNFit(generation, x, 5)
        nextGeneration.addAll(topNFit)
        while (nextGeneration.size < generationSize) {
            nextGeneration.add(randomlyBreed(topNFit, mutationRate))
        }
        return nextGeneration
    }
    fun mostFit(nns: List<GeneticAutoEncoder>, x: FloatMatrix) = sortByFitness(nns, x).first()

    private fun randomlyBreed(topNFit: List<GeneticAutoEncoder>, mutationRate: Float) =
        topNFit[random.nextInt(topNFit.size)]
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

    fun buildGeneration(size: Int, visibleSize: Int, hiddenSize: Int) = (0 until size)
        .map {
            GeneticAutoEncoder(
                learningRate = FixedLearningRate(),
                visibleSize = visibleSize,
                hiddenSize = hiddenSize,
                log = false
            )
        }
        .toList()

}
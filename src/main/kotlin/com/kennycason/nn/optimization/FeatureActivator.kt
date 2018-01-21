package com.kennycason.nn.optimization

import com.kennycason.nn.AutoEncoder
import com.kennycason.nn.apply
import org.jblas.FloatMatrix


/**
 * Stimulate inactive neurons/features
 */
class FeatureActivator(val layer: AutoEncoder,
                       val inactiveThreshold: Float = 0.1f,
                       val activationRate: Float = 0.005f) {

    private val sampleSums = FloatMatrix(1, layer.encode.columns)

    private var samples = 0

    fun sample(x: FloatMatrix) {
        sampleSums.addi(layer.encode(x))
        samples++
    }

    fun clear() {
        sampleSums.muli(0.0f)
        samples = 0
    }

    // stimulate inactive features
    fun update() {
        val inactive = generateInactivityVector()
        if (inactive.sum() == 0.0f) { return }

        // println("stimulate inactivate neurons")
        layer.encode.addiRowVector(inactive.mul(activationRate))
    }

    // build a binary (1,0) vector where inactive features are set to 1
    // use this vector for downstream updates to nn
    fun generateInactivityVector() = sampleSums.apply { i -> if (i < inactiveThreshold * samples) { 1.0f } else { 0.0f } }

}
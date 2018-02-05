package com.kennycason.nn.learning_rate


class FixedLearningRate(private val learningRate: Float = 0.1f) : LearningRate {
    override fun get() = learningRate
}

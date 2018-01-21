package com.kennycason.nn.learning_rate


class ConstantLearningRate(private val learningRate: Float) : LearningRate {
    override fun apply() = learningRate
}
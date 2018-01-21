package com.kennycason.nn.learning_rate


class ConstantLearningRate(private val learningRate: Double) : LearningRate {
    override fun apply() = learningRate
}
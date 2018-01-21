package com.kennycason.nn.data


object PrintUtils {

    fun toPixelBox(arrays: Array<FloatArray>, threshold: Double): String {
        val stringBuilder = StringBuilder()
        for (array in arrays) {
            for (i in array.indices) {
                if (array[i] >= threshold) {
                    stringBuilder.append("■")
                } else {
                    stringBuilder.append("□")
                }
            }
            stringBuilder.append('\n')
        }
        return stringBuilder.toString()
    }

    fun toPixelBox(array: FloatArray, columnSize: Int, threshold: Double): String {
        val rowSize = array.size / columnSize
        val matrix = Array(rowSize) { FloatArray(columnSize) }
        for (i in 0 until rowSize) {
            for (j in 0 until columnSize) {
                matrix[i][j] = array[i * columnSize + j]
            }
        }
        return toPixelBox(matrix, threshold)
    }

}
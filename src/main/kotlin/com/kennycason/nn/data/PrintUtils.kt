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

    fun toPixelBox(array: FloatArray, columns: Int, threshold: Double): String {
        val rows = array.size / columns
        val matrix = Array(rows) { FloatArray(columns) }
        for (col in 0 until columns) {
            for (row in 0 until rows) {
                matrix[row][col] = array[col * columns + row]
            }
        }
        return toPixelBox(matrix, threshold)
    }

}
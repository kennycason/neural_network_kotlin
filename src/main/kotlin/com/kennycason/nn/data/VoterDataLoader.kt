package com.kennycason.nn.data

import org.apache.commons.io.IOUtils
import org.jblas.FloatMatrix
import java.awt.Image


object VoterDataLoader {

    fun load() : LabeledData {
        val resource = Image::class.java.getResourceAsStream("/data/voter_history.data")
        val lines = IOUtils.readLines(resource, "ascii")

        val features = 16
        val xs = FloatMatrix(lines.size, features)
        val ys = Array<String>(lines.size, { "" })
        lines.forEachIndexed { i, row ->
            val columns = row.split(",")
            val x = FloatMatrix(1, features)
            (0 until features).forEach { j ->
                val vote = when (columns[j]) {
                    "y" -> 1.0f
                    "n" -> 0.0f
                    "u" -> 0.5f
                    else -> throw IllegalArgumentException("Unknown vote: ${columns[j]}")
                }
                x.put(j, vote)
            }
            xs.putRow(i, x)
            ys.set(i, columns[16])
        }
        return LabeledData(xs, ys)
    }
}
package com.kennycason.nn.data

import org.apache.commons.io.IOUtils
import org.jblas.FloatMatrix
import java.awt.Image

object BinaryClassDataLoader {

    fun load(resourcePathName: String) : LabeledData {
        val resource = Image::class.java.getResourceAsStream(resourcePathName)
        val lines = IOUtils.readLines(resource, "ascii")

        val featureDimensions = featureDimensions(lines.first())
        val xs = FloatMatrix(lines.size, featureDimensions)
        val ys = Array<String>(lines.size, { "" })
        lines.forEachIndexed { i, row ->
            val columns = row.split(" ")
            val x = FloatMatrix(1, featureDimensions)
            (0 until  featureDimensions).forEach { j ->
                x.put(j, columns[j + 1].toFloat())
            }
            xs.putRow(i, x)
            ys.set(i, columns[0])
        }
        return LabeledData(xs, ys)
    }

    private fun featureDimensions(line: String) = line.split(" ").size - 1

}
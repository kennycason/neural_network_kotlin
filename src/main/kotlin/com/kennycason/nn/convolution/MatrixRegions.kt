package com.kennycason.nn.convolution

import org.jblas.FloatMatrix

// read and write select regions from a "2d" matrix (represented as a 1d matrix)
// the vectors are read to a 1d vector (so it can be easily consumed by neural network
object MatrixRegions {

    fun read(m: FlatMatrix,
             startRow: Int, startCol: Int,
             rows: Int, cols: Int) =
            readi(FloatMatrix(1, rows * cols),
                    m,
                    startRow, startCol,
                    rows, cols)

    fun readi(region: FloatMatrix,
              m: FlatMatrix,
              startRow: Int, startCol: Int,
              rows: Int, cols: Int): FloatMatrix {
        var i = 0
        (0 until cols).forEach { col ->
            (0 until rows).forEach { row ->
                region.put(
                        i++,
                        m.m.get((startCol * m.rows) +
                                (col * m.rows) +
                                startRow +
                                row))
            }
        }
        return region
    }

    fun merge(ms: Array<FloatMatrix>,
              rows: Int, cols: Int,
              rowsPerChunk: Int, colsPerChunk: Int) =
            mergei(FloatMatrix(1, (rows * rowsPerChunk) * (cols * colsPerChunk)),
                    ms,
                    rows, cols,
                    rowsPerChunk, colsPerChunk)

    fun mergei(merged: FloatMatrix,
               ms: Array<FloatMatrix>,
               rows: Int, cols: Int,
               rowsPerChunk: Int, colsPerChunk: Int
    ): FlatMatrix {
        val totalRows = rows * rowsPerChunk
        val totalCols = cols * colsPerChunk

        var i = 0
        (0 until cols).forEach { col ->
            (0 until rows).forEach { row ->
                val m = ms[i++]
                val startRow = row * rowsPerChunk
                val startCol = col * colsPerChunk

                var j = 0
                (0 until colsPerChunk).forEach { mCol ->
                    (0 until rowsPerChunk).forEach { mRow ->
                        merged.put(
                                (startCol * totalRows) +
                                        (mCol * totalRows) +
                                        startRow +
                                        mRow,
                                m.get(j++))
                    }
                }
            }
        }
        return FlatMatrix(merged, totalRows, totalCols)
    }


    fun readRegions(m: FlatMatrix,
                    rowsPerChunk: Int, colsPerChunk: Int) =
            readRegionsi(
                    Array<FloatMatrix>(m.rows / rowsPerChunk *  m.cols / colsPerChunk,
                            { FloatMatrix.zeros(1, rowsPerChunk * colsPerChunk) }),
                    m,
                    rowsPerChunk, colsPerChunk)


    fun readRegionsi(ms: Array<FloatMatrix>,
                     m: FlatMatrix,
                     rowsPerChunk: Int, colsPerChunk: Int): Array<FloatMatrix> {

        val rows = m.rows / rowsPerChunk
        val cols = m.cols / colsPerChunk

        var i = 0
        (0 until cols).forEach { col ->
            (0 until rows).forEach { row ->
                readi(ms[i++],
                        m,
                        row * rowsPerChunk, col * colsPerChunk,
                        rowsPerChunk, colsPerChunk)
            }
        }

        return ms
    }

}

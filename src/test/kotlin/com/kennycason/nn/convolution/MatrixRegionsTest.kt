package com.kennycason.nn.convolution

import org.jblas.FloatMatrix
import org.junit.Assert
import org.junit.Test


class MatrixRegionsTest {

    @Test
    fun readThenWrite() {
        val m = FlatMatrix(fill(4, 4), 4, 4)

        // read single regions
        val tl = MatrixRegions.read(m, 0, 0, 2, 2)
        val tr = MatrixRegions.read(m, 0, 2, 2, 2)
        val bl = MatrixRegions.read(m, 2, 0, 2, 2)
        val br = MatrixRegions.read(m, 2, 2, 2, 2)

        Assert.assertEquals(1.0f, tl.get(0))
        Assert.assertEquals(2.0f, tl.get(1))
        Assert.assertEquals(5.0f, tl.get(2))
        Assert.assertEquals(6.0f, tl.get(3))

        Assert.assertEquals(9.0f, tr.get(0))
        Assert.assertEquals(10.0f, tr.get(1))
        Assert.assertEquals(13.0f, tr.get(2))
        Assert.assertEquals(14.0f, tr.get(3))

        Assert.assertEquals(3.0f, bl.get(0))
        Assert.assertEquals(4.0f, bl.get(1))
        Assert.assertEquals(7.0f, bl.get(2))
        Assert.assertEquals(8.0f, bl.get(3))

        Assert.assertEquals(11.0f, br.get(0))
        Assert.assertEquals(12.0f, br.get(1))
        Assert.assertEquals(15.0f, br.get(2))
        Assert.assertEquals(16.0f, br.get(3))


        // re-merge regions
        val ms = arrayOf(
                tl, bl,
                tr, br
        )
        val merged = MatrixRegions.merge(ms, 2, 2, 2, 2)
        Assert.assertEquals(m.m, merged.m)


        // bulk read single regions
        val split = MatrixRegions.readRegions(merged, 2, 2)
        Assert.assertEquals(tl, split[0])
        Assert.assertEquals(bl, split[1])
        Assert.assertEquals(tr, split[2])
        Assert.assertEquals(br, split[3])


        // re-merge
        val remerged = MatrixRegions.merge(split, 2, 2, 2, 2)
        Assert.assertEquals(m.m, remerged.m)
    }

    @Test
    fun nonSquare() {
        val filled = fill(480, 640)
        val m = FlatMatrix(filled, 480, 640)

        val tl = MatrixRegions.read(m, 0, 0, 2, 2)
        val br = MatrixRegions.read(m, 478, 638, 2, 2)

        Assert.assertEquals(1.0f, tl.get(0))
        Assert.assertEquals(2.0f, tl.get(1))
        Assert.assertEquals(481.0f, tl.get(2))
        Assert.assertEquals(482.0f, tl.get(3))

        Assert.assertEquals(306719.0f, br.get(0))
        Assert.assertEquals(306720.0f, br.get(1))
        Assert.assertEquals(307199.0f, br.get(2))
        Assert.assertEquals(307200.0f, br.get(3))


        val split = MatrixRegions.readRegions(m, 16, 16)
        Assert.assertEquals(30 * 40, split.size)
        Assert.assertEquals(m.m, MatrixRegions.merge(split, 30, 40, 16, 16).m)
    }

    @Test
    fun inPlaceEqualsCopy() {
        val filled = fill(640, 480)
        val m = FlatMatrix(filled, 640, 480)

        // copy
        val split = MatrixRegions.readRegions(m, 16, 16)
        val merged = MatrixRegions.merge(split, 30, 40, 16, 16)

        // inplace
        val chunks = 30 * 40
        val splitInplace = Array<FloatMatrix>(chunks,
                { FloatMatrix.zeros(1, 16 * 16) })
        val mergedInplace = FloatMatrix(1, 480 * 640)
        MatrixRegions.readRegionsi(splitInplace, m, 16, 16)
        MatrixRegions.mergei(mergedInplace, splitInplace, 30, 40, 16, 16)

        (0 until split.size).forEach { i ->
            Assert.assertEquals(splitInplace[i], split[i])
        }
        Assert.assertEquals(mergedInplace, merged.m)
    }

    @Test
    fun time() {
        val filled = fill(640, 480)
        val m = FlatMatrix(filled, 640, 480)

        // copy
        val start = System.currentTimeMillis()
        (0 until  1_000).forEach {
            val split = MatrixRegions.readRegions(m, 16, 16)
            val merged = MatrixRegions.merge(split, 30, 40, 16, 16)
        }
        println("test 1: ${System.currentTimeMillis() - start} ms")


        // inplace
        val chunks = 30 * 40
        val split = Array<FloatMatrix>(chunks,
                { FloatMatrix.zeros(1, 16 * 16) })
        val merged = FloatMatrix(1, 480 * 640)
        val start2 = System.currentTimeMillis()
        (0 until  1_000).forEach {
            MatrixRegions.readRegionsi(split, m, 16, 16)
            MatrixRegions.mergei(merged, split, 30, 40, 16, 16)
        }
        println("test 2: ${System.currentTimeMillis() - start2} ms")
    }


    /**
        as a single row matrix
        [0.000000, 4.000000, 8.000000, 12.000000,
        1.000000, 5.000000, 9.000000, 13.000000,
        2.000000, 6.000000, 10.000000, 14.000000,
        3.000000, 7.000000, 11.000000, 15.000000]
     */
    private fun fill(rows: Int, cols: Int): FloatMatrix {
        val size = rows * cols
        val m = FloatMatrix(1, size)
        (0 until size).forEach { i ->
            m.put(i, (i + 1).toFloat())
        }
        return m
    }

    private fun fill2(rows: Int, cols: Int): FloatMatrix {
        val m = FloatMatrix(rows, cols)
        var i = 0
        (0 until cols).forEach { col ->
            (0 until rows).forEach { row ->
                m.put(row, col, i++.toFloat())
            }
        }
        return m
    }

}
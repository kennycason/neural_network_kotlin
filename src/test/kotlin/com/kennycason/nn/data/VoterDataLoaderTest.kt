package com.kennycason.nn.data

import org.junit.Assert
import org.junit.Test

class VoterDataLoaderTest {

    @Test
    fun load() {
        val labeledData = VoterDataLoader.load()

        // first row
        // n,n,y,y,y,y,n,n,y,y,n,y,y,y,n,y,republican
        val x = labeledData.xs.getRow(0)
        Assert.assertEquals(0.0f, x[0])
        Assert.assertEquals(0.0f, x[1])
        Assert.assertEquals(1.0f, x[2])
        Assert.assertEquals(1.0f, x[3])
        Assert.assertEquals(1.0f, x[4])
        Assert.assertEquals(1.0f, x[5])
        Assert.assertEquals(0.0f, x[6])
        Assert.assertEquals(0.0f, x[7])
        Assert.assertEquals(1.0f, x[8])
        Assert.assertEquals(1.0f, x[9])
        Assert.assertEquals(0.0f, x[10])
        Assert.assertEquals(1.0f, x[11])
        Assert.assertEquals(1.0f, x[12])
        Assert.assertEquals(1.0f, x[13])
        Assert.assertEquals(0.0f, x[14])
        Assert.assertEquals(1.0f, x[15])

        Assert.assertEquals("republican", labeledData.ys[0])
    }
}
package com.kennycason.nn.convolution

import org.junit.Test


class ConvolutedAutoEncoderFactoryTest {

    @Test
    fun layerDimensions() {
        val layerDimensions = ConvolutedAutoEncoderFactory.generateDimensions(
                initialVisibleDim = Dim(480, 640),
                layers = 20)

        println(layerDimensions.joinToString(",\n"))
    }

    @Test
    fun convolutedAutoEncoder() {
        ConvolutedAutoEncoderFactory.generateConvolutedAutoencoder(
                initialVisibleDim = Dim(480, 640),
                layers = 20)
    }

}
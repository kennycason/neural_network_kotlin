package com.kennycason.nn.convolution


data class ConvolutedLayerConfig(val visibleDim: Dim,
                                 val hiddenDim: Dim,
                                 val partitions: Dim)
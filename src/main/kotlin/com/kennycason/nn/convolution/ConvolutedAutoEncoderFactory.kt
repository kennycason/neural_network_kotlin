package com.kennycason.nn.convolution

/**
 * Generate a convolutional network based on input parameters
 *
 * Despite each layer really being a 1xn matrix, many features are actually 2d.
 * As such, special care must be taken when building convolution structures to
 * preserve spatial relationships
 *
 * linearly scale down to hidden size
 */
object ConvolutedAutoEncoderFactory {

    fun generateConvolutedAutoencoder(initialVisibleDim: Dim,
                                      layers: Int,
                                      minHiddenDim: Int = 10,
                                      learningRate: Float = 0.1f): DeepConvolutedAutoEncoder {
        val layerDimensions = generateDimensions(initialVisibleDim, layers, minHiddenDim)

        println(layerDimensions.joinToString(",\n"))

        val layers = layerDimensions.map { dim ->
            ConvolutedLayer(visibleDim = dim.visibleDim,
                    paritions = dim.partitions,
                    hiddenDim = dim.hiddenDim,
                    learningRate = learningRate,
                    log = false)
        }.toTypedArray()

        return DeepConvolutedAutoEncoder(layers)
    }

    fun generateDimensions(initialVisibleDim: Dim,
                           layers: Int,
                           minHiddenDim: Int = 10): Array<ConvolutedLayerConfig> {

        val colStepSize = initialVisibleDim.cols / layers
        val rowStepSize = initialVisibleDim.rows / layers

        return Array<ConvolutedLayerConfig>(layers, { i ->
            val visibleDim = if (i == 0) {
                Dim(
                        rows = initialVisibleDim.rows,
                        cols = initialVisibleDim.cols
                )
            } else {
                Dim(
                        rows = (layers - i) * rowStepSize,
                        cols = (layers - i) * colStepSize
                )
            }
            val hiddenDim = Dim(
                    rows = Math.max(1, (layers - i - 1) * rowStepSize),
                    cols = Math.max(1, (layers - i - 1) * colStepSize)
            )
            val gcdRow = gcd(visibleDim.rows, hiddenDim.rows)
            val gcdCol = gcd(visibleDim.cols, hiddenDim.cols)
            ConvolutedLayerConfig(
                    visibleDim = visibleDim,
                    hiddenDim = hiddenDim,
                    partitions = Dim(gcdRow, gcdCol)
            )
        })
    }

    private fun findIdealChunkDim(visible: Dim, hidden: Dim) = Dim(
            rows = lcd(visible.rows, hidden.rows), // findIdealN(visible.rows),
            cols = lcd(visible.cols, hidden.cols) // findIdealN(visible.cols)
    )

    private fun lcm(a: Int, b: Int): Int {
        return a * (b / gcd(a, b))
    }

    private fun gcd(a: Int, b: Int): Int {
        var a = a
        var b = b
        while (b > 0) {
            val temp = b
            b = a % b
            a = temp
        }
        return a
    }

    private fun lcd(a: Int, b: Int): Int {
        (4.. Math.min(a, b)).forEach { i ->
            if (a % i == 0 && b % i == 0) {
                return i
            }
        }
        // prefer not to do chunks this small unless other lcds can't be found
        if (a % 3 == 0 && b % 3 == 0) { return 3 }
        if (a % 2 == 0 && b % 2 == 0) { return 2 }
        return 1
    }

    private fun ceilRoot(x: Int) = Math.ceil(Math.sqrt(x.toDouble())).toInt()

}
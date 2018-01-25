package com.kennycason.nn.data.image

import org.junit.Test

class CompositeImageReaderTest {

    @Test
    fun pokemon() {
        val allPokemon = CompositeImageReader.read(
                file = "/data/pokemon_151.png",
                matrixImageEncoder = MatrixRGBImageEncoder(),
                rows = 11,
                cols = 15,
                n = 151)
        // images are 60x60 (features x3 rgb = 10800)
        val imageDecoder = MatrixRGBImageDecoder(rows = 60)
        val image = imageDecoder.decode(allPokemon[0])
        image.save("/tmp/bulbasaur.png")
    }

}
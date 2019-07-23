package com.kennycason.nn.data.image

import java.awt.Color
import java.awt.RenderingHints
import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO


class Image {

    val bi: BufferedImage

    constructor(file: String) {
        val resource = Thread.currentThread().contextClassLoader.getResourceAsStream(file)
        if (resource != null) {
            bi = ImageIO.read(resource)
            return
        }

        val fileResource = File(file)
        if (fileResource.exists()) {
            bi = ImageIO.read(fileResource)
            return
        }
        throw RuntimeException("Could not find [$file] in resources or on file system")
    }

    constructor(bi: BufferedImage) {
        this.bi = bi
    }

    operator fun set(x: Int, y: Int, rgb: Int) {
        bi.setRGB(x, y, rgb)
    }

    operator fun get(x: Int, y: Int): Int {
        return bi.getRGB(x, y)
    }

    fun data(): BufferedImage {
        return bi
    }

    fun width(): Int {
        return bi.width
    }

    fun height(): Int {
        return bi.height
    }

    private fun getFormat(file: String): String {
        val parts = file.split(".")
        return parts[parts.size - 1]
    }

    fun save(file: String) {
        println("Writing file: $file")
        ImageIO.write(bi!!, getFormat(file), File(file))
    }

    fun saveThumbnail(file: String, scale: Double) {
        var width = (bi.width * scale).toInt()
        var height = (bi.height * scale).toInt()

        val imgWidth = bi.width
        val imgHeight = bi.height
        if (imgWidth * height < imgHeight * width) {
            width = imgWidth * height / imgHeight
        }
        else {
            height = imgHeight * width / imgWidth
        }
        val newImage = BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)
        val g = newImage.createGraphics()

        try {
            g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC)
            g.background = Color.BLACK
            g.clearRect(0, 0, width, height)
            g.drawImage(bi, 0, 0, width, height, null)
        }
        finally {
            g.dispose()
        }

        println("Writing file: $file")
        ImageIO.write(newImage, getFormat(file), File(file))
    }

}
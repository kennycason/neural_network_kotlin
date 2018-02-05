package com.kennycason.nn.data.image.helper

import com.kennycason.nn.data.image.Image
import java.awt.image.BufferedImage


object ImageHelper {

    fun equals(a: Image, b: Image) = equals(a.data(), b.data())

    fun equals(a: BufferedImage, b: BufferedImage): Boolean {
        if (a.width != b.width || a.height != b.height) {
            return false
        }

        for (y in 0 until a.height) {
            for (x in 0 until a.width) {
                if (a.getRGB(x, y) != b.getRGB(x, y)) {
                    return false
                }
            }
        }
        return true
    }

}
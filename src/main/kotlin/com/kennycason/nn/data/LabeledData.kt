package com.kennycason.nn.data

import com.sun.xml.internal.fastinfoset.util.StringArray
import org.jblas.FloatMatrix


data class LabeledData(val xs: FloatMatrix,
                       val ys: Array<String>)
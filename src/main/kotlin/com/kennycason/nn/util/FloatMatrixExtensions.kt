import org.jblas.FloatMatrix

fun FloatMatrix.toFormattedString(): String {
    return toString("%f", "[", "]", ", ", "\n")
}
package utils

import org.jetbrains.numkt.core.KtNDArray
import org.jetbrains.numkt.core.*
import org.jetbrains.numkt.math.*

@ExperimentalNumkt
fun euclidean(x: KtNDArray<Double>, y:KtNDArray<Double>): KtNDArray<Double> {
    return sqrt(sum(square(x - y), axis = 0))
}

@ExperimentalNumkt
fun manhattan(x: KtNDArray<Double>, y: KtNDArray<Double>): KtNDArray<Double> {
    return absolute(x - y)
}

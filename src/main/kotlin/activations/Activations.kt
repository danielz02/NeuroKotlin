package activations

import org.jetbrains.numkt.core.ExperimentalNumkt
import org.jetbrains.numkt.core.KtNDArray
import org.jetbrains.numkt.core.reshape
import org.jetbrains.numkt.math.exp
import org.jetbrains.numkt.math.maximum
import org.jetbrains.numkt.math.plus
import org.jetbrains.numkt.math.tanh
import org.jetbrains.numkt.math.unaryMinus
import org.jetbrains.numkt.zeros

@ExperimentalNumkt
abstract class Activations {
    operator fun invoke(input: KtNDArray<Double>): KtNDArray<Double> {
        return if (input.ndim == 1) {
            input.reshape(1, -1)
        } else {
            fn(input)
        }
    }
    abstract fun fn(input: KtNDArray<Double>): KtNDArray<Double>
    abstract fun grad(input: KtNDArray<Double>): KtNDArray<Double>
    abstract override fun toString(): String
}

/**
 * Computes the forward pass for a layer of rectified linear units (ReLUs).
 * @param x: Inputs, of any shape
 * @return a list of(output, x) where output is the same shape as x, x as cache.
 */
@ExperimentalNumkt
fun ReLU(x: KtNDArray<Float>): List<KtNDArray<Float>> {
    val out = maximum(x, zeros(x.shape[0], x.shape[1]))
    val cache = x
    return listOf(out, cache)
}

/** Computes the forward pass for a layer of Sigmoid, same signiture as ReLU */
@ExperimentalNumkt
fun Sigmoid(x: KtNDArray<Float>): List<KtNDArray<Float>> {
    val out = 1 / 1 + exp(-x) as KtNDArray<Float>
    val cache = x
    return listOf(out, cache)
}

/** Computes the forward pass for a layer of tanh, same signiture as ReLU */
@ExperimentalNumkt
fun Tanh(x: KtNDArray<Float>): List<KtNDArray<Float>> {
    val out = tanh(x) as KtNDArray<Float>
    val cache = x
    return listOf(out, cache)
}

package forward

import org.jetbrains.numkt.core.*
import org.jetbrains.numkt.math.*
import org.jetbrains.numkt.zeros

/**
 * Computes the forward pass for a layer of rectified linear units (ReLUs).
 * @param x: Inputs, of any shape
 * @return a list of(output, x) where output is the same shape as x, x as cache.
 */
@ExperimentalNumkt
fun ReLU(x: KtNDArray<Float>): List<KtNDArray<Float>> {
    val out = maximum(x, zeros(x.shape[0], x.shape[1]));
    val cache = x;
    return listOf(out, cache);
}

/** Computes the forward pass for a layer of Sigmoid, same signiture as ReLU */
@ExperimentalNumkt
fun Sigmoid(x: KtNDArray<Float>): List<KtNDArray<Float>> {
    val out = 1 / 1 + exp(-x) as KtNDArray<Float>;
    val cache = x;
    return listOf(out, cache);
}

/** Computes the forward pass for a layer of tanh, same signiture as ReLU */
@ExperimentalNumkt
fun Tanh(x: KtNDArray<Float>): List<KtNDArray<Float>> {
    val out = tanh(x) as KtNDArray<Float>;
    val cache = x;
    return listOf(out, cache);
}

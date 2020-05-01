package layers

import activations.ReLU
import activations.Sigmoid
import activations.Tanh
import org.jetbrains.numkt.core.*
import org.jetbrains.numkt.math.*

/**
 * Perform a fully connected layer forward pass with x@w + b
 * @param x: an array of mini-batch of N examples, size (N, d_1, d_2...d_k),
 *           each mini-batch have shape (d_1, d_2...d_k), will reshape into
 *           size (N, D). x in returned cache is the same shape as original x.
 * @param w: an array of weight size (D, M)
 * @param b: bias of shape (M,)
 * @return a list in order of (out, x, w, b)
 */
@ExperimentalNumkt
fun affine(x: KtNDArray<Float>, w: KtNDArray<Float>, b: KtNDArray<Float>): List<KtNDArray<Float>> {

    val shape = x.shape
    x.reshape(shape[0], -1)
    val out = x.dot(w) + b
    x.reshape(shape[0], shape[1])

    return listOf(out, x, w, b)
}

/**
 * A convience layer that performs affine transform forward by a ReLU
 * Wraps a affine layer and a ReLU layer
 * @param x: input to affine layer
 * @param w: weight to affine
 * @param b: bias
 * @return list in order(output, (affine caches), (ReLU cache))
 */
@ExperimentalNumkt
fun AffineReLUForward(x: KtNDArray<Float>, w: KtNDArray<Float>, b: KtNDArray<Float>): List<KtNDArray<Float>> {
    val afCache = affine(x, w, b)
    val a = afCache[0]
    val reLUCache = ReLU(a)
    return listOf(reLUCache[0], afCache[1], afCache[2], afCache[3], reLUCache[1])
}

/** Convience layer for sigmoid forward */
@ExperimentalNumkt
fun AffineSigmoidForward(x: KtNDArray<Float>, w: KtNDArray<Float>, b: KtNDArray<Float>): List<KtNDArray<Float>> {
    val afCache = affine(x, w, b)
    val a = afCache[0]
    val sigCache = Sigmoid(a)
    return listOf(sigCache[0], afCache[1], afCache[2], afCache[3], sigCache[1])
}

/** Convience layer for tanh forward */
@ExperimentalNumkt
fun AffineTanhForward(x: KtNDArray<Float>, w: KtNDArray<Float>, b: KtNDArray<Float>): List<KtNDArray<Float>> {
    val afCache = affine(x, w, b)
    val a = afCache[0]
    val tanhCache = Tanh(a)
    return listOf(tanhCache[0], afCache[1], afCache[2], afCache[3], tanhCache[1])
}

// !TODO cov/bn/dropout

// !TODO: Conv/bn/dropout

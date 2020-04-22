package forward

import org.jetbrains.numkt.core.ExperimentalNumkt
import org.jetbrains.numkt.core.KtNDArray

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
    val afCache = Affine(x, w, b);
    val a = afCache[0];
    val reLUCache = ReLU(a);
    return listOf(reLUCache[0], afCache[1], afCache[2], afCache[3], reLUCache[1]);
}

/** Convience layer for sigmoid forward */
@ExperimentalNumkt
fun AffineSigmoidForward(x: KtNDArray<Float>, w: KtNDArray<Float>, b: KtNDArray<Float>): List<KtNDArray<Float>> {
    val afCache = Affine(x, w, b);
    val a = afCache[0];
    val sigCache = Sigmoid(a);
    return listOf(sigCache[0], afCache[1], afCache[2], afCache[3], sigCache[1]);
}

/** Convience layer for tanh forward */
@ExperimentalNumkt
fun AffineTanhForward(x: KtNDArray<Float>, w: KtNDArray<Float>, b: KtNDArray<Float>): List<KtNDArray<Float>> {
    val afCache = Affine(x, w, b);
    val a = afCache[0];
    val tanhCache = Tanh(a);
    return listOf(tanhCache[0], afCache[1], afCache[2], afCache[3], tanhCache[1]);
}

//!TODO cov/bn/dropout
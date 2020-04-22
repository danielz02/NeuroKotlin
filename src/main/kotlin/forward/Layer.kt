package forward

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
fun Affine(x: KtNDArray<Float>, w: KtNDArray<Float>, b: KtNDArray<Float>): List<KtNDArray<Float>> {

    val shape = x.shape;
    x.reshape(shape[0], -1);
    val out = x.dot(w) + b;
    x.reshape(shape[0], shape[1]);

    return listOf(out, x, w, b);
}

//!TODO: Conv/bn/dropout

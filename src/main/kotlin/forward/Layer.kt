package forward

import org.jetbrains.numkt.core.*
import org.jetbrains.numkt.math.*

object Layer{
    @ExperimentalNumkt
    fun Affine(x: KtNDArray<Float>, w: KtNDArray<Float>, b: KtNDArray<Float>): List<KtNDArray<Float>> {

        val shape = x.shape;
        x.reshape(shape[0], -1);
        val out = x.dot(w) + b;
        x.reshape(shape[0], shape[1]);

        return listOf(out, x, w, b);
    }

    //!TODO: Conv/bn/dropout
}
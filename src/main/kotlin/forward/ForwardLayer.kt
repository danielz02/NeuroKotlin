package forward

import org.jetbrains.numkt.core.*
import org.jetbrains.numkt.math.*
import org.jetbrains.numkt.zeros
import org.jetbrains.numkt.array

import forward.Layer.Affine


object ForwardLayer {
    @ExperimentalNumkt
    fun AffineReLUForward(x: KtNDArray<Float>, w: KtNDArray<Float>, b: KtNDArray<Float>): List<KtNDArray<Float>> {
        val afCache = Affine(x,w,b);
        val a = afCache[0];
        val reLUCache = Activation.ReLU(a);
        return listOf(reLUCache[0], afCache[1], afCache[2], afCache[3], reLUCache[1]);
    }
    //!TODO TanH and Sigmoid in same manner
}
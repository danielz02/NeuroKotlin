package forward

import org.jetbrains.numkt.core.*
import org.jetbrains.numkt.math.*
import org.jetbrains.numkt.zeros

object Activation{
    @ExperimentalNumkt
    fun ReLU(x: KtNDArray<Float>): List<KtNDArray<Float>> {
        val out = maximum(x, zeros(x.shape[0], x.shape[1]));
        val cache = x;
        return listOf(out, cache);
    }

    @ExperimentalNumkt
    fun Sigmoid(x: KtNDArray<Float>): List<KtNDArray<Float>> {
        val out = Sigmoid(x) as KtNDArray<Float>;
        val cache = x;
        return listOf(out, cache);
    }

    @ExperimentalNumkt
    fun TanH(x: KtNDArray<Float>): List<KtNDArray<Float>> {
        val out = tanh(x) as KtNDArray<Float>;
        val cache = x;
        return listOf(out, cache);
    }
}
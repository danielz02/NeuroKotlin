package activations

import org.jetbrains.numkt.core.ExperimentalNumkt
import org.jetbrains.numkt.core.KtNDArray
import org.jetbrains.numkt.core.reshape
import org.jetbrains.numkt.math.clip

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

@ExperimentalNumkt
class ReLU : Activations() {
    override fun fn(input: KtNDArray<Double>): KtNDArray<Double> {
        return clip(input, 0.0, Double.POSITIVE_INFINITY)
    }

    override fun grad(input: KtNDArray<Double>): KtNDArray<Double> {
        TODO("Not yet implemented")
    }

    override fun toString(): String {
        TODO("Not yet implemented")
    }
}

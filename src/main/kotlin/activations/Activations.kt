package activations

import org.jetbrains.numkt.core.ExperimentalNumkt
import org.jetbrains.numkt.core.KtNDArray
import org.jetbrains.numkt.core.flatten
import org.jetbrains.numkt.core.reshape
import org.jetbrains.numkt.math.`**`
import org.jetbrains.numkt.math.clip
import org.jetbrains.numkt.math.exp
import org.jetbrains.numkt.math.minus
import org.jetbrains.numkt.math.plus
import org.jetbrains.numkt.math.tanh
import org.jetbrains.numkt.math.times
import org.jetbrains.numkt.math.unaryMinus
import org.jetbrains.numkt.onesLike
import org.jetbrains.numkt.zerosLike

@ExperimentalNumkt
abstract class Activations {
    operator fun invoke(input: KtNDArray<Double>): KtNDArray<Double> {
        return if (input.ndim == 1) {
            input.reshape(1, -1)
        } else {
            fn(input)
        }
    }
    abstract fun fn(x: KtNDArray<Double>): KtNDArray<Double>
    abstract fun grad(x: KtNDArray<Double>): KtNDArray<Double>
    abstract fun grad2(x: KtNDArray<Double>): KtNDArray<Double>
    abstract override fun toString(): String
}

@ExperimentalNumkt
class ReLU : Activations() {

    override fun fn(x: KtNDArray<Double>): KtNDArray<Double> {
        return clip(x, .0, Double.POSITIVE_INFINITY)
    }

    override fun grad(x: KtNDArray<Double>): KtNDArray<Double> {
        return x.apply { flatten().toList().count { it > 0 } } // Where the fuck is apply_along_axis' Kotlin version???
    }

    override fun grad2(x: KtNDArray<Double>): KtNDArray<Double> {
        return zerosLike(x)
    }

    override fun toString(): String {
        return "ReLU!"
    }
}

@ExperimentalNumkt
class Sigmoid : Activations() {
    override fun fn(x: KtNDArray<Double>): KtNDArray<Double> {
        return 1 / 1 + exp(-x)
    }

    override fun grad(x: KtNDArray<Double>): KtNDArray<Double> {
        return this.fn(x) * (onesLike(x) - this.fn(x))
    }

    override fun grad2(x: KtNDArray<Double>): KtNDArray<Double> {
        val fnX = fn(x)
        return fnX * (1 - fnX) * (1 - 2 * fnX)
    }

    override fun toString(): String {
        return "Sigmoid!"
    }
}

@ExperimentalNumkt
class Tanh : Activations() {
    override fun fn(x: KtNDArray<Double>): KtNDArray<Double> {
        return tanh(x)
    }

    override fun grad(x: KtNDArray<Double>): KtNDArray<Double> {
        return 1 - tanh(x) `**` 2
    }

    override fun grad2(x: KtNDArray<Double>): KtNDArray<Double> {
        val fnX = this.fn(x)
        return -2 * fnX * (1 - fnX `**` 2)
    }

    override fun toString(): String {
        return "Hyperbolic Tangent!"
    }
}

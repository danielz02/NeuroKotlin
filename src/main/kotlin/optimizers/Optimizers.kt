package optimizers

import org.jetbrains.numkt.*
import org.jetbrains.numkt.core.*
import org.jetbrains.numkt.linalg.Linalg
import org.jetbrains.numkt.math.*
import schedulers.Scheduler

@ExperimentalNumkt
abstract class Optimizer {
    abstract val scheduler: Scheduler?
    protected val cache: MutableMap<String, KtNDArray<Double>> = mutableMapOf()
    protected var currentStep: Int = 0
    abstract fun update(
        param: KtNDArray<Double>,
        paramGrad: KtNDArray<Double>,
        paramName: String,
        currentLoss: KtNDArray<Double>? = null
    ): KtNDArray<Double>
    fun step() { this.currentStep++ }
    operator fun invoke(
        param: KtNDArray<Double>,
        paramGrad: KtNDArray<Double>,
        paramName: String,
        currentLoss: KtNDArray<Double>?
    ): KtNDArray<Double> {
        return this.update(param, paramGrad, paramName, currentLoss)
    }
}

@ExperimentalNumkt
class SGD(
    private var lr: Double = 0.01,
    private val momentum: Double = 0.0,
    private val clipNorm: Double? = null,
    override val scheduler: Scheduler
) : Optimizer() {
    override fun update(
        param: KtNDArray<Double>,
        paramGrad: KtNDArray<Double>,
        paramName: String,
        currentLoss: KtNDArray<Double>?
    ): KtNDArray<Double> {

        this.lr = scheduler(this.currentStep)

        // if(!cache.containsKey(paramName)) cache[paramName] = zerosLike(paramGrad)

        val t = this.clipNorm ?: Double.POSITIVE_INFINITY
        val normedParamGrad = if (Linalg.norm(paramGrad) > t) {
            paramGrad * t / Linalg.norm(paramGrad)
        } else {
            paramGrad
        }
        val update = this.momentum * (cache[paramName] ?: zerosLike(paramGrad)) + this.lr * normedParamGrad
        this.cache[paramName] = update
        return param - update
    }
}

// TODO: Implement Adam. Pseudocode: https://arxiv.org/pdf/1412.6980.pdf
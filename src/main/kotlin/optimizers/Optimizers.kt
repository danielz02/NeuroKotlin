package optimizers

import org.jetbrains.numkt.*
import org.jetbrains.numkt.core.*
import org.jetbrains.numkt.linalg.Linalg
import org.jetbrains.numkt.math.*
import schedulers.Scheduler

@ExperimentalNumkt
abstract class Optimizer {
    abstract val scheduler: Scheduler?
    protected val cache: MutableMap<String, KtNDArray<Double>> = mutableMapOf<String, KtNDArray<Double>>()
    protected val currentStep: Int = 0
    abstract fun update(
        param: KtNDArray<Double>,
        paramGrad: KtNDArray<Double>,
        paramName: String,
        currentLoss: Double? = null
    ): KtNDArray<Double>
}

@ExperimentalNumkt
class SGD(
    private var lr: Double = 0.01,
    private val momentum: Double = 0.0,
    private val clipNorm: Double? = null,
    override val scheduler: Scheduler? = null
) : Optimizer() {
    override fun update(
        param: KtNDArray<Double>,
        paramGrad: KtNDArray<Double>,
        paramName: String,
        currentLoss: Double?
    ): KtNDArray<Double> {

        this.lr = (scheduler?.invoke(currentStep)) as Double

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

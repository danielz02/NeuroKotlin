package optimizers

import org.jetbrains.numkt.*
import org.jetbrains.numkt.core.*
import org.jetbrains.numkt.linalg.Linalg
import org.jetbrains.numkt.math.*
import schedulers.Scheduler
import kotlin.math.pow

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

@ExperimentalNumkt
class Adam(
    private var lr: Double = 1e-3,
    private val beta1: Double = 0.9,
    private val beta2: Double = 0.999,
    private val eps: Double = 1e-8,
    override val scheduler: Scheduler
) : Optimizer() {
    var m:KtNDArray<Double>? = null
    var v:KtNDArray<Double>? = null

    override fun update(
        param: KtNDArray<Double>,
        paramGrad: KtNDArray<Double>,
        paramName: String,
        currentLoss: KtNDArray<Double>?
    ): KtNDArray<Double> {
        currentStep++
        this.lr = scheduler(this.currentStep)

        if (m == null){
            m = zerosLike(param)
        }
        if (v == null) {
            v = zerosLike(param)
        }
        m = m!! * beta1 + (1 - beta1) * paramGrad
        v = v!! * beta2 + (1 - beta2) * paramGrad * paramGrad

        val firstBias = m!!.div(1 - beta1.pow(currentStep))
        val secondBias = v!!.div(1 - beta2.pow(currentStep))

        return lr * firstBias / (sqrt(secondBias) + eps)
    }
}


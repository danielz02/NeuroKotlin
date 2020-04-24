package optimizers

import org.jetbrains.numkt.*
import org.jetbrains.numkt.core.*
import org.jetbrains.numkt.math.*
import schedulers.Scheduler

@ExperimentalNumkt
abstract class Optimizer {
    abstract val scheduler: Scheduler?
    private val cache: Map<String, KtNDArray<Double>> = mutableMapOf<String, KtNDArray<Double>>()
    private val currentStep: Int = 0
    abstract fun update(param: KtNDArray<Double>,
                        paramGrad: KtNDArray<Double>,
                        paramName: String,
                        currentLoss: KtNDArray<Double>? = null)
}

@ExperimentalNumkt
class SGD(
    lr: Double = 0.01,
    momentum: Double = 0.0,
    clipNorm: Double? = null,
    override val scheduler: Scheduler? = null
) : Optimizer() {
    override fun update(
        param: KtNDArray<Double>,
        paramGrad: KtNDArray<Double>,
        paramName: String,
        currentLoss: KtNDArray<Double>?
    ) {
        TODO("Not yet implemented")
    }
}

package schedulers

import kotlin.math.pow

abstract class Scheduler(private val initialLr: Double = 0.01) {
    abstract fun learningRate(step: Int): Double
    abstract operator fun invoke(step: Int): Double
}

class ConstantScheduler(private val initialLr: Double = 0.01) : Scheduler(initialLr) {
    override fun learningRate(step: Int): Double {
        return this.initialLr
    }

    override fun toString(): String {
        return "Constant Scheduler: Learning Rate ${this.initialLr}"
    }

    override operator fun invoke(step: Int): Double {
        return learningRate(step)
    }
}

class ExponentialScheduler(
    private val initialLr: Double = 0.01,
    private val stageLength: Int = 500,
    private val stairCase: Boolean = false,
    private val decay: Double = 0.01
) : Scheduler(initialLr) {

    override fun learningRate(step: Int): Double {
        var currentStage = (step / this.stageLength.toDouble())
        return if (this.stairCase) {
            currentStage = kotlin.math.floor(currentStage)
            this.initialLr * this.decay.pow(currentStage)
        } else {
            this.initialLr * this.decay.pow(currentStage)
        }
    }

    override fun toString(): String {
        return "Exponential Scheduler: \n" +
                "Initial Learning Rate: ${this.initialLr}\n" +
                "Stage Length ${this.stageLength}\n" +
                "Stair Case Learning Rate: ${this.stairCase}\n" +
                "Decay Factor: ${this.decay}"
    }

    override operator fun invoke(step: Int): Double {
        return this.learningRate(step)
    }
}

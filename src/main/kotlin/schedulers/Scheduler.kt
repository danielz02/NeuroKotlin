package schedulers

import kotlin.math.pow

class ExponentialScheduler(private val initialLr: Number = 0.01,
                           private val stageLength: Int = 500,
                           private val stairCase: Boolean = false,
                           private val decay: Double = 0.01) {

    fun learningRate(step: Int): Double {
        var currentStage = (step / this.stageLength.toDouble())
        return if (this.stairCase) {
            currentStage = kotlin.math.floor(currentStage)
            this.initialLr.toDouble() * this.decay.pow(currentStage)
        } else {
            this.initialLr.toDouble() * this.decay.pow(currentStage)
        }
    }

    override fun toString(): String {
        return "Exponential Scheduler: \n" +
                "Initial Learning Rate: ${this.initialLr}\n" +
                "Stage Length ${this.stageLength}\n" +
                "Stair Case Learning Rate: ${this.stairCase}\n" +
                "Decay Factor: ${this.decay}"
    }

    operator fun invoke(step: Int) {
        this.learningRate(step)
    }
}
package losses

import org.jetbrains.numkt.arange
import org.jetbrains.numkt.core.*
import org.jetbrains.numkt.math.*
import org.jetbrains.numkt.zeros

@ExperimentalNumkt
abstract class Losses {
    abstract fun loss(yTrue: KtNDArray<Double>, yPred: KtNDArray<Double>): Double
    abstract fun grad(yTrue: KtNDArray<Double>, yPred: KtNDArray<Double>): KtNDArray<Double>
}

@ExperimentalNumkt
class L2Squared : Losses() {
    /**
     * Compute the L2 Squared loss, which is defined as sum(x_j - y_j)^2
     * @param yPred input, the prediction of model. Shape (N, C) where x[i,j] is the score for item i, class j
     * @param yTrue label, shape (N, C), y[i] is the label for x[i]
     * * @return A pair in form (loss: scalar of loss, dx: gradient respect to x)
     */
    override fun loss(yTrue: KtNDArray<Double>, yPred: KtNDArray<Double>): Double {
        val lossVec = ((yTrue - yPred) `**` 2).sum(axis = 1)
        return sum(lossVec) / (yTrue.shape[0])
    }

    override fun grad(yTrue: KtNDArray<Double>, yPred: KtNDArray<Double>): KtNDArray<Double> {
        return 2 * (yPred - yTrue) / yTrue.shape[0]
    }
}

@ExperimentalNumkt
class L1Absolute : Losses() {
    /**
     * Compute the L1 absolute loss, which is defined as sum(abs(x_j-y_j))
     * @param yTrue input, the prediction of model. Shape (N, C) where x[i,j] is the score for item i, class j
     * @param yPred label, shape (N, C), y[i] is the label for x[i]
     * @return A pair in form (loss: scalar of loss, dx: gradient respect to x)
     */
    override fun loss(yTrue: KtNDArray<Double>, yPred: KtNDArray<Double>): Double {
        val lossVec = sum(absolute(yPred - yTrue), axis = 1)
        return sum(lossVec) / yPred.shape[0]
    }

    override fun grad(yTrue: KtNDArray<Double>, yPred: KtNDArray<Double>): KtNDArray<Double> {
        return absolute(yPred - yTrue) / yPred.shape[0]
    }
}

@ExperimentalNumkt
class Hinge : Losses() {
    /**
     * Calculate Hinge loss and gradient using SVM classification
     * Loss for item i, class j will be 0 if x[i,j] is smaller than margin(1)
     * @param yPred input, the prediction of model. Shape (N, C) where x[i,j] is the score for item i, class j
     * @param yTrue label, shape (N,), y[i] is the label for x[i] and 0 <= y[i] < C
     * @return A pair in form (loss: scalar of loss, dx: gradient respect to x)
     */
    override fun loss(yTrue: KtNDArray<Double>, yPred: KtNDArray<Double>): Double {
        val n = yPred.shape[0]
        val correct = yPred[arange(n), yTrue].reshape(-1, 1)
        val margin = clip(yPred - correct, aMin = 0.0, aMax = 1e20)
        margin[arange(n), yTrue] = margin[arange(n), yTrue].minus(margin[arange(n), yTrue])
        var loss = sum(margin)
        loss /= n
        return loss
    }

    override fun grad(yTrue: KtNDArray<Double>, yPred: KtNDArray<Double>): KtNDArray<Double> {
        val n = yPred.shape[0]
        val correct = yPred[arange(n), yTrue].reshape(-1, 1)
        val margin = clip(yPred - correct, aMin = 0.0, aMax = 1e20)
        val dx = (zeros<Double>(yPred.shape[0], yPred.shape[1])).apply { margin.nonZero() }
        dx[arange(n), yPred] = - dx.sum(1)
        return dx
    }
}

@ExperimentalNumkt
class Softmax : Losses() {
    override fun loss(yTrue: KtNDArray<Double>, yPred: KtNDArray<Double>): Double {
        val shift = yPred - yPred.max(1)
        val z = sum(exp(shift), axis = 1)
        val logProbs = shift - log(z)
        val n = yPred.shape[0]
        return - sum(logProbs[arange(n), yTrue]) / n
    }

    override fun grad(yTrue: KtNDArray<Double>, yPred: KtNDArray<Double>): KtNDArray<Double> {
        val shift = yPred - yPred.max(1)
        val z = sum(exp(shift), axis = 1)
        val logProbs = shift - log(z)
        val probs = exp(logProbs)
        val n = yPred.shape[0]
        val dx = probs.copy()
        dx[arange(n), yTrue] -= 1
        dx /= n
        return dx
    }
}

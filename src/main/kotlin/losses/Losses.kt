package losses

import org.jetbrains.numkt.arange
import org.jetbrains.numkt.core.*
import org.jetbrains.numkt.math.*
import org.jetbrains.numkt.zeros

@ExperimentalNumkt
    /**
     * Calculate Hinge loss and gradient using SVM classification
     * Loss for item i, class j will be 0 if x[i,j] is smaller than margin(1)
     * @param x input, the prediction of model. Shape (N, C) where x[i,j] is the score for item i, class j
     * @param y label, shape (N,), y[i] is the label for x[i] and 0 <= y[i] < C
     * @return A pair in form (loss: scalar of loss, dx: gradient respect to x)
     */
fun Hinge(x: KtNDArray<Float>, y: KtNDArray<Int>): Pair<Float, KtNDArray<Float>> {

// Naive implementation with looping
//    var loss = 0.0;
//    var dx = zeros<Float>(N, x.shape[1])
//    for (i in 0..N) {
//        val correct = x[y[i]];
//        for (j in 0..x.shape[1]) {
//            if (y[i] as Int == j) {
//                continue;
//            }
//            var margin = x[i,j] as Double - correct as Double + 1.0;
//            if (margin > 0) {
//                loss += margin;
//                dx[0..N, j] = dx[0..N, j].plus(x[i]);
//                dx[0..N, y[i]] = dx[0..N, j].minus(x[i]);
//            }
//        }
//    }
//    dx = dx.div(N);
//    loss /= N;
//    return Pair(loss, dx);

    val N = x.shape[0]
    val correct = x[arange(N), y].reshape(-1, 1)
    val margin = clip(x - correct, aMin = 0.0f, aMax = 1e20f)
    margin[arange(N), y] = margin[arange(N), y].minus(margin[arange(N), y])
    var loss = sum(margin)
    loss /= N
    val dx = (zeros<Float>(x.shape[0], x.shape[1])).apply { margin.nonZero() }
    dx[arange(N), y] = -dx.sum(1)
    return Pair(loss, dx)
}

@ExperimentalNumkt
    /**
     * Compute the softmax loss, which is defined as -log(e^f_yi / sum(e^f_j)) where f's are normed
     * @param x input, the prediction of model. Shape (N, C) where x[i,j] is the score for item i, class j
     * @param y label, shape (N,), y[i] is the label for x[i] and 0 <= y[i] < C
     * @return A pair in form (loss: scalar of loss, dx: gradient respect to x)
     */
fun Softmax(x: KtNDArray<Float>, y: KtNDArray<Int>): Pair<Float, KtNDArray<Float>> {
    val shift = x - x.max(1)
    val z = sum(exp(shift), axis = 1)
    val logProbs = shift - log(z)
    val probs = exp(logProbs)
    val N = x.shape[0]
    val loss = -sum(logProbs[arange(N), y]) / N
    val dx = probs.copy() as KtNDArray<Float>
    dx[arange(N), y] -= 1
    dx /= N
    return Pair(loss.toFloat(), dx)
}

@ExperimentalNumkt
    /**
     * Compute the L2 Squared loss, which is defined as sum(x_j - y_j)^2
     * @param x input, the prediction of model. Shape (N, C) where x[i,j] is the score for item i, class j
     * @param y label, shape (N, C), y[i] is the label for x[i]
     * * @return A pair in form (loss: scalar of loss, dx: gradient respect to x)
     */
fun L2Squared(x: KtNDArray<Float>, y: KtNDArray<Float>): Pair<Float, KtNDArray<Float>> {
    val lossVec = sum(power(x - y, 2), axis = 1)
    val loss = sum(lossVec) / x.shape[0]
    val dx = 2 * (x - y) / x.shape[0]
    return Pair(loss, dx)
}

@ExperimentalNumkt
    /**
     * Compute the L1 absolute loss, which is defined as sum(abs(x_j-y_j))
     * @param x input, the prediction of model. Shape (N, C) where x[i,j] is the score for item i, class j
     * @param y label, shape (N, C), y[i] is the label for x[i]
     * @return A pair in form (loss: scalar of loss, dx: gradient respect to x)
     */
fun L1Absolute(x: KtNDArray<Float>, y: KtNDArray<Float>): Pair<Float, KtNDArray<Float>> {
    val lossVec = sum(absolute(x - y), axis = 1)
    val loss = sum(lossVec) / x.shape[0]
    val dx = absolute(x - y) / x.shape[0]
    return Pair(loss, dx)
}

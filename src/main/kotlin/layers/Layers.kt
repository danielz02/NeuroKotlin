package layers

import optimizers.Optimizer
import activations.Activations
import org.jetbrains.numkt.zeros
import org.jetbrains.numkt.core.*
import org.jetbrains.numkt.math.*
import org.jetbrains.numkt.zerosLike
import org.jetbrains.numkt.random.Random.Companion.randn

@ExperimentalNumkt
abstract class Layers(
    /** An instance of Optimizers class. */
    private val optimizer: Optimizer,
    /** An instance of Activations class. If null, no activation function is applied */
    private val activations: Activations? = null,
    /** Whether to freeze the gradient of this layer. */
    private var trainable: Boolean = true,
    /** Cache of the variable in the forward pass. */
    protected val x: MutableList<KtNDArray<Double>> = mutableListOf(),
    /** Cache of the gradients. */
    private var gradients: MutableMap<String, KtNDArray<Double>> = mutableMapOf(),
    /** The parameters to be optimized. */
    private val parameters: MutableMap<String, KtNDArray<Double>> = mutableMapOf(),
    /** Intermediate variables. */
    private val derivedVariables: MutableMap<String, MutableList<KtNDArray<Double>>> = mutableMapOf()) {

    /** For children to override. */
    abstract fun forward(x: KtNDArray<Double>, saveDerived: Boolean = true): KtNDArray<Double>

    /** For children to override. */
    abstract fun backward(dLdy: MutableList<KtNDArray<Double>>, saveGradient: Boolean = true): MutableList<KtNDArray<Double>>

    /**
     * @return String: describes this layer, Int: para count
     */
    abstract fun info():Pair<String, Int>;

    fun freeze() {
        this.trainable = false
    }

    fun unfreeze() {
        this.trainable = true
    }

    /** Flush the toilet. */
    fun flush() {
        assert(this.trainable) { "Frozen Layer! Can't flush the gradient!" }
        this.x.clear()
        this.derivedVariables.onEach { it.value.clear() }
        this.gradients = this.gradients.onEach { it.key to zerosLike(it.value) } // Can't modify in place?
    }

    /** Let our blackbox do its job. */
    fun update(currentLoss: KtNDArray<Double>? = null) {
        assert(this.trainable) { "Frozen Layer! Can't update the parameter!" }
        this.optimizer.step()
        this.gradients.onEach {
            this.parameters[it.key] = this.optimizer(this.parameters[it.key]!!, it.value, it.key, currentLoss)
        }
        this.flush()
    }

}

@ExperimentalNumkt
class FullyConnected(
    private val nOut: Int,
    private val optimizer: Optimizer,
    private val trainable: Boolean = true,
    private val activations: Activations? = null,
    private val derivedVariables: MutableMap<String, MutableList<KtNDArray<Double>>> = mutableMapOf("Z" to mutableListOf()),
    private val parameters: MutableMap<String, KtNDArray<Double>> = mutableMapOf() ,
    private var gradients: MutableMap<String, KtNDArray<Double>> = mutableMapOf(),
    /** Determined by the input vector. */
    private var nIn: Int? = null,
    /** TODO: Remove this later and let Elvis do its job. */
    private var isInitialized: Boolean = false
) : Layers(optimizer, activations, trainable) {

    private fun initVariables() {
        this.parameters["W"] = randn(this.nIn!!, this.nOut)
        this.parameters["b"] = zeros(1, this.nOut)
        this.gradients = mutableMapOf("W" to zerosLike(this.parameters["W"]!!), "b" to zerosLike(this.parameters["b"]!!))
        this.isInitialized = true
        // TODO(Use Kotlin style to initialize weights... Variables like isInitialized shouldn't exist)
    }

    override fun forward(x: KtNDArray<Double>, saveDerived: Boolean): KtNDArray<Double> {
        val (y, z) = forward(x)
        if (saveDerived) {
            this.x.add(x)
            this.derivedVariables["Z"]!!.add(z)
        }
        return y
    }

    private fun forward(x: KtNDArray<Double>): Pair<KtNDArray<Double>, KtNDArray<Double>> {
        if (!isInitialized) {
            this.nIn = x.shape[1]
            initVariables()
        }
        val w = this.parameters["W"]!!
        val b = this.parameters["b"]!!
        val z = x `@` w + b
        val y = this.activations?.invoke(z) ?: z
        return Pair(y, z)
    }

    override fun backward(dLdy: MutableList<KtNDArray<Double>>, saveGradient: Boolean): MutableList<KtNDArray<Double>> {
        assert(this.trainable) { "Layer Frozen!" }
        val dX = mutableListOf<KtNDArray<Double>>()
        val x = this.x
        (dLdy zip x).map {
            val (dx, dw, db) = (this.backward(it.first, it.second))
            dX.add(dx)

            if (saveGradient) {
                this.gradients["W"]!! += dw
                this.gradients["b"]!! += db
            }
        }

        return if (x.size == 1) mutableListOf(dX[0]) else dX
    }

    /**
     * The private helper of backward
     * Formula Reference http://cs231n.stanford.edu/handouts/linear-backprop.pdf
     * @param dLdy the gradient of the loss function w.r.t. y
     */
    private fun backward(dLdy: KtNDArray<Double>,
        x: KtNDArray<Double>): Triple<KtNDArray<Double>, KtNDArray<Double>, KtNDArray<Double>> {

        val w = this.parameters["W"]!!
        val b = this.parameters["b"]!!

        val z = x `@` w + b
        val dz = dLdy * (this.activations?.invoke(z) ?: z)

        val dx = dz `@` w.t
        val dw = x.t `@` dz
        val dB = dz.sum(axis = 0)
        return Triple(dx, dw, dB)
    }

    override fun toString(): String {
        val numParam = nIn!! * nOut + nOut
        return "Fully Connected, Weight ($nIn, $nOut), Bias ($nOut), Param count $numParam\n" +
                "Activation is $activations"
    }

    override fun info(): Pair<String, Int> {
        val numParam = nIn!! * nOut + nOut
        return Pair(toString(), numParam)
    }
}


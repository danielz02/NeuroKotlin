package model

import layers.FullyConnected
import layers.Layers
import losses.Losses
import org.jetbrains.numkt.core.ExperimentalNumkt
import org.jetbrains.numkt.core.KtNDArray
import org.jetbrains.numkt.core.copy

@ExperimentalNumkt
abstract class Model {
    /** Number of layers of the model*/
    private var numLayers: Int = 0
    /** A list that stores all layers */
    private var layers: MutableList<Layers> = mutableListOf()
    /** Loss function as defined in losses.Losses */
    private var loss: Losses? = null
    /** A checker that enforces compile before train */
    private var compiled = false

    /** Add layer to layers */
    abstract fun addLayer(layer: Layers)

    /** Add Loss Function */
    abstract fun addLoss(loss: Losses)

    /** Compile model, sanity check, initialize necessary variables */
    abstract fun compile()

    /**
     * Main training loop, go through forward and backward path
     * @param x: samples in shape [N, S1..SN] where each sample has shape (S1...SN)
     * @param y: labels in shape [N,..], depends on loss functions
     * @param epoch: number of epoch use
     * @param batchSize: Not helpful now
     * @param verbose: whether to print out loss at each epoch
     */
    abstract fun train(
        x: KtNDArray<Double>,
        y: KtNDArray<Double>,
        epoch: Int,
        batchSize: Int = 32,
        verbose: Boolean = true
    )

    /** Do a prediction after training, will produce terrible result if not trained */
    abstract fun predict(x: KtNDArray<Double>): KtNDArray<Double>

    /** Do a validation after training, will produce terrible result if not trained */
    abstract fun validation(x: KtNDArray<Double>, y: KtNDArray<Double>): Double

    /** Printout model description */
    abstract fun describe()
}

@ExperimentalNumkt
class FCN : Model() {

    private var numLayers: Int = 0
    private var layers: MutableList<FullyConnected> = mutableListOf()
    private var lossFun: Losses? = null
    private var compiled = false

    override fun addLayer(layer: Layers) {
        assert(layer is FullyConnected) { "Not a Fully Connected Layer" }
        numLayers += 1
        layers.add(layer as FullyConnected)
    }

    override fun addLoss(loss: Losses) {
        lossFun = loss
    }

    // Not really helpful for fnc since all initialization are done in forward path
    // Ideally we should initialize
    override fun compile() {
        assert(layers.isNotEmpty()) { "No Layers to compile" }
        assert(lossFun != null) { "No Loss Found" }
        compiled = true
    }

    override fun train(
        x: KtNDArray<Double>,
        y: KtNDArray<Double>,
        epoch: Int,
        batchSize: Int,
        verbose: Boolean
    ) {
        //!TODO BatchSize does nothing right now
        assert(compiled) { "Please Compile First" }
        assert(x.shape[0] == y.shape[0]) { "Training Length Does Not Match Label" }

        print("Start training with " + x.shape[0] + " samples")
        var trainX = x.copy()
        var loss = 0.0
        for (i in 0..epoch) {
            if (verbose) {
                print("At $i/$epoch loss is $loss")
            }
            for (layer in layers) {
                trainX = layer.forward(trainX)
            }

            // Wait why is it (label, prediction)
            loss = lossFun!!.loss(y, trainX)
            val grad = lossFun!!.grad(y, trainX)
            //!TODO backprop api没看懂
        }
        print("Done")
    }

    override fun predict(x: KtNDArray<Double>): KtNDArray<Double> {
        var pred = x.copy()
        for (layer in layers) {
            pred = layer.forward(pred, false)
        }
        return pred
    }

    override fun validation(x: KtNDArray<Double>, y: KtNDArray<Double>): Double {
        var validation = x.copy()
        for (layer in layers) {
            validation = layer.forward(validation, false)
        }
        return lossFun!!.loss(y, validation)
    }

    override fun describe() {
        var totalParam = 0
        for (layer in layers) {
            print("----------------------------")
            val info = layer.info();
            print(info.first)
            totalParam += info.second
        }
        print("----------------------------")
        print("Total param is$totalParam")
    }
}
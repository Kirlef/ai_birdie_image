package dev.smitjivani.aibirdieimage.tflite

import android.app.Activity
import org.tensorflow.lite.support.common.TensorOperator
import org.tensorflow.lite.support.common.ops.NormalizeOp

class ClassifierEfficientNetB4(activity: Activity, device: Device, numThreads: Int) : Classifier(activity, device, numThreads) {

    /** Float MobileNet requires additional normalization of the used input.  */
    private val IMAGE_MEAN = 127.5f

    private val IMAGE_STD = 127.5f

    /**
     * Float model does not need dequantization in the post-processing. Setting mean and std as 0.0f
     * and 1.0f, repectively, to bypass the normalization.
     */
    private val PROBABILITY_MEAN = 0.0f

    private val PROBABILITY_STD = 1.0f


    override fun getModelPath(): String? {
        return "model-ENB4.tflite"
    }

    override fun getLabelPath(): String? {
        return "labels-ENB4.txt"
    }

    override fun getPreprocessNormalizeOp(): TensorOperator? {
        return NormalizeOp(IMAGE_MEAN, IMAGE_STD)
    }

    override fun getPostprocessNormalizeOp(): TensorOperator? {
        return NormalizeOp(PROBABILITY_MEAN, PROBABILITY_STD)
    }
}
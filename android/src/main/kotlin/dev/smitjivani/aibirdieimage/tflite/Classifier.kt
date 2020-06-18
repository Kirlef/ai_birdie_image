package dev.smitjivani.aibirdieimage.tflite

import android.app.Activity
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.RectF
import android.os.SystemClock
import android.os.Trace
import android.view.Surface
import dev.smitjivani.aibirdieimage.env.Logger
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.TensorOperator
import org.tensorflow.lite.support.common.TensorProcessor
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.ResizeOp.ResizeMethod
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp
import org.tensorflow.lite.support.label.TensorLabel
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.IOException
import java.nio.MappedByteBuffer
import java.util.*
import kotlin.Comparator
import kotlin.collections.ArrayList

abstract class Classifier {
    private val LOGGER: Logger = Logger()

    private lateinit var activity: Activity

    /** The runtime device type used for executing classification.  */
    enum class Device {
        CPU, GPU
    }

    /** Number of results to show in the UI.  */
    private val MAX_RESULTS = 20

    /** The loaded TensorFlow Lite model.  */
    private var tfliteModel: MappedByteBuffer? = null

    /** Image size along the x axis.  */
    private var imageSizeX = 0

    /** Image size along the y axis.  */
    private var imageSizeY = 0

    /** Optional GPU delegate for acceleration.  */
    private var gpuDelegate: GpuDelegate? = null

    /** An instance of the driver class to run model inference with Tensorflow Lite.  */
    protected var tflite: Interpreter? = null

    /** Options for configuring the Interpreter.  */
    private val tfliteOptions = Interpreter.Options()

    /** Labels corresponding to the output of the vision model.  */
    private var labels: List<String>? = null

    /** Input image TensorBuffer.  */
    private var inputImageBuffer: TensorImage? = null

    /** Output probability TensorBuffer.  */
    private var outputProbabilityBuffer: TensorBuffer? = null

    /** Processer to apply post processing of the output probability.  */
    private var probabilityProcessor: TensorProcessor? = null

    /**
     * Creates a classifier with the provided configuration.
     *
     * @param activity The current Activity.
     * @param device The device to use for classification.
     * @param numThreads The number of threads to use for classification.
     * @return A classifier with the desired configuration.
     */
    @Throws(IOException::class)
    open fun create(activity: Activity?, device: Device?, numThreads: Int): Classifier? {
        // TODO: Fix (NS)
        return ClassifierEfficientNetB4(activity!!, device!!, numThreads)
    }

    /** An immutable result returned by a Classifier describing what was recognized.  */
    class Recognition(
            /**
             * A unique identifier for what has been recognized. Specific to the class, not the instance of
             * the object.
             */
            val id: String?,
            /** Display name for the recognition.  */
            val title: String?,
            /**
             * A sortable score for how good the recognition is relative to others. Higher should be better.
             */
            val confidence: Float?,
            /** Optional location within the source image for the location of the recognized object.  */
            private var location: RectF?) {

        fun getLocation(): RectF {
            return RectF(location)
        }

        fun setLocation(location: RectF?) {
            this.location = location
        }

        override fun toString(): String {
            var resultString = ""
            if (id != null) {
                resultString += "[$id] "
            }
            if (title != null) {
                resultString += "$title "
            }
            if (confidence != null) {
                resultString += String.format("(%.1f%%) ", confidence * 100.0f)
            }
            if (location != null) {
                resultString += location.toString() + " "
            }
            return resultString.trim { it <= ' ' }
        }

    }

    protected open fun getScreenOrientation(): Int {
        return when (activity.windowManager.defaultDisplay.rotation) {
            Surface.ROTATION_270 -> 270
            Surface.ROTATION_180 -> 180
            Surface.ROTATION_90 -> 90
            else -> 0
        }
    }

    /** Initializes a `Classifier`.  */
    @Throws(IOException::class)
    protected constructor(activity: Activity, device: Device?, numThreads: Int) {
        this.activity = activity
        LOGGER.e(getModelPath())
        tfliteModel = FileUtil.loadMappedFile(activity, getModelPath()!!)
        when (device) {
            Device.GPU -> {
                gpuDelegate = GpuDelegate()
                tfliteOptions.addDelegate(gpuDelegate)
            }
            Device.CPU -> {
            }
        }
        tfliteOptions.setNumThreads(numThreads)
        tflite = Interpreter(tfliteModel!!, tfliteOptions)

        // Loads labels out from the label file.

        labels = FileUtil.loadLabels(activity, getLabelPath()!!)

        // Reads type and shape of input and output tensors, respectively.
        val imageTensorIndex = 0
        val imageShape = tflite!!.getInputTensor(imageTensorIndex).shape() // {1, height, width, 3}
        imageSizeY = imageShape[1]
        imageSizeX = imageShape[2]
        val imageDataType = tflite!!.getInputTensor(imageTensorIndex).dataType()
        val probabilityTensorIndex = 0
        val probabilityShape = tflite!!.getOutputTensor(probabilityTensorIndex).shape() // {1, NUM_CLASSES}
        val probabilityDataType = tflite!!.getOutputTensor(probabilityTensorIndex).dataType()

        // Creates the input tensor.
        inputImageBuffer = TensorImage(imageDataType)

        // Creates the output tensor and its processor.
        outputProbabilityBuffer = TensorBuffer.createFixedSize(probabilityShape, probabilityDataType)

        // Creates the post processor for the output probability.
        probabilityProcessor = TensorProcessor.Builder().add(getPostprocessNormalizeOp()).build()
        LOGGER.d("Created a Tensorflow Lite Image Classifier.")
    }


    /** Runs inference and returns the classification results.  */
    open fun recognizeImage(path: String): Map<String, List<Any>> {
        // Logs this method so that it can be analyzed with systrace.
        Trace.beginSection("recognizeImage")
        Trace.beginSection("loadImage")
        val startTimeForLoadImage = SystemClock.uptimeMillis()
        val bitmap = BitmapFactory.decodeFile(path)
        inputImageBuffer = loadImage(bitmap)
        val endTimeForLoadImage = SystemClock.uptimeMillis()
        Trace.endSection()
        LOGGER.v("Timecost to load the image: " + (endTimeForLoadImage - startTimeForLoadImage))

        // Runs the inference call.
        Trace.beginSection("runInference")
        val startTimeForReference = SystemClock.uptimeMillis()
        tflite!!.run(inputImageBuffer!!.buffer, outputProbabilityBuffer!!.buffer.rewind())
        val endTimeForReference = SystemClock.uptimeMillis()
        Trace.endSection()
        LOGGER.v("Timecost to run model inference: " + (endTimeForReference - startTimeForReference))

        // Gets the map of label and probability.
        val labeledProbability = TensorLabel(labels!!, probabilityProcessor!!.process(outputProbabilityBuffer))
                .mapWithFloatValue
        Trace.endSection()

        // Gets top-k results.
        return getTopKProbability(labeledProbability)
    }

    /** Closes the interpreter and model to release resources.  */
    open fun close() {
        if (tflite != null) {
            tflite!!.close()
            tflite = null
        }
        if (gpuDelegate != null) {
            gpuDelegate!!.close()
            gpuDelegate = null
        }
        tfliteModel = null
    }

    /** Get the image size along the x axis.  */
    open fun getImageSizeX(): Int {
        return imageSizeX
    }

    /** Get the image size along the y axis.  */
    open fun getImageSizeY(): Int {
        return imageSizeY
    }

    /** Loads input image, and applies preprocessing.  */
    private fun loadImage(bitmap: Bitmap): TensorImage? {
        // Loads bitmap into a TensorImage.
        inputImageBuffer!!.load(bitmap)

        // Creates processor for the TensorImage.
        val cropSize = Math.min(bitmap.width, bitmap.height)
        // TODO(b/143564309): Fuse ops inside ImageProcessor.
        // TODO: Hardcoded size passed. Fix
        val imageProcessor = ImageProcessor.Builder()
                .add(ResizeWithCropOrPadOp(512, 512))
                .add(ResizeOp(imageSizeX, imageSizeY, ResizeMethod.NEAREST_NEIGHBOR))
                .add(getPreprocessNormalizeOp())
                .build()
        return imageProcessor.process(inputImageBuffer)
    }

    /** Gets the top-k results.  */
    private fun getTopKProbability(probabilities: MutableMap<String, Float>): Map<String, List<Any>> {
        // Find the best classifications.
        val pq = PriorityQueue(
                MAX_RESULTS,
                Comparator<Pair<Int, Float>> { lhs, rhs -> // Intentionally reversed to put high confidence at the head of the queue.
                    java.lang.Float.compare(rhs.second, lhs.second)
                })
        for (probability in probabilities) {
            pq.add(Pair(probability.key.toInt(), probability.value))
        }

        val recognitions = ArrayList<Pair<Int, Float>?>()
        val recognitionsSize = Math.min(pq.size, MAX_RESULTS)
        for (i in 0 until recognitionsSize) {
            recognitions.add(pq.poll())
        }

        val ids = recognitions.map { element -> element!!.first }
        val prob = recognitions.map { element -> element!!.second }

        return mapOf("id" to ids, "probabilities" to prob)
    }

    /** Gets the name of the model file stored in Assets.  */
    protected abstract fun getModelPath(): String?

    /** Gets the name of the label file stored in Assets.  */
    protected abstract fun getLabelPath(): String?

    /** Gets the TensorOperator to nomalize the input image in preprocessing.  */
    protected abstract fun getPreprocessNormalizeOp(): TensorOperator?

    /**
     * Gets the TensorOperator to dequantize the output probability in post processing.
     *
     *
     * For quantized model, we need de-quantize the prediction with NormalizeOp (as they are all
     * essentially linear transformation). For float model, de-quantize is not required. But to
     * uniform the API, de-quantize is added to float model too. Mean and std are set to 0.0f and
     * 1.0f, respectively.
     */
    protected abstract fun getPostprocessNormalizeOp(): TensorOperator?


}
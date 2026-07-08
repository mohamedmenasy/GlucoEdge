package com.glucoedge.app

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.glucoedge.app.inference.ModelFile
import com.glucoedge.app.inference.QuantizationMath
import com.glucoedge.app.inference.TrendClassifier
import java.security.MessageDigest
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class GoldenParityTest {
    private val appContext = InstrumentationRegistry.getInstrumentation().targetContext
    private val testContext = InstrumentationRegistry.getInstrumentation().context

    private fun goldens(): JSONObject =
        JSONObject(testContext.assets.open("golden_vectors.json").bufferedReader().readText())

    private fun sha256OfAsset(name: String): String {
        val bytes = appContext.assets.open(name).readBytes()
        return MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
    }

    @Test
    fun bundledModelsMatchGoldenHashes() {
        val expected = goldens().getJSONObject("model_sha256")
        for (model in ModelFile.entries) {
            assertEquals(
                "asset ${model.assetName} drifted from the goldens - regenerate both together",
                expected.getString(model.assetName),
                sha256OfAsset(model.assetName),
            )
        }
    }

    @Test
    fun floatModelMatchesPythonLogitsAndClasses() {
        val vectors = goldens().getJSONArray("vectors")
        val classifier = TrendClassifier.create(appContext, ModelFile.FLOAT)
        try {
            for (i in 0 until vectors.length()) {
                val v = vectors.getJSONObject(i)
                val window = v.getJSONArray("window").let { arr ->
                    FloatArray(arr.length()) { arr.getDouble(it).toFloat() }
                }
                val prediction = classifier.classify(window)
                assertEquals("vector $i class", v.getInt("float_class"), prediction.classIndex)

                // Rung 1 held (writeInt8/readInt8 are real methods on TensorBuffer), so we can
                // also assert on raw logits, not just the softmax-then-argmax class - this
                // catches logit drift that a class-only check could mask.
                val expectedLogits = v.getJSONArray("float_logits")
                assertEquals("vector $i logit count", expectedLogits.length(), prediction.logits.size)
                for (j in 0 until expectedLogits.length()) {
                    assertEquals(
                        "vector $i logit $j",
                        expectedLogits.getDouble(j),
                        prediction.logits[j].toDouble(),
                        1e-5,
                    )
                }
            }
        } finally {
            classifier.close()
        }
    }

    @Test
    fun int8ModelMatchesPythonClasses() {
        val goldens = goldens()
        val vectors = goldens.getJSONArray("vectors")
        val outputQuant = goldens.getJSONObject("int8_output_quant")
        val scale = outputQuant.getDouble("scale").toFloat()
        val zeroPoint = outputQuant.getInt("zero_point")
        val classifier = TrendClassifier.create(appContext, ModelFile.INT8)
        try {
            for (i in 0 until vectors.length()) {
                val v = vectors.getJSONObject(i)
                val window = v.getJSONArray("window").let { arr ->
                    FloatArray(arr.length()) { arr.getDouble(it).toFloat() }
                }
                val prediction = classifier.classify(window)
                assertEquals("vector $i int8 class", v.getInt("int8_class"), prediction.classIndex)

                // Exact parity, not just argmax: dequantize the golden raw int8 output in
                // Kotlin via the same QuantizationMath the app uses, and require bit-for-bit
                // agreement with the on-device logits. Both sides compute (q - zeroPoint) *
                // scale in fp32 from the same integer inputs, so exact equality is achievable -
                // unlike the float model, there is no accumulated floating-point arithmetic
                // difference to tolerate.
                val rawOutput = v.getJSONArray("int8_raw_output")
                val rawBytes = ByteArray(rawOutput.length()) { rawOutput.getInt(it).toByte() }
                val expectedLogits = QuantizationMath.dequantizeInt8(rawBytes, scale, zeroPoint)
                assertEquals("vector $i int8 logit count", expectedLogits.size, prediction.logits.size)
                for (j in expectedLogits.indices) {
                    assertEquals(
                        "vector $i int8 logit $j",
                        expectedLogits[j],
                        prediction.logits[j],
                        0.0f,
                    )
                }
            }
        } finally {
            classifier.close()
        }
    }
}

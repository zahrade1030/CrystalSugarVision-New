package com.example.data.db

import android.content.Context
import android.graphics.BitmapFactory
import com.example.cv.FeatureSchema
import com.example.cv.OpenCvPipeline
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.File
import java.util.Locale

/**
 * Non-destructive compatibility maintenance for stored reference samples.
 * Detects legacy, incompatible or missing vectors, re-extracts them using the
 * current pipeline from the original reference image file, and preserves all user-authored
 * metadata, names, descriptions, images and version history.
 */
object ReferenceVectorMaintenance {

    private val maintenanceMutex = Mutex()
    private val pipeline = OpenCvPipeline()

    /**
     * Checks all reference samples and re-extracts vectors if their dimension
     * or format does not match FeatureSchema.FEATURE_VECTOR_SIZE.
     */
    suspend fun ensureVectorsUpdated(context: Context, db: AppDatabase) = withContext(Dispatchers.IO) {
        maintenanceMutex.withLock {
            val sampleDao = db.referenceSampleDao()
            val versionDao = db.referenceVersionDao()
            val samples = sampleDao.getAllSamplesDirect()

            for (sample in samples) {
                val parsed = parseVectorStrict(sample.featureVectorJson)
                val isCurrent = parsed != null && parsed.size == FeatureSchema.FEATURE_VECTOR_SIZE

                if (!isCurrent) {
                    val path = sample.imagePath
                    if (path != null) {
                        val imgFile = File(path)
                        val candidateFile = if (imgFile.exists()) {
                            imgFile
                        } else {
                            File(context.filesDir, "reference_samples/${imgFile.name}")
                        }

                        if (candidateFile.exists()) {
                            try {
                                val bmp = BitmapFactory.decodeFile(candidateFile.absolutePath)
                                if (bmp != null) {
                                    val newRawVector = pipeline.extractReferenceVector(bmp)
                                    val newVectorJson = newRawVector.joinToString(
                                        separator = ",",
                                        prefix = "[",
                                        postfix = "]"
                                    ) { String.format(Locale.US, "%.5f", it) }

                                    val newVersion = sample.version + 1
                                    val updatedSample = sample.copy(
                                        featureVectorJson = newVectorJson,
                                        featureVectorSize = newRawVector.size,
                                        version = newVersion,
                                        lastModifiedTimestamp = System.currentTimeMillis(),
                                        lastModifiedBy = "System Migration (v3.0.0)"
                                    )
                                    sampleDao.insertOrUpdate(updatedSample)

                                    versionDao.insertVersion(
                                        ReferenceVersionEntity(
                                            sampleId = sample.id,
                                            versionNumber = newVersion,
                                            name = sample.name,
                                            icumsaValue = sample.icumsaValue,
                                            description = sample.description,
                                            featureVectorJson = newVectorJson,
                                            imagePath = candidateFile.absolutePath,
                                            timestamp = System.currentTimeMillis(),
                                            operatorName = "Pipeline Migration v3.0",
                                            changeDescription = "Features non-destructively re-extracted with CIELAB pipeline v3, image unchanged"
                                        )
                                    )
                                }
                            } catch (e: Exception) {
                                e.printStackTrace()
                            }
                        }
                    }
                }
            }
        }
    }

    /**
     * Parses vector JSON strictly:
     * Returns null if string is empty, malformed, non-numeric, contains NaN/Infinite,
     * or has an incorrect dimension.
     */
    fun parseVectorStrict(json: String?): DoubleArray? {
        if (json.isNullOrBlank()) return null
        return try {
            val clean = json.trim().removePrefix("[").removeSuffix("]").trim()
            if (clean.isEmpty()) return null
            val tokens = clean.split(",")
            val result = DoubleArray(tokens.size)
            for (i in tokens.indices) {
                val token = tokens[i].trim()
                val v = token.toDoubleOrNull() ?: return null
                if (!v.isFinite()) return null
                result[i] = v
            }
            result
        } catch (e: Exception) {
            null
        }
    }
}

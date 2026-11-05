package com.example.cv

import android.graphics.Bitmap

/**
 * Image Quality Validation Result
 */
data class ImageQualityResult(
    val isPassed: Boolean,
    val blurScore: Double,          // Laplacian variance (higher = sharper)
    val brightnessScore: Double,    // Mean luminance (0 - 255)
    val contrastScore: Double,      // Std dev of luminance
    val centerAlignmentScore: Double,// 0.0 - 1.0 (closeness to center)
    val overallScore: Double,       // 0 - 100%
    val rejectionReason: String? = null,
    val isClassifiable: Boolean = true
)

/**
 * Feature vector representation for Single ROI
 */
data class FeatureVector(
    val regionIndex: Int,
    val regionName: String,
    val values: DoubleArray
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false
        other as FeatureVector
        if (regionIndex != other.regionIndex) return false
        if (regionName != other.regionName) return false
        if (!values.contentEquals(other.values)) return false
        return true
    }

    override fun hashCode(): Int {
        var result = regionIndex
        result = 31 * result + regionName.hashCode()
        result = 31 * result + values.contentHashCode()
        return result
    }
}

/**
 * Extracted color and statistical feature set for Single ROI (Whole Image)
 */
data class CompleteFeatureSet(
    val regions: List<FeatureVector>,
    val concatenatedVector: DoubleArray,
    val extractionTimeMs: Long,
    val colorHistogram: FloatArray,
    val rgbMeans: Triple<Double, Double, Double>,
    val hsvMeans: Triple<Double, Double, Double>,
    val labMeans: Triple<Double, Double, Double>
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false
        other as CompleteFeatureSet
        return concatenatedVector.contentEquals(other.concatenatedVector)
    }

    override fun hashCode(): Int {
        return concatenatedVector.contentHashCode()
    }
}

/**
 * Individual metric match result against a reference sample
 * Using Euclidean Distance, Cosine Similarity, and Bhattacharyya Histogram Distance
 */
data class ReferenceMatch(
    val sampleId: Int,
    val sampleName: String,
    val icumsaValue: Int,
    val euclideanDistance: Double,
    val euclideanSimilarityPct: Double,
    val cosineSimilarity: Double,
    val cosineSimilarityPct: Double,
    val bhattacharyyaDistance: Double = 0.0,
    val bhattacharyyaSimilarityPct: Double = 0.0,
    val finalSimilarityPct: Double
)

/**
 * Confidence Level Enum
 */
enum class ConfidenceLevel {
    VERY_HIGH,
    HIGH,
    MEDIUM,
    LOW
}

/**
 * Final Classification Result
 */
data class ClassificationResult(
    val nearestSampleId: Int,
    val nearestSampleName: String,
    val isUnknown: Boolean,
    val confidenceScorePct: Double,
    val confidenceLevel: ConfidenceLevel,
    val topMatch: ReferenceMatch,
    val allMatches: List<ReferenceMatch>,
    val processingTimeMs: Long,
    val thresholdUsed: Double,
    val marginPct: Double = 0.0,
    val isClassifiable: Boolean = true
)

/**
 * Debug Stage Bitmap representation for Developer Mode
 */
data class ProcessingStage(
    val stageName: String,
    val description: String,
    val bitmap: Bitmap
)

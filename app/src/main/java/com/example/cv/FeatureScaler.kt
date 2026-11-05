package com.example.cv

enum class NormalizationMethod {
    STANDARD_SCALER,
    MIN_MAX_SCALER
}

/**
 * Normalizes feature vectors using fixed, per-feature physical scaling constants.
 * Does NOT normalize an image using its own internal statistics, guaranteeing that identical
 * physical differences between reference samples and captured images produce identical metric distances.
 */
class FeatureScaler {

    /**
     * Scales feature vector by fixed physical tolerances.
     * Features with size != FeatureSchema.FEATURE_VECTOR_SIZE are returned unchanged.
     */
    fun domainScale(vector: DoubleArray): DoubleArray {
        if (vector.size != FeatureSchema.FEATURE_VECTOR_SIZE) {
            return vector.clone()
        }
        val scaled = DoubleArray(vector.size)

        // Group 1: Color location means and medians
        scaled[0] = vector[0] / FeatureSchema.TOLERANCE_L_MEAN // Mean L*
        scaled[1] = vector[1] / FeatureSchema.TOLERANCE_L_MEAN // Med L*
        scaled[2] = vector[2] / FeatureSchema.TOLERANCE_A_MEAN // Mean a*
        scaled[3] = vector[3] / FeatureSchema.TOLERANCE_A_MEAN // Med a*
        scaled[4] = vector[4] / FeatureSchema.TOLERANCE_B_MEAN // Mean b*
        scaled[5] = vector[5] / FeatureSchema.TOLERANCE_B_MEAN // Med b*
        scaled[6] = vector[6] / FeatureSchema.TOLERANCE_CHROMA // Mean Chroma

        // Group 2: Color spread
        scaled[7] = vector[7] / FeatureSchema.TOLERANCE_L_STD  // StdDev L*
        scaled[8] = vector[8] / FeatureSchema.TOLERANCE_A_STD  // StdDev a*
        scaled[9] = vector[9] / FeatureSchema.TOLERANCE_B_STD  // StdDev b*

        // Group 3: Tonal distribution of L*
        scaled[10] = vector[10] / FeatureSchema.TOLERANCE_TONAL // p10
        scaled[11] = vector[11] / FeatureSchema.TOLERANCE_TONAL // p90
        scaled[12] = vector[12]                                 // Skewness
        scaled[13] = vector[13]                                 // Kurtosis
        scaled[14] = vector[14]                                 // Entropy

        // Group 4: Soft histograms (15 until 62) are already normalized probability distributions in [0, 1]
        for (i in 15 until vector.size) {
            scaled[i] = vector[i]
        }

        return scaled
    }

    /**
     * Scale dispatch. Both STANDARD_SCALER and MIN_MAX_SCALER route to the mathematically
     * valid domain-tolerance transform to keep the user settings visible and functional without
     * per-image distortion.
     */
    fun scale(vector: DoubleArray, method: NormalizationMethod = NormalizationMethod.STANDARD_SCALER): DoubleArray {
        if (vector.size != FeatureSchema.FEATURE_VECTOR_SIZE) {
            return vector.clone()
        }
        return domainScale(vector)
    }

    /**
     * Public utility method retained for backwards compatibility or external callers.
     * Not used in classification pipeline.
     */
    fun minMaxScale(vector: DoubleArray): DoubleArray {
        if (vector.isEmpty()) return vector
        val scaled = DoubleArray(vector.size)
        var minVal = Double.MAX_VALUE
        var maxVal = -Double.MAX_VALUE
        for (v in vector) {
            if (v < minVal) minVal = v
            if (v > maxVal) maxVal = v
        }
        val range = (maxVal - minVal).coerceAtLeast(1e-6)
        for (i in vector.indices) {
            scaled[i] = (vector[i] - minVal) / range
        }
        return scaled
    }
}

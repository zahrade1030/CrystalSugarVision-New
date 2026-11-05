package com.example.cv

import android.graphics.Bitmap

/**
 * Extracts compact, physically motivated CIELAB and statistical feature vector
 * over the segmented sugar sample mask.
 * Vector dimension: 63 features (FeatureSchema.FEATURE_VECTOR_SIZE).
 */
class FeatureExtractor {

    /**
     * Extracts physical CIELAB feature vector for a Bitmap.
     * Segments sample pixels adaptively, ensuring background pixels are excluded.
     */
    fun extractRegionFeatures(
        bitmap: Bitmap,
        regionIndex: Int = 0,
        regionName: String = "Single ROI (Analysis Mask)"
    ): FeatureVector {
        val w = bitmap.width
        val h = bitmap.height
        val total = w * h
        val pixels = IntArray(total)
        bitmap.getPixels(pixels, 0, w, 0, 0, w, h)

        val seg = PixelCore.segmentSample(pixels, w, h)
        val vec = PixelCore.extractFeatureVector(pixels, w, h, seg.mask)

        return FeatureVector(
            regionIndex = regionIndex,
            regionName = regionName,
            values = vec
        )
    }

    /**
     * Pure array extraction overload for testing and internal pipeline processing.
     */
    fun extractFromPixels(
        pixels: IntArray,
        width: Int,
        height: Int,
        mask: BooleanArray? = null,
        regionIndex: Int = 0,
        regionName: String = "Single ROI (Analysis Mask)"
    ): FeatureVector {
        val actualMask = mask ?: PixelCore.segmentSample(pixels, width, height).mask
        val vec = PixelCore.extractFeatureVector(pixels, width, height, actualMask)
        return FeatureVector(
            regionIndex = regionIndex,
            regionName = regionName,
            values = vec
        )
    }
}

package com.example.cv

import android.graphics.Bitmap

class MultiRegionExtractor(private val extractor: FeatureExtractor = FeatureExtractor()) {

    /**
     * Extracts physical features from a sample bitmap over the adaptive foreground mask.
     * Preserves aspect ratio, eliminates background influence, and populates genuine
     * CIELAB, HSV (circular hue mean), and RGB means for display.
     */
    fun extractAllRegions(bitmap: Bitmap): CompleteFeatureSet {
        val startTime = System.currentTimeMillis()

        val w = bitmap.width
        val h = bitmap.height
        val total = w * h
        val pixels = IntArray(total)
        bitmap.getPixels(pixels, 0, w, 0, 0, w, h)

        return extractFromPixels(pixels, w, h, startTime)
    }

    /**
     * Pure array implementation for testability on plain JVM without Robolectric.
     */
    fun extractFromPixels(
        pixels: IntArray,
        w: Int,
        h: Int,
        startTime: Long = System.currentTimeMillis()
    ): CompleteFeatureSet {
        val seg = PixelCore.segmentSample(pixels, w, h)
        val mask = seg.mask

        val region = extractor.extractFromPixels(pixels, w, h, mask, 0, "Single ROI (Analysis Mask)")
        val concat = region.values

        // Display RGB 64-bin Color Histogram computed over foreground sample pixels only
        val histogram = FloatArray(64)
        var sumR = 0.0
        var sumG = 0.0
        var sumB = 0.0
        var sampleCount = 0

        val hues = mutableListOf<Double>()
        val sats = mutableListOf<Double>()
        var sumV = 0.0

        for (i in pixels.indices) {
            if (!mask[i]) continue
            val c = pixels[i]
            val r = (c ushr 16) and 0xFF
            val g = (c ushr 8) and 0xFF
            val b = c and 0xFF

            sumR += r
            sumG += g
            sumB += b
            sampleCount++

            // 64-bin RGB histogram (4 bins per channel)
            val rBin = (r / 64).coerceIn(0, 3)
            val gBin = (g / 64).coerceIn(0, 3)
            val bBin = (b / 64).coerceIn(0, 3)
            val bin = rBin * 16 + gBin * 4 + bBin
            histogram[bin] += 1.0f

            val (hsvH, hsvS, hsvV) = ColorConversions.rgbToHsv(r, g, b)
            hues.add(hsvH)
            sats.add(hsvS)
            sumV += hsvV
        }

        if (sampleCount > 0) {
            val totalF = sampleCount.toFloat()
            for (i in histogram.indices) {
                histogram[i] /= totalF
            }
        }

        val meanR = if (sampleCount > 0) sumR / sampleCount else 0.0
        val meanG = if (sampleCount > 0) sumG / sampleCount else 0.0
        val meanB = if (sampleCount > 0) sumB / sampleCount else 0.0

        // CIELAB means extracted directly from feature vector
        val meanL = if (concat.isNotEmpty()) concat[0] else 0.0
        val meanA = if (concat.size > 2) concat[2] else 0.0
        val meanB_lab = if (concat.size > 4) concat[4] else 0.0

        // Saturation-weighted circular hue mean
        val circH = if (sampleCount > 0) {
            ColorConversions.saturationWeightedHueMean(hues.toDoubleArray(), sats.toDoubleArray())
        } else 0.0
        val meanS = if (sampleCount > 0) sats.average() else 0.0
        val meanV = if (sampleCount > 0) sumV / sampleCount else 0.0

        val extractionTime = System.currentTimeMillis() - startTime

        return CompleteFeatureSet(
            regions = listOf(region),
            concatenatedVector = concat,
            extractionTimeMs = extractionTime,
            colorHistogram = histogram,
            rgbMeans = Triple(meanR, meanG, meanB),
            hsvMeans = Triple(circH, meanS, meanV),
            labMeans = Triple(meanL, meanA, meanB_lab)
        )
    }
}

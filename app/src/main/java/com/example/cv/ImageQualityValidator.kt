package com.example.cv

import android.graphics.Bitmap
import kotlin.math.abs
import kotlin.math.pow
import kotlin.math.sqrt

/**
 * Validates laboratory image quality over the sugar sample region.
 * Evaluates blur, exposure, contrast, clipping, and centering.
 * Distinguishes unmeasurable fatal conditions (no sample, pure black, clipped highlights)
 * from advisory conditions (blur, low contrast, off-centre).
 */
class ImageQualityValidator {

    /**
     * Bitmap validation entry-point.
     * Downscales images larger than 512px by area-averaging to bound memory
     * and make quality scoring resolution-independent.
     */
    fun validate(bitmap: Bitmap): ImageQualityResult {
        val w = bitmap.width
        val h = bitmap.height
        val total = w * h
        val pixels = IntArray(total)
        bitmap.getPixels(pixels, 0, w, 0, 0, w, h)

        return validate(pixels, w, h)
    }

    /**
     * Pure array validation overload for testability on plain JVM.
     */
    fun validate(pixels: IntArray, width: Int, height: Int): ImageQualityResult {
        if (width <= 0 || height <= 0 || pixels.size != width * height) {
            return ImageQualityResult(
                isPassed = false,
                blurScore = 0.0,
                brightnessScore = 0.0,
                contrastScore = 0.0,
                centerAlignmentScore = 0.0,
                overallScore = 0.0,
                rejectionReason = "Invalid image dimensions",
                isClassifiable = false
            )
        }

        // Bound memory and ensure resolution-independence: if max dimension > 512, area-average downscale
        val (procPixels, procW, procH) = if (maxOf(width, height) > 512) {
            val rescaled = PixelCore.downscaleAreaAverage(pixels, width, height, maxDim = 512)
            Triple(rescaled.pixels, rescaled.width, rescaled.height)
        } else {
            Triple(pixels, width, height)
        }

        val totalProc = procW * procH
        val seg = PixelCore.segmentSample(procPixels, procW, procH)
        val hasSample = seg.isSampleDetected && seg.foregroundCount > 0
        val mask = if (hasSample) seg.mask else BooleanArray(totalProc) { true }
        val evalCount = if (hasSample) seg.foregroundCount else totalProc

        // Compute grayscale luminances
        val grayValues = DoubleArray(totalProc)
        var sumGray = 0.0
        var clippedCount = 0

        for (i in 0 until totalProc) {
            val c = procPixels[i]
            val r = (c ushr 16) and 0xFF
            val g = (c ushr 8) and 0xFF
            val b = c and 0xFF

            val gray = 0.299 * r + 0.587 * g + 0.114 * b
            grayValues[i] = gray

            if (mask[i]) {
                sumGray += gray
                if (r >= 250 && g >= 250 && b >= 250) {
                    clippedCount++
                }
            }
        }

        val meanGray = sumGray / evalCount

        // Contrast (Standard deviation of luminance over sample mask)
        var sumSqDiff = 0.0
        for (i in 0 until totalProc) {
            if (mask[i]) {
                val d = grayValues[i] - meanGray
                sumSqDiff += d * d
            }
        }
        val varianceGray = sumSqDiff / evalCount
        val contrastStd = sqrt(varianceGray)

        // Blur score via Laplacian variance:
        // Measured ONLY on sample interior pixels (whose 4-neighbors all belong to the sample)
        // so the sharp background edge does not artificially inflate the sharpness score!
        val blurScore = computeInteriorLaplacianVariance(grayValues, mask, procW, procH)

        // Center Alignment Score (center of mass of sample mask relative to frame center)
        var weightedX = 0.0
        var weightedY = 0.0
        var fgPoints = 0

        for (y in 0 until procH) {
            for (x in 0 until procW) {
                val idx = y * procW + x
                if (mask[idx]) {
                    weightedX += x
                    weightedY += y
                    fgPoints++
                }
            }
        }

        val centerAlignmentScore = if (fgPoints > 0) {
            val centerX = weightedX / fgPoints
            val centerY = weightedY / fgPoints
            val idealX = procW / 2.0
            val idealY = procH / 2.0
            val dx = abs(centerX - idealX) / idealX
            val dy = abs(centerY - idealY) / idealY
            (1.0 - ((dx + dy) / 2.0)).coerceIn(0.0, 1.0)
        } else {
            0.0
        }

        // Quality Rules & Rejection Flags
        val minBlurThreshold = 5.0
        val minBrightnessThreshold = 10.0
        val maxBrightnessThreshold = 248.0
        val minContrastThreshold = 2.0
        val minAlignmentThreshold = 0.10

        var rejectionReason: String? = null
        var isClassifiable = true

        val clippedRatio = clippedCount.toDouble() / evalCount

        if (!seg.isSampleDetected) {
            rejectionReason = "Sample not detected. Place sugar sample in center guide."
            isClassifiable = false
        } else if (meanGray < minBrightnessThreshold) {
            rejectionReason = "Image is underexposed. Check light box illumination."
            isClassifiable = false
        } else if (clippedRatio > 0.60 || meanGray > maxBrightnessThreshold) {
            rejectionReason = "Severe glare/clipped highlights. Lower exposure or light intensity."
            isClassifiable = false
        } else if (blurScore < minBlurThreshold) {
            // Advisory problem: blur does not alter average color tone, so still classifiable
            rejectionReason = "Image is too blurry (Laplacian score: ${blurScore.toInt()}). Please hold camera steady."
            isClassifiable = true
        } else if (contrastStd < minContrastThreshold) {
            // Advisory problem
            rejectionReason = "Low image contrast. Sample cannot be cleanly distinguished."
            isClassifiable = true
        } else if (centerAlignmentScore < minAlignmentThreshold) {
            // Advisory problem
            rejectionReason = "Sample is incorrectly positioned. Place sugar sample in center guide."
            isClassifiable = true
        }

        val isPassed = rejectionReason == null

        // Overall score (0 - 100%)
        val blurNorm = (blurScore / 300.0).coerceAtMost(1.0) * 30.0
        val brightnessNorm = (1.0 - abs(meanGray - 128.0) / 128.0).coerceIn(0.0, 1.0) * 25.0
        val contrastNorm = (contrastStd / 80.0).coerceAtMost(1.0) * 20.0
        val alignmentNorm = centerAlignmentScore * 25.0
        val overallScore = (blurNorm + brightnessNorm + contrastNorm + alignmentNorm).coerceIn(0.0, 100.0)

        return ImageQualityResult(
            isPassed = isPassed,
            blurScore = blurScore,
            brightnessScore = meanGray,
            contrastScore = contrastStd,
            centerAlignmentScore = centerAlignmentScore,
            overallScore = overallScore,
            rejectionReason = rejectionReason,
            isClassifiable = isClassifiable
        )
    }

    /**
     * Computes Laplacian variance strictly on interior mask pixels (all 4-neighbors inside mask).
     * Prevents sample-to-background edge contrast from acting as false crystal texture sharpness.
     */
    private fun computeInteriorLaplacianVariance(
        gray: DoubleArray,
        mask: BooleanArray,
        w: Int,
        h: Int
    ): Double {
        var count = 0
        val lapValues = DoubleArray(w * h)
        var lapSum = 0.0

        for (y in 1 until h - 1) {
            for (x in 1 until w - 1) {
                val idx = y * w + x
                // Interior check: pixel and all 4 cardinal neighbors must be inside the sample mask
                if (mask[idx] &&
                    mask[(y - 1) * w + x] &&
                    mask[(y + 1) * w + x] &&
                    mask[y * w + (x - 1)] &&
                    mask[y * w + (x + 1)]
                ) {
                    val center = gray[idx]
                    val top = gray[(y - 1) * w + x]
                    val btm = gray[(y + 1) * w + x]
                    val lft = gray[y * w + (x - 1)]
                    val rgt = gray[y * w + (x + 1)]

                    val lap = top + btm + lft + rgt - 4.0 * center
                    lapValues[count++] = lap
                    lapSum += lap
                }
            }
        }

        if (count < 10) {
            // Fall back to entire frame Laplacian if interior sample is too small
            return fallbackLaplacianVariance(gray, w, h)
        }

        val lapMean = lapSum / count
        var varSum = 0.0
        for (i in 0 until count) {
            val diff = lapValues[i] - lapMean
            varSum += diff * diff
        }

        return varSum / count
    }

    private fun fallbackLaplacianVariance(gray: DoubleArray, w: Int, h: Int): Double {
        var count = 0
        var lapSum = 0.0
        val lapValues = DoubleArray((w - 2) * (h - 2))

        for (y in 1 until h - 1) {
            for (x in 1 until w - 1) {
                val idx = y * w + x
                val lap = gray[(y - 1) * w + x] + gray[(y + 1) * w + x] +
                          gray[y * w + (x - 1)] + gray[y * w + (x + 1)] -
                          4.0 * gray[idx]
                lapValues[count++] = lap
                lapSum += lap
            }
        }
        if (count == 0) return 0.0
        val mean = lapSum / count
        var varSum = 0.0
        for (i in 0 until count) {
            val d = lapValues[i] - mean
            varSum += d * d
        }
        return varSum / count
    }
}

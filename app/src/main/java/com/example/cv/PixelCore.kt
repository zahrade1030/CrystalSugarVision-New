package com.example.cv

import kotlin.math.*

/**
 * Pure Kotlin core image-processing algorithms.
 * Operates on packed ARGB_8888 IntArray buffers with zero android.graphics dependencies,
 * making all segmentation, extraction, and quality math unit-testable on the plain JVM.
 */
object PixelCore {

    data class SegmentationResult(
        val mask: BooleanArray,
        val foregroundCount: Int,
        val width: Int,
        val height: Int,
        val backgroundLuminance: Double,
        val hasDarkBackground: Boolean,
        val isSampleDetected: Boolean
    ) {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (javaClass != other?.javaClass) return false
            other as SegmentationResult
            if (foregroundCount != other.foregroundCount) return false
            if (width != other.width) return false
            if (height != other.height) return false
            if (!mask.contentEquals(other.mask)) return false
            return true
        }

        override fun hashCode(): Int {
            var result = foregroundCount
            result = 31 * result + width
            result = 31 * result + height
            result = 31 * result + mask.contentHashCode()
            return result
        }
    }

    data class RescaledImage(
        val pixels: IntArray,
        val width: Int,
        val height: Int,
        val scaleFactor: Int
    )

    /**
     * Integer box (area) averaging downscale.
     * Preserves exact width-to-height aspect ratio and eliminates high-frequency noise
     * without bilinear blurring or boundary stretching.
     */
    fun downscaleAreaAverage(pixels: IntArray, width: Int, height: Int, maxDim: Int = 256): RescaledImage {
        if (width <= 0 || height <= 0 || pixels.size != width * height) {
            return RescaledImage(IntArray(0), 0, 0, 1)
        }
        val currentMax = maxOf(width, height)
        if (currentMax <= maxDim) {
            return RescaledImage(pixels.clone(), width, height, 1)
        }

        val factor = maxOf(1, (currentMax + maxDim - 1) / maxDim)
        val newW = maxOf(1, width / factor)
        val newH = maxOf(1, height / factor)
        val out = IntArray(newW * newH)

        for (ny in 0 until newH) {
            val startY = ny * factor
            val endY = minOf(height, (ny + 1) * factor)
            val rowCount = endY - startY

            for (nx in 0 until newW) {
                val startX = nx * factor
                val endX = minOf(width, (nx + 1) * factor)
                val colCount = endX - startX
                val totalPix = rowCount * colCount

                var sumA = 0
                var sumR = 0
                var sumG = 0
                var sumB = 0

                for (y in startY until endY) {
                    val rowOffset = y * width
                    for (x in startX until endX) {
                        val c = pixels[rowOffset + x]
                        sumA += (c ushr 24) and 0xFF
                        sumR += (c ushr 16) and 0xFF
                        sumG += (c ushr 8) and 0xFF
                        sumB += c and 0xFF
                    }
                }

                val avgA = sumA / totalPix
                val avgR = sumR / totalPix
                val avgG = sumG / totalPix
                val avgB = sumB / totalPix

                out[ny * newW + nx] = (avgA shl 24) or (avgR shl 16) or (avgG shl 8) or avgB
            }
        }

        return RescaledImage(out, newW, newH, factor)
    }

    /**
     * Adaptive sample segmentation.
     * Replaces fixed-threshold crop and background removal with adaptive analysis:
     * 1. Samples frame borders to measure dark background level.
     * 2. Detects if dark background exists (if sample fills frame, no background removal is applied).
     * 3. Uses Otsu thresholding clamped relative to measured background/foreground levels.
     * 4. Preserves ALL legitimate foreground components/grains (including small and disconnected grains).
     * 5. Filters only isolated single-pixel dust/noise while strictly retaining small sugar crystals.
     * 6. Fills enclosed internal holes (shadows between sugar crystals).
     */
    fun segmentSample(pixels: IntArray, width: Int, height: Int): SegmentationResult {
        val total = width * height
        if (total == 0 || pixels.size != total) {
            return SegmentationResult(BooleanArray(0), 0, width, height, 0.0, false, false)
        }

        // 1. Compute luminance for all pixels
        val lums = DoubleArray(total)
        var totalLum = 0.0
        for (i in 0 until total) {
            val c = pixels[i]
            val r = (c ushr 16) and 0xFF
            val g = (c ushr 8) and 0xFF
            val b = c and 0xFF
            // Perceptual Rec. 601 luma
            val lum = 0.299 * r + 0.587 * g + 0.114 * b
            lums[i] = lum
            totalLum += lum
        }
        val meanFrameLum = totalLum / total

        // 2. Measure border background statistics (3-pixel border margin)
        val borderLums = mutableListOf<Double>()
        val margin = minOf(3, width / 4, height / 4).coerceAtLeast(1)

        for (y in 0 until height) {
            for (x in 0 until width) {
                if (x < margin || x >= width - margin || y < margin || y >= height - margin) {
                    borderLums.add(lums[y * width + x])
                }
            }
        }
        borderLums.sort()
        val borderMedian = if (borderLums.isNotEmpty()) borderLums[borderLums.size / 2] else 0.0
        val borderP90 = if (borderLums.isNotEmpty()) borderLums[(borderLums.size * 9) / 10] else 0.0

        // Check if sample fills the entire frame (no distinct dark background)
        // If border median luminance is bright (> 55.0) and high contrast with a true black lightbox is absent:
        val hasDarkBackground = borderMedian < 42.0 && borderP90 < 65.0

        if (!hasDarkBackground) {
            // Sample fills frame: all pixels belong to the sample
            val fullMask = BooleanArray(total) { true }
            return SegmentationResult(
                mask = fullMask,
                foregroundCount = total,
                width = width,
                height = height,
                backgroundLuminance = borderMedian,
                hasDarkBackground = false,
                isSampleDetected = meanFrameLum >= 8.0
            )
        }

        // 3. Compute Otsu threshold between background and foreground
        val otsuThresh = computeOtsuThreshold(lums)
        // Clamp threshold: must be safely above border background noise (at least borderMedian + 8.0)
        // and at least 14.0, but not higher than sample median
        val threshold = maxOf(14.0, maxOf(borderMedian + 8.0, otsuThresh * 0.85))

        // Initial candidate mask
        val rawMask = BooleanArray(total)
        for (i in 0 until total) {
            rawMask[i] = lums[i] >= threshold
        }

        // 4. Connected Components Labeling (8-connectivity)
        val labels = IntArray(total)
        var currentLabel = 0
        val componentSizes = mutableListOf<Int>()
        componentSizes.add(0) // label 0 is background

        val queue = IntArray(total)

        for (y in 0 until height) {
            for (x in 0 until width) {
                val idx = y * width + x
                if (rawMask[idx] && labels[idx] == 0) {
                    currentLabel++
                    var head = 0
                    var tail = 0
                    queue[tail++] = idx
                    labels[idx] = currentLabel
                    var compSize = 0

                    while (head < tail) {
                        val curr = queue[head++]
                        compSize++
                        val cy = curr / width
                        val cx = curr % width

                        for (dy in -1..1) {
                            for (dx in -1..1) {
                                if (dx == 0 && dy == 0) continue
                                val ny = cy + dy
                                val nx = cx + dx
                                if (ny in 0 until height && nx in 0 until width) {
                                    val nIdx = ny * width + nx
                                    if (rawMask[nIdx] && labels[nIdx] == 0) {
                                        labels[nIdx] = currentLabel
                                        queue[tail++] = nIdx
                                    }
                                }
                            }
                        }
                    }
                    componentSizes.add(compSize)
                }
            }
        }

        // 5. Component Filtering:
        // Do NOT discard valid sugar grains simply because they are small or physically disconnected!
        // Sugar samples can be multiple grains, dust clusters, or scattered crystals.
        // We reject only isolated 1-4 pixel sensor dust/noise (size <= 4 pixels).
        val minValidGrainSize = 5
        val validMask = BooleanArray(total)
        var fgCount = 0

        for (i in 0 until total) {
            val lbl = labels[i]
            if (lbl > 0 && componentSizes[lbl] >= minValidGrainSize) {
                validMask[i] = true
                fgCount++
            }
        }

        // 6. Fill internal enclosed holes (dark crystal shadow pockets inside the mound)
        fillEnclosedHoles(validMask, width, height)
        fgCount = validMask.count { it }

        // 7. Check minimum coverage
        val coverageRatio = fgCount.toDouble() / total
        val isDetected = coverageRatio >= 0.015 && meanFrameLum >= 8.0

        return SegmentationResult(
            mask = validMask,
            foregroundCount = fgCount,
            width = width,
            height = height,
            backgroundLuminance = borderMedian,
            hasDarkBackground = true,
            isSampleDetected = isDetected
        )
    }

    /**
     * Fills background components that do NOT touch the image boundary.
     * Gaps/shadows inside the crystal sample belong to the sugar sample.
     */
    private fun fillEnclosedHoles(mask: BooleanArray, width: Int, height: Int) {
        val total = width * height
        val visited = BooleanArray(total)
        val queue = IntArray(total)
        var head = 0
        var tail = 0

        // Seed boundary background pixels
        for (x in 0 until width) {
            val top = x
            val btm = (height - 1) * width + x
            if (!mask[top] && !visited[top]) { visited[top] = true; queue[tail++] = top }
            if (!mask[btm] && !visited[btm]) { visited[btm] = true; queue[tail++] = btm }
        }
        for (y in 0 until height) {
            val lft = y * width
            val rgt = y * width + (width - 1)
            if (!mask[lft] && !visited[lft]) { visited[lft] = true; queue[tail++] = lft }
            if (!mask[rgt] && !visited[rgt]) { visited[rgt] = true; queue[tail++] = rgt }
        }

        // BFS flood-fill outer background
        while (head < tail) {
            val curr = queue[head++]
            val cy = curr / width
            val cx = curr % width

            val neighbors = intArrayOf(
                (cy - 1) * width + cx,
                (cy + 1) * width + cx,
                cy * width + (cx - 1),
                cy * width + (cx + 1)
            )

            for (n in neighbors) {
                val ny = n / width
                val nx = n % width
                if (ny in 0 until height && nx in 0 until width) {
                    if (!mask[n] && !visited[n]) {
                        visited[n] = true
                        queue[tail++] = n
                    }
                }
            }
        }

        // Any background pixel not connected to border is an internal enclosed hole -> fill it!
        for (i in 0 until total) {
            if (!mask[i] && !visited[i]) {
                mask[i] = true
            }
        }
    }

    private fun computeOtsuThreshold(lums: DoubleArray): Double {
        val hist = IntArray(256)
        for (v in lums) {
            val bin = v.roundToInt().coerceIn(0, 255)
            hist[bin]++
        }
        val total = lums.size
        var sum = 0.0
        for (t in 0 until 256) sum += t * hist[t]

        var sumB = 0.0
        var wB = 0
        var maxVariance = 0.0
        var bestThreshold = 25.0

        for (t in 0 until 256) {
            wB += hist[t]
            if (wB == 0) continue
            val wF = total - wB
            if (wF == 0) break

            sumB += t.toDouble() * hist[t]
            val mB = sumB / wB
            val mF = (sum - sumB) / wF

            val betweenVariance = wB.toDouble() * wF.toDouble() * (mB - mF) * (mB - mF)
            if (betweenVariance > maxVariance) {
                maxVariance = betweenVariance
                bestThreshold = t.toDouble()
            }
        }
        return bestThreshold
    }

    /**
     * Extracts physical CIELAB feature vector (63 features) over sample mask only.
     * Computes colour location, colour spread, L* tonal shape, and soft histograms.
     */
    fun extractFeatureVector(pixels: IntArray, width: Int, height: Int, mask: BooleanArray): DoubleArray {
        val vec = DoubleArray(FeatureSchema.FEATURE_VECTOR_SIZE)
        val sampleCount = mask.count { it }
        if (sampleCount == 0 || pixels.size != width * height || mask.size != width * height) {
            return vec
        }

        val lStarArr = DoubleArray(sampleCount)
        val aStarArr = DoubleArray(sampleCount)
        val bStarArr = DoubleArray(sampleCount)
        val chromaArr = DoubleArray(sampleCount)

        var idx = 0
        var sumL = 0.0
        var sumA = 0.0
        var sumB = 0.0
        var sumC = 0.0

        for (i in pixels.indices) {
            if (!mask[i]) continue
            val c = pixels[i]
            val r = (c ushr 16) and 0xFF
            val g = (c ushr 8) and 0xFF
            val b = c and 0xFF

            val (lStar, aStar, bStar) = ColorConversions.sRgbToCieLab(r, g, b)
            val chroma = sqrt(aStar * aStar + bStar * bStar)

            lStarArr[idx] = lStar
            aStarArr[idx] = aStar
            bStarArr[idx] = bStar
            chromaArr[idx] = chroma

            sumL += lStar
            sumA += aStar
            sumB += bStar
            sumC += chroma
            idx++
        }

        // Group 1: Color location means and medians
        val meanL = sumL / sampleCount
        val meanA = sumA / sampleCount
        val meanB = sumB / sampleCount
        val meanC = sumC / sampleCount

        lStarArr.sort()
        aStarArr.sort()
        bStarArr.sort()
        chromaArr.sort()

        val medL = getMedian(lStarArr)
        val medA = getMedian(aStarArr)
        val medB = getMedian(bStarArr)

        vec[0] = meanL
        vec[1] = medL
        vec[2] = meanA
        vec[3] = medA
        vec[4] = meanB
        vec[5] = medB
        vec[6] = meanC

        // Group 2: Color spread (Standard deviations)
        var varSumL = 0.0
        var varSumA = 0.0
        var varSumB = 0.0
        for (i in 0 until sampleCount) {
            val dL = lStarArr[i] - meanL
            val dA = aStarArr[i] - meanA
            val dB = bStarArr[i] - meanB
            varSumL += dL * dL
            varSumA += dA * dA
            varSumB += dB * dB
        }
        val stdL = sqrt(varSumL / sampleCount)
        val stdA = sqrt(varSumA / sampleCount)
        val stdB = sqrt(varSumB / sampleCount)

        vec[7] = stdL
        vec[8] = stdA
        vec[9] = stdB

        // Group 3: Tonal distribution of L*
        val p10 = getPercentile(lStarArr, 0.10)
        val p90 = getPercentile(lStarArr, 0.90)

        var m3L = 0.0
        var m4L = 0.0
        for (v in lStarArr) {
            val dL = v - meanL
            val d2 = dL * dL
            m3L += d2 * dL
            m4L += d2 * d2
        }
        val skewL = if (stdL > 1e-5) (m3L / sampleCount) / (stdL * stdL * stdL) else 0.0
        val kurtL = if (stdL > 1e-5) ((m4L / sampleCount) / (stdL * stdL * stdL * stdL)) - 3.0 else 0.0

        // Shannon entropy of L* over fixed 32 bins in [0, 100]
        val entropyBins = IntArray(32)
        for (v in lStarArr) {
            val bIdx = ((v / 100.0) * 31.99).toInt().coerceIn(0, 31)
            entropyBins[bIdx]++
        }
        var entropyL = 0.0
        for (count in entropyBins) {
            if (count > 0) {
                val p = count.toDouble() / sampleCount
                entropyL -= p * (ln(p) / ln(2.0))
            }
        }

        vec[10] = p10
        vec[11] = p90
        vec[12] = skewL
        vec[13] = kurtL
        vec[14] = entropyL

        // Group 4: Soft-binned 1D Histograms (16 bins each)
        // Linear interpolation eliminates bin-edge cliff effects
        fillSoftHistogram(lStarArr, vec, FeatureSchema.HIST_L_START, FeatureSchema.HIST_L_BINS, FeatureSchema.HIST_L_MIN, FeatureSchema.HIST_L_MAX)
        fillSoftHistogram(aStarArr, vec, FeatureSchema.HIST_A_START, FeatureSchema.HIST_A_BINS, FeatureSchema.HIST_A_MIN, FeatureSchema.HIST_A_MAX)
        fillSoftHistogram(bStarArr, vec, FeatureSchema.HIST_B_START, FeatureSchema.HIST_B_BINS, FeatureSchema.HIST_B_MIN, FeatureSchema.HIST_B_MAX)

        return vec
    }

    private fun fillSoftHistogram(
        sortedData: DoubleArray,
        target: DoubleArray,
        offset: Int,
        binCount: Int,
        minVal: Double,
        maxVal: Double
    ) {
        val range = (maxVal - minVal).coerceAtLeast(1e-5)
        val binWidth = range / binCount
        var totalWeight = 0.0

        for (v in sortedData) {
            val pos = ((v - minVal) / binWidth) - 0.5
            val leftBin = floor(pos).toInt()
            val rightBin = leftBin + 1
            val frac = pos - leftBin

            if (leftBin in 0 until binCount) {
                val w = 1.0 - frac
                target[offset + leftBin] += w
                totalWeight += w
            }
            if (rightBin in 0 until binCount) {
                val w = frac
                target[offset + rightBin] += w
                totalWeight += w
            }
        }

        if (totalWeight > 0.0) {
            for (b in 0 until binCount) {
                target[offset + b] /= totalWeight
            }
        }
    }

    private fun getMedian(sorted: DoubleArray): Double {
        val n = sorted.size
        return if (n % 2 == 0) (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0 else sorted[n / 2]
    }

    private fun getPercentile(sorted: DoubleArray, p: Double): Double {
        val n = sorted.size
        if (n == 0) return 0.0
        val pos = (n - 1) * p.coerceIn(0.0, 1.0)
        val low = floor(pos).toInt()
        val high = ceil(pos).toInt()
        val frac = pos - low
        return if (low == high) sorted[low] else sorted[low] * (1.0 - frac) + sorted[high] * frac
    }
}

package com.example.cv

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.Locale

class CvRecognitionTest {

    // ---------------------------------------------------------
    // 1. ColorConversions Tests
    // ---------------------------------------------------------

    @Test
    fun testRgbToLab_pureWhite() {
        val (l, a, b) = ColorConversions.sRgbToCieLab(255, 255, 255)
        assertEquals(100.0, l, 0.5)
        assertEquals(0.0, a, 1.0)
        assertEquals(0.0, b, 1.0)
    }

    @Test
    fun testRgbToLab_pureBlack() {
        val (l, a, b) = ColorConversions.sRgbToCieLab(0, 0, 0)
        assertEquals(0.0, l, 0.1)
        assertEquals(0.0, a, 0.1)
        assertEquals(0.0, b, 0.1)
    }

    @Test
    fun testRgbToLab_midGray() {
        val (l, a, b) = ColorConversions.sRgbToCieLab(128, 128, 128)
        // Mid-gray sRGB (128,128,128) has L* around 53.6
        assertEquals(53.6, l, 1.0)
        assertEquals(0.0, a, 0.5)
        assertEquals(0.0, b, 0.5)
    }

    @Test
    fun testRgbToLab_sugarYellowTint() {
        // Sugar with slight yellow tint (R=245, G=240, B=215)
        val (l, a, b) = ColorConversions.sRgbToCieLab(245, 240, 215)
        assertTrue("L* should be high for bright sugar", l > 90.0)
        assertTrue("b* should be positive indicating yellow chrominance", b > 5.0)
    }

    @Test
    fun testCircularHueMean_boundaryWrap() {
        // Two hues at 10 deg and 350 deg (wrap across 0/360)
        val hues = doubleArrayOf(10.0, 350.0)
        val saturations = doubleArrayOf(0.5, 0.5)
        val circularMean = ColorConversions.saturationWeightedHueMean(hues, saturations)
        // Average should be around 0 deg (or 360 deg), NOT 180 deg (arithmetic average)
        assertTrue(
            "Circular mean of 10 and 350 degrees must be close to 0 or 360, was $circularMean",
            circularMean < 5.0 || circularMean > 355.0
        )
    }

    // ---------------------------------------------------------
    // 2. PixelCore Segmentation & Feature Extraction Tests
    // ---------------------------------------------------------

    @Test
    fun testDownscaleAreaAverage() {
        val width = 64
        val height = 48
        val pixels = IntArray(width * height) { 0xFFFFFFFF.toInt() }

        val rescaled = PixelCore.downscaleAreaAverage(pixels, width, height, maxDim = 32)
        assertEquals(32, rescaled.width)
        assertEquals(24, rescaled.height)
        assertEquals(32 * 24, rescaled.pixels.size)
        // Check pixel value preserved (white)
        val samplePixel = rescaled.pixels[0]
        val r = (samplePixel shr 16) and 0xFF
        val g = (samplePixel shr 8) and 0xFF
        val b = samplePixel and 0xFF
        assertEquals(255, r)
        assertEquals(255, g)
        assertEquals(255, b)
    }

    @Test
    fun testAdaptiveSegmentation_separatedGrains() {
        val width = 60
        val height = 60
        val pixels = IntArray(width * height) { 0xFF101010.toInt() } // Dark lightbox background

        // Place a cluster of bright sugar crystal pixels in the center (size 10x10 = 100 pixels)
        for (y in 25 until 35) {
            for (x in 25 until 35) {
                pixels[y * width + x] = 0xFFF5F2EA.toInt()
            }
        }
        // Place a single noise pixel (size 1) at (5, 5)
        pixels[5 * width + 5] = 0xFFFFFFFF.toInt()

        val result = PixelCore.segmentSample(pixels, width, height)
        val mask = result.mask

        assertTrue("Foreground should detect the crystal cluster", result.foregroundCount >= 90)
        assertFalse("Single isolated noise pixel must be rejected", mask[5 * width + 5])
    }

    @Test
    fun testFeatureVectorExtraction_uniformField() {
        val width = 32
        val height = 32
        val pixels = IntArray(width * height) { 0xFFE0E0E0.toInt() }
        val mask = BooleanArray(width * height) { true }

        val vector = PixelCore.extractFeatureVector(pixels, width, height, mask)
        assertEquals(FeatureSchema.FEATURE_VECTOR_SIZE, vector.size)

        // For uniform color:
        // L mean ~ 89.4, spread should be ~0.0
        val lMean = vector[0]
        val lStd = vector[7]

        assertTrue("L* mean should be in valid range", lMean > 85.0 && lMean < 95.0)
        assertEquals(0.0, lStd, 0.5)

        // Verify soft histograms sum to ~1.0
        val lHistSum = (FeatureSchema.HIST_L_START until (FeatureSchema.HIST_L_START + FeatureSchema.HIST_L_BINS)).sumOf { vector[it] }
        val aHistSum = (FeatureSchema.HIST_A_START until (FeatureSchema.HIST_A_START + FeatureSchema.HIST_A_BINS)).sumOf { vector[it] }
        val bHistSum = (FeatureSchema.HIST_B_START until (FeatureSchema.HIST_B_START + FeatureSchema.HIST_B_BINS)).sumOf { vector[it] }

        assertEquals(1.0, lHistSum, 0.05)
        assertEquals(1.0, aHistSum, 0.05)
        assertEquals(1.0, bHistSum, 0.05)
    }

    @Test
    fun testIcumsaMonotonicity_lDecreasesAndbIncreases() {
        // Simulate ICUMSA grades with typical RGB tones:
        // ICUMSA 45: Bright refined white (245, 245, 243)
        // ICUMSA 150: Extra white with slight yellow (240, 237, 225)
        // ICUMSA 500: Plantation white / light cane (228, 218, 190)
        // ICUMSA 800: Brown cane sugar (205, 185, 140)

        val tones = listOf(
            0xFFF5F5F3.toInt(), // 45
            0xFFF0EDE1.toInt(), // 150
            0xFFE4DABE.toInt(), // 500
            0xFFCDB98C.toInt()  // 800
        )

        val lMeans = mutableListOf<Double>()
        val bMeans = mutableListOf<Double>()

        for (tone in tones) {
            val pixels = IntArray(16 * 16) { tone }
            val mask = BooleanArray(16 * 16) { true }
            val vec = PixelCore.extractFeatureVector(pixels, 16, 16, mask)
            lMeans.add(vec[0]) // L* mean
            bMeans.add(vec[4]) // b* mean
        }

        // Check L* decreases monotonically
        for (i in 0 until lMeans.size - 1) {
            assertTrue(
                "L* must decrease with ICUMSA color: ${lMeans[i]} vs ${lMeans[i + 1]}",
                lMeans[i] > lMeans[i + 1]
            )
        }

        // Check b* (yellow chrominance) increases monotonically
        for (i in 0 until bMeans.size - 1) {
            assertTrue(
                "b* must increase with ICUMSA brownness: ${bMeans[i]} vs ${bMeans[i + 1]}",
                bMeans[i] < bMeans[i + 1]
            )
        }
    }

    // ---------------------------------------------------------
    // 3. SimilarityCalculator & Matching Tests
    // ---------------------------------------------------------

    @Test
    fun testSimilarity_identicalVectorIs100Percent() {
        val pixels = IntArray(32 * 32) { 0xFFF2EFE8.toInt() }
        val mask = BooleanArray(32 * 32) { true }
        val vector = PixelCore.extractFeatureVector(pixels, 32, 32, mask)

        val calc = SimilarityCalculator()
        val ref = StoredReferenceVector(1, "Sample 1", 45, vector)
        val result = calc.classifySample(vector, listOf(ref))

        assertEquals(100.0, result.confidenceScorePct, 0.01)
        assertFalse(result.isUnknown)
        assertEquals(1, result.nearestSampleId)
    }

    @Test
    fun testSimilarity_closeVsDistantSample() {
        val mask = BooleanArray(16 * 16) { true }
        val refPixels = IntArray(16 * 16) { 0xFFF5F5F0.toInt() } // Reference (ICUMSA 45)
        val closePixels = IntArray(16 * 16) { 0xFFF2F2EC.toInt() } // Close sample
        val distantPixels = IntArray(16 * 16) { 0xFFC0A870.toInt() } // Distant raw brown sample

        val refVec = PixelCore.extractFeatureVector(refPixels, 16, 16, mask)
        val closeVec = PixelCore.extractFeatureVector(closePixels, 16, 16, mask)
        val distantVec = PixelCore.extractFeatureVector(distantPixels, 16, 16, mask)

        val calc = SimilarityCalculator()
        val ref = StoredReferenceVector(1, "Sample 1", 45, refVec)

        val closeRes = calc.classifySample(closeVec, listOf(ref))
        val distantRes = calc.classifySample(distantVec, listOf(ref))

        assertTrue("Close sample should score high (> 85%), was ${closeRes.confidenceScorePct}", closeRes.confidenceScorePct > 85.0)
        assertTrue("Distant sample should score much lower than close sample", distantRes.confidenceScorePct < closeRes.confidenceScorePct)
        assertTrue("Distant sample should score < 60%, was ${distantRes.confidenceScorePct}", distantRes.confidenceScorePct < 60.0)
    }

    @Test
    fun testSimilarity_alienColorRejection() {
        val mask = BooleanArray(16 * 16) { true }
        val whiteSugar = PixelCore.extractFeatureVector(IntArray(16 * 16) { 0xFFF8F8F4.toInt() }, 16, 16, mask)
        val brightGreen = PixelCore.extractFeatureVector(IntArray(16 * 16) { 0xFF00FF00.toInt() }, 16, 16, mask)

        val calc = SimilarityCalculator()
        val ref = StoredReferenceVector(1, "White Sugar", 45, whiteSugar)
        val result = calc.classifySample(brightGreen, listOf(ref), similarityThresholdPct = 65.0)

        assertTrue("Alien non-sugar color should score near 0%, was ${result.confidenceScorePct}", result.confidenceScorePct < 25.0)
        assertTrue("Alien non-sugar color must be classified as Unknown", result.isUnknown)
    }

    @Test
    fun testSimilarity_invalidDimensionRejection() {
        val validVec = DoubleArray(63) { 1.0 }
        val oldLegacyVec = DoubleArray(191) { 1.0 }

        val calc = SimilarityCalculator()
        val ref = StoredReferenceVector(1, "Sample", 45, validVec)
        val result = calc.classifySample(oldLegacyVec, listOf(ref))

        assertEquals("Mismatched dimensions must yield 0.0 confidence", 0.0, result.confidenceScorePct, 0.001)
        assertTrue(result.isUnknown)
    }

    // ---------------------------------------------------------
    // 4. FeatureScaler Tests
    // ---------------------------------------------------------

    @Test
    fun testFeatureScaler_domainScaling() {
        val rawVec = DoubleArray(63)
        rawVec[0] = FeatureSchema.TOLERANCE_L_MEAN // 1 tolerance unit -> 1.0
        rawVec[2] = FeatureSchema.TOLERANCE_A_MEAN // 1 tolerance unit -> 1.0
        rawVec[4] = FeatureSchema.TOLERANCE_B_MEAN // 1 tolerance unit -> 1.0

        val scaler = FeatureScaler()
        val scaled = scaler.domainScale(rawVec)
        assertEquals(63, scaled.size)
        assertEquals(1.0, scaled[0], 0.001)
        assertEquals(1.0, scaled[2], 0.001)
        assertEquals(1.0, scaled[4], 0.001)
    }

    // ---------------------------------------------------------
    // 5. Locale Independence in Serialization & Formatting
    // ---------------------------------------------------------

    @Test
    fun testVectorSerialization_localeIndependence() {
        val originalLocale = Locale.getDefault()
        try {
            // Force a locale that uses commas as decimal separators (e.g. GERMANY)
            Locale.setDefault(Locale.GERMANY)

            val originalValues = doubleArrayOf(12.345, 67.891, 0.0001, -45.67)
            val jsonString = originalValues.joinToString(prefix = "[", postfix = "]") {
                String.format(Locale.US, "%.6f", it)
            }

            // Verify the serialized string uses '.' and not ','
            assertTrue("Serialized JSON must use dot decimals", jsonString.contains("12.345"))
            assertFalse("Serialized JSON must NOT contain commas inside numbers", jsonString.contains("12,345"))

            // Verify deserialization
            val parsed = jsonString.removeSurrounding("[", "]")
                .split(",")
                .map { it.trim().toDouble() }
                .toDoubleArray()

            assertEquals(originalValues.size, parsed.size)
            for (i in originalValues.indices) {
                assertEquals(originalValues[i], parsed[i], 0.0001)
            }
        } finally {
            Locale.setDefault(originalLocale)
        }
    }
}

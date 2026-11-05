package com.example.cv

import kotlin.math.*

/**
 * Pure Kotlin colorimetric conversion utilities (zero Android platform dependencies).
 * Implements standard CIE 1931 sRGB to CIE L*a*b* under CIE Standard Illuminant D65
 * (Xn = 95.0489, Yn = 100.0, Zn = 108.8840).
 */
object ColorConversions {

    private const val XN = 95.0489
    private const val YN = 100.0000
    private const val ZN = 108.8840

    // Standard CIE delta threshold: 6/29
    private const val DELTA = 6.0 / 29.0
    private const val DELTA_CUBE = DELTA * DELTA * DELTA // approx 0.008856451679
    private const val ONE_OVER_THREE_DELTA_SQ = 1.0 / (3.0 * DELTA * DELTA) // approx 7.787037037
    private const val FOUR_OVER_TWENTY_NINE = 4.0 / 29.0 // approx 0.137931034

    /**
     * Converts sRGB byte channel (0..255) to linear sRGB value in [0.0, 1.0].
     * Standard IEC 61966-2-1 inverse transfer characteristic.
     */
    fun sRgbToLinear(c8: Int): Double {
        val v = c8 / 255.0
        return if (v <= 0.04045) {
            v / 12.92
        } else {
            ((v + 0.055) / 1.055).pow(2.4)
        }
    }

    /**
     * Converts linear sRGB value [0.0, 1.0] to gamma-compressed sRGB byte (0..255).
     */
    fun linearToSRgb(v: Double): Int {
        val clamped = v.coerceIn(0.0, 1.0)
        val s = if (clamped <= 0.0031308) {
            clamped * 12.92
        } else {
            1.055 * clamped.pow(1.0 / 2.4) - 0.055
        }
        return (s * 255.0).roundToInt().coerceIn(0, 255)
    }

    /**
     * Converts 8-bit sRGB (0..255) to CIE 1976 L*a*b* under D65 illuminant.
     * Output:
     * - L*: [0.0, 100.0]
     * - a*: approx [-128.0, 127.0] (for sugar typically -5.0..+35.0)
     * - b*: approx [-128.0, 127.0] (for sugar typically -5.0..+70.0)
     */
    fun sRgbToCieLab(r: Int, g: Int, b: Int): Triple<Double, Double, Double> {
        val rLin = sRgbToLinear(r)
        val gLin = sRgbToLinear(g)
        val bLin = sRgbToLinear(b)

        // Linear sRGB to CIE 1931 XYZ (D65)
        val x = (0.4124564 * rLin + 0.3575761 * gLin + 0.1804375 * bLin) * 100.0
        val y = (0.2126729 * rLin + 0.7151522 * gLin + 0.0721750 * bLin) * 100.0
        val z = (0.0193339 * rLin + 0.1191920 * gLin + 0.9503041 * bLin) * 100.0

        val fx = fLab(x / XN)
        val fy = fLab(y / YN)
        val fz = fLab(z / ZN)

        val lStar = (116.0 * fy - 16.0).coerceIn(0.0, 100.0)
        val aStar = 500.0 * (fx - fy)
        val bStar = 200.0 * (fy - fz)

        return Triple(lStar, aStar, bStar)
    }

    private fun fLab(t: Double): Double {
        return if (t > DELTA_CUBE) {
            t.pow(1.0 / 3.0)
        } else {
            t * ONE_OVER_THREE_DELTA_SQ + FOUR_OVER_TWENTY_NINE
        }
    }

    private fun fInvLab(t: Double): Double {
        return if (t > DELTA) {
            t * t * t
        } else {
            3.0 * DELTA * DELTA * (t - FOUR_OVER_TWENTY_NINE)
        }
    }

    /**
     * Converts CIE 1976 L*a*b* back to packed ARGB_8888 sRGB integer.
     * Useful for authentic stage visualisations.
     */
    fun cieLabToPackedArgb(lStar: Double, aStar: Double, bStar: Double, alpha: Int = 0xFF): Int {
        val fy = (lStar + 16.0) / 116.0
        val fx = aStar / 500.0 + fy
        val fz = fy - bStar / 200.0

        val x = XN * fInvLab(fx)
        val y = YN * fInvLab(fy)
        val z = ZN * fInvLab(fz)

        // XYZ to linear sRGB
        val xNorm = x / 100.0
        val yNorm = y / 100.0
        val zNorm = z / 100.0

        val rLin =  3.2404542 * xNorm - 1.5371385 * yNorm - 0.4985314 * zNorm
        val gLin = -0.9692660 * xNorm + 1.8760108 * yNorm + 0.0415560 * zNorm
        val bLin =  0.0556434 * xNorm - 0.2040259 * yNorm + 1.0572252 * zNorm

        val r = linearToSRgb(rLin)
        val g = linearToSRgb(gLin)
        val b = linearToSRgb(bLin)

        return (alpha shl 24) or (r shl 16) or (g shl 8) or b
    }

    /**
     * Converts sRGB (0..255) to HSV.
     * Returns Triple(H, S, V):
     * - H in [0.0, 360.0)
     * - S in [0.0, 1.0]
     * - V in [0.0, 1.0]
     */
    fun rgbToHsv(r: Int, g: Int, b: Int): Triple<Double, Double, Double> {
        val rf = r / 255.0
        val gf = g / 255.0
        val bf = b / 255.0

        val maxVal = maxOf(rf, gf, bf)
        val minVal = minOf(rf, gf, bf)
        val delta = maxVal - minVal

        val v = maxVal
        val s = if (maxVal > 1e-6) delta / maxVal else 0.0

        val h = when {
            delta < 1e-6 -> 0.0
            maxVal == rf -> 60.0 * (((gf - bf) / delta) % 6.0)
            maxVal == gf -> 60.0 * (((bf - rf) / delta) + 2.0)
            else -> 60.0 * (((rf - gf) / delta) + 4.0)
        }
        val normalizedH = (h + 360.0) % 360.0
        return Triple(normalizedH, s, v)
    }

    /**
     * Calculates saturation-weighted circular mean of Hues.
     * Prevents wrap-around distortion across 0°/360° (red hues) and avoids noise
     * from achromatic near-white crystals with near-zero saturation.
     */
    fun saturationWeightedHueMean(hues: DoubleArray, sats: DoubleArray): Double {
        if (hues.isEmpty() || hues.size != sats.size) return 0.0
        var sinSum = 0.0
        var cosSum = 0.0
        var totalWeight = 0.0

        for (i in hues.indices) {
            val weight = sats[i].coerceIn(0.0, 1.0)
            val rad = Math.toRadians(hues[i])
            sinSum += weight * sin(rad)
            cosSum += weight * cos(rad)
            totalWeight += weight
        }

        if (totalWeight < 1e-5) return 0.0
        val avgRad = atan2(sinSum, cosSum)
        val deg = Math.toDegrees(avgRad)
        return (deg + 360.0) % 360.0
    }
}

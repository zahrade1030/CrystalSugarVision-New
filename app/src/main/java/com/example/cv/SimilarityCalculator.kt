package com.example.cv

import kotlin.math.*

data class StoredReferenceVector(
    val sampleId: Int,
    val sampleName: String,
    val icumsaValue: Int,
    val vector: DoubleArray
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false
        other as StoredReferenceVector
        return vector.contentEquals(other.vector)
    }

    override fun hashCode(): Int {
        return vector.contentHashCode()
    }
}

class SimilarityCalculator {

    /**
     * Euclidean distance between two vectors.
     * Returns 999.0 if lengths mismatch or are empty.
     */
    fun euclideanDistance(v1: DoubleArray, v2: DoubleArray): Double {
        if (v1.isEmpty() || v2.isEmpty() || v1.size != v2.size) return 999.0
        var sum = 0.0
        for (i in v1.indices) {
            val d = v1[i] - v2[i]
            sum += d * d
        }
        return sqrt(sum)
    }

    /**
     * Un-centered conventional Cosine similarity on non-negative feature space:
     * (v1 . v2) / (||v1|| * ||v2||) in [0.0, 1.0].
     * Preserved as a diagnostic metric for existing UI tables.
     */
    fun cosineSimilarity(v1: DoubleArray, v2: DoubleArray): Double {
        if (v1.isEmpty() || v2.isEmpty() || v1.size != v2.size) return 0.0
        var dot = 0.0
        var norm1 = 0.0
        var norm2 = 0.0
        for (i in v1.indices) {
            dot += v1[i] * v2[i]
            norm1 += v1[i] * v1[i]
            norm2 += v2[i] * v2[i]
        }
        val denom = sqrt(norm1 * norm2)
        return if (denom > 1e-8) (dot / denom).coerceIn(0.0, 1.0) else 0.0
    }

    /**
     * Calculates Bhattacharyya distance and coefficient between two normalized histograms.
     * BC = sum(sqrt(h1[i] * h2[i])) in [0.0, 1.0].
     * Distance = sqrt(1.0 - BC).
     */
    fun bhattacharyyaDistance(h1: DoubleArray, h2: DoubleArray): Pair<Double, Double> {
        val minLen = minOf(h1.size, h2.size)
        if (minLen == 0) return Pair(1.0, 0.0)

        var bc = 0.0
        for (i in 0 until minLen) {
            bc += sqrt(maxOf(0.0, h1[i] * h2[i]))
        }
        bc = bc.coerceIn(0.0, 1.0)
        val distance = sqrt((1.0 - bc).coerceAtLeast(0.0))
        return Pair(distance, bc)
    }

    /**
     * Verifies if a vector is usable: correct size, all finite numbers, and non-zero energy.
     */
    private fun isUsable(vec: DoubleArray): Boolean {
        if (vec.size != FeatureSchema.FEATURE_VECTOR_SIZE) return false
        var hasNonZero = false
        for (v in vec) {
            if (!v.isFinite()) return false
            if (abs(v) > 1e-6) hasNonZero = true
        }
        return hasNonZero
    }

    /**
     * Evaluates input vector against stored reference samples.
     * Employs group-weighted distance scaling, soft-histogram Bhattacharyya coefficient,
     * and margin-aware confidence.
     */
    fun classifySample(
        inputVector: DoubleArray,
        references: List<StoredReferenceVector>,
        similarityThresholdPct: Double = 65.0,
        startTimeMs: Long = System.currentTimeMillis()
    ): ClassificationResult {
        val isInputValid = isUsable(inputVector)

        if (!isInputValid || references.isEmpty()) {
            val emptyMatches = references.map { ref ->
                ReferenceMatch(
                    sampleId = ref.sampleId,
                    sampleName = ref.sampleName,
                    icumsaValue = ref.icumsaValue,
                    euclideanDistance = 999.0,
                    euclideanSimilarityPct = 0.0,
                    cosineSimilarity = 0.0,
                    cosineSimilarityPct = 0.0,
                    bhattacharyyaDistance = 1.0,
                    bhattacharyyaSimilarityPct = 0.0,
                    finalSimilarityPct = 0.0
                )
            }.ifEmpty {
                listOf(
                    ReferenceMatch(
                        sampleId = 0,
                        sampleName = "Unknown Sample",
                        icumsaValue = 0,
                        euclideanDistance = 999.0,
                        euclideanSimilarityPct = 0.0,
                        cosineSimilarity = 0.0,
                        cosineSimilarityPct = 0.0,
                        bhattacharyyaDistance = 1.0,
                        bhattacharyyaSimilarityPct = 0.0,
                        finalSimilarityPct = 0.0
                    )
                )
            }

            return ClassificationResult(
                nearestSampleId = 0,
                nearestSampleName = "Unknown Sample",
                isUnknown = true,
                confidenceScorePct = 0.0,
                confidenceLevel = ConfidenceLevel.LOW,
                topMatch = emptyMatches.first(),
                allMatches = emptyMatches,
                processingTimeMs = System.currentTimeMillis() - startTimeMs,
                thresholdUsed = similarityThresholdPct,
                marginPct = 0.0,
                isClassifiable = false
            )
        }

        val matches = mutableListOf<ReferenceMatch>()

        for (ref in references) {
            if (!isUsable(ref.vector)) {
                // Invalid or legacy reference vector: rank lowest, 0% match, never win
                matches.add(
                    ReferenceMatch(
                        sampleId = ref.sampleId,
                        sampleName = ref.sampleName,
                        icumsaValue = ref.icumsaValue,
                        euclideanDistance = 999.0,
                        euclideanSimilarityPct = 0.0,
                        cosineSimilarity = 0.0,
                        cosineSimilarityPct = 0.0,
                        bhattacharyyaDistance = 1.0,
                        bhattacharyyaSimilarityPct = 0.0,
                        finalSimilarityPct = 0.0
                    )
                )
                continue
            }

            // 1. Group-weighted distance in tolerance units
            // Group 1: Color location (features 0..6)
            var sumLocSq = 0.0
            for (i in 0 until FeatureSchema.COLOR_LOCATION_COUNT) {
                val d = inputVector[FeatureSchema.COLOR_LOCATION_START + i] - ref.vector[FeatureSchema.COLOR_LOCATION_START + i]
                sumLocSq += d * d
            }
            val meanLocSq = sumLocSq / FeatureSchema.COLOR_LOCATION_COUNT

            // Group 2: Color spread (features 7..9)
            var sumSpreadSq = 0.0
            for (i in 0 until FeatureSchema.COLOR_SPREAD_COUNT) {
                val d = inputVector[FeatureSchema.COLOR_SPREAD_START + i] - ref.vector[FeatureSchema.COLOR_SPREAD_START + i]
                sumSpreadSq += d * d
            }
            val meanSpreadSq = sumSpreadSq / FeatureSchema.COLOR_SPREAD_COUNT

            // Group 3: Tonal distribution (features 10..14)
            var sumToneSq = 0.0
            for (i in 0 until FeatureSchema.TONAL_SHAPE_COUNT) {
                val d = inputVector[FeatureSchema.TONAL_SHAPE_START + i] - ref.vector[FeatureSchema.TONAL_SHAPE_START + i]
                sumToneSq += d * d
            }
            val meanToneSq = sumToneSq / FeatureSchema.TONAL_SHAPE_COUNT

            val weightedDistSq = (FeatureSchema.WEIGHT_COLOR_LOCATION * meanLocSq +
                                 FeatureSchema.WEIGHT_COLOR_SPREAD * meanSpreadSq +
                                 FeatureSchema.WEIGHT_TONAL_SHAPE * meanToneSq)

            // Gaussian kernel on calibrated tolerance distance:
            // 100% when distance is zero, falls smoothly with increasing difference
            val sigmaSq = 2.0 * FeatureSchema.GAUSSIAN_SIGMA * FeatureSchema.GAUSSIAN_SIGMA
            val distSimPct = (exp(-weightedDistSq / sigmaSq) * 100.0).coerceIn(0.0, 100.0)

            // 2. Soft-histogram Bhattacharyya overlap (L*, a*, b* soft histograms)
            val histL1 = inputVector.sliceArray(FeatureSchema.HIST_L_START until (FeatureSchema.HIST_L_START + FeatureSchema.HIST_L_BINS))
            val histL2 = ref.vector.sliceArray(FeatureSchema.HIST_L_START until (FeatureSchema.HIST_L_START + FeatureSchema.HIST_L_BINS))
            val (_, bcL) = bhattacharyyaDistance(histL1, histL2)

            val histA1 = inputVector.sliceArray(FeatureSchema.HIST_A_START until (FeatureSchema.HIST_A_START + FeatureSchema.HIST_A_BINS))
            val histA2 = ref.vector.sliceArray(FeatureSchema.HIST_A_START until (FeatureSchema.HIST_A_START + FeatureSchema.HIST_A_BINS))
            val (_, bcA) = bhattacharyyaDistance(histA1, histA2)

            val histB1 = inputVector.sliceArray(FeatureSchema.HIST_B_START until (FeatureSchema.HIST_B_START + FeatureSchema.HIST_B_BINS))
            val histB2 = ref.vector.sliceArray(FeatureSchema.HIST_B_START until (FeatureSchema.HIST_B_START + FeatureSchema.HIST_B_BINS))
            val (_, bcB) = bhattacharyyaDistance(histB1, histB2)

            val avgBc = (bcL + bcA + bcB) / 3.0
            val bhattDist = sqrt((1.0 - avgBc).coerceAtLeast(0.0))
            val histSimPct = (avgBc * 100.0).coerceIn(0.0, 100.0)

            // 3. Combined score
            val combinedScore = (FeatureSchema.WEIGHT_DISTANCE * distSimPct + FeatureSchema.WEIGHT_HISTOGRAM * histSimPct).coerceIn(0.0, 100.0)

            // Diagnostic raw Euclidean and conventional Cosine
            val rawEuc = euclideanDistance(inputVector, ref.vector)
            val rawCos = cosineSimilarity(inputVector, ref.vector)

            matches.add(
                ReferenceMatch(
                    sampleId = ref.sampleId,
                    sampleName = ref.sampleName,
                    icumsaValue = ref.icumsaValue,
                    euclideanDistance = rawEuc,
                    euclideanSimilarityPct = distSimPct,
                    cosineSimilarity = rawCos,
                    cosineSimilarityPct = rawCos * 100.0,
                    bhattacharyyaDistance = bhattDist,
                    bhattacharyyaSimilarityPct = histSimPct,
                    finalSimilarityPct = combinedScore
                )
            )
        }

        // Sort descending by final similarity
        matches.sortByDescending { it.finalSimilarityPct }

        val topMatch = matches.first()
        val runnerUp = if (matches.size > 1) matches[1] else null
        val margin = if (runnerUp != null) (topMatch.finalSimilarityPct - runnerUp.finalSimilarityPct).coerceAtLeast(0.0) else 100.0

        val hasValidWinner = topMatch.finalSimilarityPct > 0.0 && topMatch.sampleId > 0
        val isUnknown = !hasValidWinner || topMatch.finalSimilarityPct < similarityThresholdPct

        val finalName = if (isUnknown) "Unknown Sample" else topMatch.sampleName
        val finalId = if (isUnknown) 0 else topMatch.sampleId

        // Margin-aware confidence:
        // ICUMSA grades form an ordered continuum. If a sample lies between two adjacent grades,
        // the margin between #1 and #2 is naturally narrow. It should not be marked as VERY_HIGH,
        // but remains validly classified (not rejected as Unknown unless below the threshold).
        val confidenceLevel = when {
            isUnknown -> ConfidenceLevel.LOW
            topMatch.finalSimilarityPct >= 88.0 && margin >= 6.0 -> ConfidenceLevel.VERY_HIGH
            topMatch.finalSimilarityPct >= 75.0 && margin >= 4.0 -> ConfidenceLevel.HIGH
            topMatch.finalSimilarityPct >= 60.0 -> ConfidenceLevel.MEDIUM
            else -> ConfidenceLevel.LOW
        }

        return ClassificationResult(
            nearestSampleId = finalId,
            nearestSampleName = finalName,
            isUnknown = isUnknown,
            confidenceScorePct = topMatch.finalSimilarityPct,
            confidenceLevel = confidenceLevel,
            topMatch = topMatch,
            allMatches = matches,
            processingTimeMs = System.currentTimeMillis() - startTimeMs,
            thresholdUsed = similarityThresholdPct,
            marginPct = margin,
            isClassifiable = true
        )
    }
}

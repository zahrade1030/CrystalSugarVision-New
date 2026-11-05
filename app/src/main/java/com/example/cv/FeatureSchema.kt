package com.example.cv

/**
 * Single source of truth for feature vector schema, dimensional layout,
 * and recognition prior parameters for the Crystal Sugar Vision classical CV engine.
 */
object FeatureSchema {
    const val SCHEMA_VERSION = 3
    const val FEATURE_VECTOR_SIZE = 63

    // Feature group slices (total 63 dimensions)
    // Group 1: Color location (7 features: Mean L*, Median L*, Mean a*, Median a*, Mean b*, Median b*, Mean Chroma)
    const val COLOR_LOCATION_START = 0
    const val COLOR_LOCATION_COUNT = 7

    // Group 2: Color spread (3 features: StdDev L*, StdDev a*, StdDev b*)
    const val COLOR_SPREAD_START = 7
    const val COLOR_SPREAD_COUNT = 3

    // Group 3: Tonal distribution (5 features: 10th pct L*, 90th pct L*, Skewness L*, Kurtosis L*, Entropy L*)
    const val TONAL_SHAPE_START = 10
    const val TONAL_SHAPE_COUNT = 5

    // Group 4: Soft histograms (16 bins L* [0,100], 16 bins a* [-20,40], 16 bins b* [-10,70]) = 48 features
    const val HIST_L_START = 15
    const val HIST_L_BINS = 16
    const val HIST_A_START = 31
    const val HIST_A_BINS = 16
    const val HIST_B_START = 47
    const val HIST_B_BINS = 16

    // Soft histogram boundaries
    const val HIST_L_MIN = 0.0
    const val HIST_L_MAX = 100.0
    const val HIST_A_MIN = -20.0
    const val HIST_A_MAX = 40.0
    const val HIST_B_MIN = -10.0
    const val HIST_B_MAX = 70.0

    // Feature group weights in distance matching.
    // Rationale: ICUMSA sugar grade is primarily a yellowness (b*) and lightness (L*) rating.
    // Color location carries the decisive physical information separating grades.
    // Tone and spread validate sample homogeneity and crystal reflectance texture.
    // [UNVALIDATED ENGINEERING PRIORS - to be calibrated with replicate real photographs]
    const val WEIGHT_COLOR_LOCATION = 0.65
    const val WEIGHT_COLOR_SPREAD = 0.20
    const val WEIGHT_TONAL_SHAPE = 0.15

    // Combined similarity weighting:
    // Calibrated feature distance (Gaussian kernel on squared weighted mismatch): 70%
    // Soft-histogram overlap (Bhattacharyya coefficient): 30%
    // [UNVALIDATED ENGINEERING PRIORS - to be calibrated with replicate real photographs]
    const val WEIGHT_DISTANCE = 0.70
    const val WEIGHT_HISTOGRAM = 0.30

    // Per-feature tolerances (one unit of mismatch = difference / tolerance)
    // Physical tolerances in CIELAB units:
    // [UNVALIDATED ENGINEERING PRIORS - to be calibrated with replicate real photographs]
    const val TOLERANCE_L_MEAN = 4.5    // L* units
    const val TOLERANCE_A_MEAN = 2.0    // a* units
    const val TOLERANCE_B_MEAN = 3.5    // b* units
    const val TOLERANCE_CHROMA = 3.5    // C* units
    const val TOLERANCE_L_STD = 3.0
    const val TOLERANCE_A_STD = 2.0
    const val TOLERANCE_B_STD = 2.5
    const val TOLERANCE_TONAL = 5.0

    // Gaussian kernel bandwidth parameter sigma
    const val GAUSSIAN_SIGMA = 1.8
}

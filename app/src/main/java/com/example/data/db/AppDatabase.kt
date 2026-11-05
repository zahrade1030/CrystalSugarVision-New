package com.example.data.db

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.sqlite.db.SupportSQLiteDatabase
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

import androidx.room.migration.Migration

@Database(
    entities = [
        ReferenceSampleEntity::class,
        ReferenceVersionEntity::class,
        AnalysisHistoryEntity::class,
        CalibrationProfileEntity::class
    ],
    version = 3,
    exportSchema = false
)
abstract class AppDatabase : RoomDatabase() {

    abstract fun referenceSampleDao(): ReferenceSampleDao
    abstract fun referenceVersionDao(): ReferenceVersionDao
    abstract fun analysisHistoryDao(): AnalysisHistoryDao
    abstract fun calibrationProfileDao(): CalibrationProfileDao

    companion object {
        @Volatile
        private var INSTANCE: AppDatabase? = null

        val MIGRATION_1_2 = object : Migration(1, 2) {
            override fun migrate(db: SupportSQLiteDatabase) {
                // Version 1 to 2 migration
            }
        }

        val MIGRATION_2_3 = object : Migration(2, 3) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL("ALTER TABLE analysis_history ADD COLUMN normalizationMethod TEXT NOT NULL DEFAULT 'STANDARD_SCALER'")
                db.execSQL("ALTER TABLE analysis_history ADD COLUMN algorithmVersion TEXT NOT NULL DEFAULT 'v2.1.0'")
                db.execSQL("ALTER TABLE analysis_history ADD COLUMN matchedImagePath TEXT DEFAULT NULL")
            }
        }

        fun getDatabase(context: Context): AppDatabase {
            return INSTANCE ?: synchronized(this) {
                val instance = Room.databaseBuilder(
                    context.applicationContext,
                    AppDatabase::class.java,
                    "crystal_sugar_db"
                )
                    .addMigrations(MIGRATION_2_3)
                    .addCallback(DatabaseCallback(context.applicationContext))
                    .fallbackToDestructiveMigration()
                    .build()
                INSTANCE = instance
                instance
            }
        }

        private class DatabaseCallback(private val context: Context) : RoomDatabase.Callback() {
            override fun onCreate(db: SupportSQLiteDatabase) {
                super.onCreate(db)
                INSTANCE?.let { database ->
                    CoroutineScope(Dispatchers.IO).launch {
                        ensureSamplesUpdated(context, database)
                    }
                }
            }

            override fun onOpen(db: SupportSQLiteDatabase) {
                super.onOpen(db)
                INSTANCE?.let { database ->
                    CoroutineScope(Dispatchers.IO).launch {
                        ensureSamplesUpdated(context, database)
                    }
                }
            }
        }

        suspend fun ensureSamplesUpdated(context: Context, db: AppDatabase) {
            val dao = db.referenceSampleDao()
            val existing = dao.getAllSamplesDirect()
            val existingIds = existing.map { it.id }.toSet()

            // Only insert baseline samples whose IDs are strictly missing.
            // Never overwrite or delete any existing user sample, image or name!
            val missingIds = (1..6).filter { it !in existingIds }
            if (missingIds.isNotEmpty()) {
                populateMissingReferences(context, db, missingIds)
            }

            // Perform non-destructive compatibility maintenance to ensure vectors match current pipeline schema
            ReferenceVectorMaintenance.ensureVectorsUpdated(context, db)
        }

        // NOTE: Baseline reference samples generated below are synthetic placeholders
        // (coloured discs with procedural noise, not real sugar photographs). Real reference
        // photographs must be loaded through the existing Reference Database screen for operational laboratory analysis.
        private suspend fun populateMissingReferences(context: Context, db: AppDatabase, missingIds: List<Int>) {
            val dao = db.referenceSampleDao()
            val versionDao = db.referenceVersionDao()
            val pipeline = com.example.cv.OpenCvPipeline()

            val dir = java.io.File(context.filesDir, "reference_samples")
            if (!dir.exists()) dir.mkdirs()

            val allConfigs = listOf(
                Triple(1, "نمونه ۱ (خیلی روشن) - Sample 1 (Ultra Light)", 45) to Triple("شکر تصفیه‌شده کریستالی با بالاترین خلوص، درخشندگی و سفیدی (خیلی روشن)", android.graphics.Color.rgb(252, 252, 254), "sample_1.jpg"),
                Triple(2, "نمونه ۲ (روشن) - Sample 2 (Light)", 85) to Triple("شکر سفید کریستالی استاندارد مصرفی با درجه شفافیت بالا (روشن)", android.graphics.Color.rgb(242, 240, 228), "sample_2.jpg"),
                Triple(3, "نمونه ۳ (نیمه روشن) - Sample 3 (Medium Light)", 150) to Triple("شکر سفید کارخانه‌ای درجه دو با ته‌رنگ عاجی ملایم (نیمه روشن)", android.graphics.Color.rgb(230, 220, 195), "sample_3.jpg"),
                Triple(4, "نمونه ۴ (نیمه تیره) - Sample 4 (Medium Dark)", 300) to Triple("شکر قهوه‌ای روشن با بقایای طبیعی ملاس نیشکر (نیمه تیره)", android.graphics.Color.rgb(205, 172, 120), "sample_4.jpg"),
                Triple(5, "نمونه ۵ (تیره) - Sample 5 (Dark)", 500) to Triple("شکر قهوه‌ای تیره با دانه‌های کاراملی ملاس‌دار (تیره)", android.graphics.Color.rgb(165, 115, 60), "sample_5.jpg"),
                Triple(6, "نمونه ۶ (خیلی تیره) - Sample 6 (Ultra Dark)", 800) to Triple("شکر سرخ خام نیشکر با بالاترین میزان رنگدانه و تیرگی (خیلی تیره)", android.graphics.Color.rgb(115, 65, 25), "sample_6.jpg")
            )

            val missingConfigs = allConfigs.filter { (info, _) -> info.first in missingIds }
            val newSamples = mutableListOf<ReferenceSampleEntity>()

            for ((info, detail) in missingConfigs) {
                val (id, name, icumsa) = info
                val (desc, colorInt, fileName) = detail

                val file = java.io.File(dir, fileName)
                val bmp = generateSugarSampleBitmap(colorInt, id)

                try {
                    java.io.FileOutputStream(file).use { out ->
                        bmp.compress(android.graphics.Bitmap.CompressFormat.JPEG, 92, out)
                    }
                } catch (e: Exception) {
                    e.printStackTrace()
                }

                // Process baseline image through the shared reference extraction pipeline
                val rawVector = pipeline.extractReferenceVector(bmp)
                val vectorJson = rawVector.joinToString(
                    separator = ",",
                    prefix = "[",
                    postfix = "]"
                ) { String.format(java.util.Locale.US, "%.5f", it) }

                val entity = ReferenceSampleEntity(
                    id = id,
                    name = name,
                    icumsaValue = icumsa,
                    description = desc,
                    featureVectorJson = vectorJson,
                    imagePath = file.absolutePath,
                    featureVectorSize = rawVector.size
                )
                newSamples.add(entity)
            }

            if (newSamples.isNotEmpty()) {
                dao.insertAll(newSamples)
                for (s in newSamples) {
                    versionDao.insertVersion(
                        ReferenceVersionEntity(
                            sampleId = s.id,
                            versionNumber = 1,
                            name = s.name,
                            icumsaValue = s.icumsaValue,
                            description = s.description,
                            featureVectorJson = s.featureVectorJson,
                            imagePath = s.imagePath,
                            operatorName = "System Initializer",
                            changeDescription = "Initial baseline reference placeholder initialization"
                        )
                    )
                }
            }
        }

        private fun generateSugarSampleBitmap(baseColorInt: Int, seed: Int): android.graphics.Bitmap {
            val bmp = android.graphics.Bitmap.createBitmap(512, 512, android.graphics.Bitmap.Config.ARGB_8888)
            val canvas = android.graphics.Canvas(bmp)
            canvas.drawColor(android.graphics.Color.BLACK)

            val paint = android.graphics.Paint().apply { isAntiAlias = true }
            paint.color = baseColorInt
            canvas.drawCircle(256f, 256f, 185f, paint)

            val random = java.util.Random( seed * 100L )
            for (i in 0 until 4000) {
                val angle = random.nextDouble() * 2 * Math.PI
                val r = random.nextDouble() * 180.0
                val cx = 256f + (r * Math.cos(angle)).toFloat()
                val cy = 256f + (r * Math.sin(angle)).toFloat()
                val noisePaint = android.graphics.Paint().apply {
                    color = if (random.nextBoolean()) android.graphics.Color.WHITE else android.graphics.Color.DKGRAY
                    alpha = 75
                }
                canvas.drawCircle(cx, cy, 1f + random.nextFloat() * 1.5f, noisePaint)
            }
            return bmp
        }
    }
}

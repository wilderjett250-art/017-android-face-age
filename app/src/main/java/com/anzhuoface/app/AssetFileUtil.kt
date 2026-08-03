package com.anzhuoface.app

import android.content.Context
import java.io.File

object AssetFileUtil {
    fun copyAssetToFile(context: Context, assetName: String, targetDirName: String = "models"): File {
        val targetDir = File(context.filesDir, targetDirName)
        if (!targetDir.exists()) {
            targetDir.mkdirs()
        }

        val targetFile = File(targetDir, assetName)
        if (targetFile.exists() && targetFile.length() > 0L) {
            return targetFile
        }

        context.assets.open(assetName).use { input ->
            targetFile.outputStream().use { output ->
                input.copyTo(output)
            }
        }
        return targetFile
    }
}

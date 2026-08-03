package com.anzhuoface.app

data class HistoryEntry(
    val timestamp: Long,
    val source: String,
    val modelName: String,
    val faceCount: Int,
    val summary: String,
    val durationMs: Long,
    val imageWidth: Int,
    val imageHeight: Int,
    val success: Boolean
)

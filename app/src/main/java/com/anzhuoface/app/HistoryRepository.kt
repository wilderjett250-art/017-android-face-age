package com.anzhuoface.app

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject

class HistoryRepository(context: Context) {

    private val preferences = context.getSharedPreferences("analysis_history", Context.MODE_PRIVATE)

    fun load(): List<HistoryEntry> {
        val raw = preferences.getString(KEY_ENTRIES, "[]") ?: "[]"
        val array = JSONArray(raw)
        return buildList {
            for (index in 0 until array.length()) {
                val item = array.getJSONObject(index)
                add(
                    HistoryEntry(
                        timestamp = item.optLong("timestamp"),
                        source = item.optString("source"),
                        modelName = item.optString("modelName", "标准版模型"),
                        faceCount = item.optInt("faceCount"),
                        summary = item.optString("summary"),
                        durationMs = item.optLong("durationMs"),
                        imageWidth = item.optInt("imageWidth"),
                        imageHeight = item.optInt("imageHeight"),
                        success = item.optBoolean("success", true)
                    )
                )
            }
        }
    }

    fun save(entry: HistoryEntry) {
        val newList = listOf(entry) + load()
        val trimmed = newList.take(MAX_ENTRIES)
        val array = JSONArray()
        trimmed.forEach { item ->
            array.put(
                JSONObject().apply {
                    put("timestamp", item.timestamp)
                    put("source", item.source)
                    put("modelName", item.modelName)
                    put("faceCount", item.faceCount)
                    put("summary", item.summary)
                    put("durationMs", item.durationMs)
                    put("imageWidth", item.imageWidth)
                    put("imageHeight", item.imageHeight)
                    put("success", item.success)
                }
            )
        }
        preferences.edit().putString(KEY_ENTRIES, array.toString()).apply()
    }

    fun clear() {
        preferences.edit().remove(KEY_ENTRIES).apply()
    }

    companion object {
        private const val KEY_ENTRIES = "entries"
        private const val MAX_ENTRIES = 10
    }
}

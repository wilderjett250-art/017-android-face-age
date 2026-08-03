package com.anzhuoface.app

data class AgePrediction(
    val ageLabel: String,
    val estimatedAge: Int,
    val ageConfidence: Float,
    val genderLabel: String,
    val genderConfidence: Float
)

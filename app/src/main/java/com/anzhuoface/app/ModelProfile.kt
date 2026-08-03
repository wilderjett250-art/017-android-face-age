package com.anzhuoface.app

data class ModelProfile(
    val key: String,
    val displayName: String,
    val assetName: String,
    val inputSize: Int,
    val backbone: String,
    val trainingDataset: String,
    val trainingStrategy: String,
    val ageTop1: String,
    val ageOneOff: String,
    val ageMaeProxy: String,
    val cs5Proxy: String,
    val modelSize: String,
    val lightweightStrategy: String
)

object ModelProfiles {
    val standard = ModelProfile(
        key = "standard",
        displayName = "标准版模型",
        assetName = "fairface_multitask.onnx",
        inputSize = 160,
        backbone = "MobileNetV3 Large",
        trainingDataset = "FairFace（东亚增强采样）",
        trainingStrategy = "年龄分布损失 + 回归一致性约束 + 东亚/东南亚样本加权 + 性别分支增权",
        ageTop1 = "53.21%",
        ageOneOff = "93.92%",
        ageMaeProxy = "5.60",
        cs5Proxy = "54.81%",
        modelSize = "14.39 MB",
        lightweightStrategy = "工程交付主模型，精度优先，针对东亚样本做了增强采样"
    )

    val lite = ModelProfile(
        key = "lite",
        displayName = "轻量版模型",
        assetName = "fairface_multitask_lite.onnx",
        inputSize = 128,
        backbone = "MobileNetV3 Small",
        trainingDataset = "FairFace",
        trainingStrategy = "小主干网络 + 小输入尺寸 + 类别重加权采样 + 序数分布损失",
        ageTop1 = "49.72%",
        ageOneOff = "90.61%",
        ageMaeProxy = "6.44",
        cs5Proxy = "50.58%",
        modelSize = "5.44 MB",
        lightweightStrategy = "轻量化部署版本，体积更小，适合展示移动端部署路线"
    )

    val thesis = ModelProfile(
        key = "thesis",
        displayName = "论文复现版",
        assetName = "fairface_mobilenetv2_dex_distill.onnx",
        inputSize = 160,
        backbone = "MobileNetV2",
        trainingDataset = "FairFace（东亚增强采样）",
        trainingStrategy = "DEX风格年龄期望输出 + 年龄分布损失 + 回归一致性约束 + 教师蒸馏",
        ageTop1 = "54.57%",
        ageOneOff = "93.94%",
        ageMaeProxy = "5.55",
        cs5Proxy = "55.88%",
        modelSize = "12.03 MB",
        lightweightStrategy = "贴近论文方法线，保留 MobileNetV2 + DEX 风格叙述，同时用蒸馏提升效果"
    )

    val all = listOf(standard, lite, thesis)

    fun fromKey(key: String?): ModelProfile {
        return all.firstOrNull { it.key == key } ?: standard
    }
}

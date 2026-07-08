import com.android.build.api.artifact.SingleArtifact

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
}

android {
    namespace = "com.glucoedge.app"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.glucoedge.app"
        minSdk = 26
        targetSdk = 36
        versionCode = 1
        versionName = "0.1"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    buildFeatures { compose = true }
    sourceSets["main"].java.srcDirs("src/main/kotlin")
    sourceSets["test"].java.srcDirs("src/test/kotlin")
    sourceSets["androidTest"].java.srcDirs("src/androidTest/kotlin")

    // litertlm-android bundles its own copy of lib/{arm64-v8a,x86_64}/libLiteRt.so
    // (a different build than the one in the standalone `litert` artifact used by
    // TrendClassifier - different size/date/md5, confirmed by inspecting both AARs).
    // This is a known, currently-unresolved upstream conflict when combining LiteRT
    // and LiteRT-LM in one app: see google-ai-edge/LiteRT-LM#2351 (open, no official
    // guidance as of this writing). pickFirsts unblocks the build but does NOT prove
    // ABI compatibility between the two .so builds - this must be verified on-device
    // (both TrendClassifier accuracy and LiteRT-LM engine init) before this is trusted;
    // see the task report for details.
    packaging {
        jniLibs {
            pickFirsts += setOf(
                "lib/arm64-v8a/libLiteRt.so",
                "lib/x86_64/libLiteRt.so",
            )
        }
    }
}

dependencies {
    implementation(platform(libs.compose.bom))
    implementation(libs.androidx.activity.compose)
    implementation(libs.androidx.lifecycle.viewmodel.compose)
    implementation(libs.compose.material3)
    implementation(libs.compose.ui)
    implementation(libs.compose.ui.tooling.preview)
    implementation(libs.coroutines.core)
    implementation(libs.litert)
    implementation(libs.litertlm.android)
    testImplementation(libs.junit)
    testImplementation(libs.coroutines.test)
    androidTestImplementation(libs.androidx.test.runner)
    androidTestImplementation(libs.androidx.test.junit)
}

// Structural no-network guarantee: fail `check` if the MERGED manifest
// (ours plus every dependency's) requests INTERNET.
androidComponents {
    onVariants(selector().withName("debug")) { variant ->
        val manifest = variant.artifacts.get(SingleArtifact.MERGED_MANIFEST)
        val checkTask = project.tasks.register("checkNoInternetPermission") {
            inputs.file(manifest)
            doLast {
                val text = manifest.get().asFile.readText()
                check(!text.contains("android.permission.INTERNET")) {
                    "Merged manifest must not request INTERNET permission (no network calls for inference, ever)"
                }
            }
        }
        project.tasks.named("check") { dependsOn(checkTask) }
    }
}

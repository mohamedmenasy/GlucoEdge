import com.android.build.api.artifact.SingleArtifact
import org.gradle.api.artifacts.component.ModuleComponentIdentifier
import org.gradle.api.artifacts.type.ArtifactTypeDefinition
import java.security.MessageDigest
import java.util.zip.ZipFile

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
    // see the task report for details. Which .so pickFirsts actually keeps is an
    // artifact-resolution-order accident, not a guarantee - `checkNativeLibProvenance`
    // below fails the build if the packaged binary ever stops matching the litert AAR
    // that the on-device parity suite was verified against.
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

        // Native-lib provenance guard: the `packaging.jniLibs.pickFirsts` above resolves
        // a collision between `litert` and `litertlm-android`'s bundled `libLiteRt.so`
        // builds. Today the winner happens to be `litert`'s build - the one the on-device
        // parity suite was verified against - but pickFirsts winners are decided by
        // artifact-resolution order, not by any explicit rule; a dependency reorder,
        // version bump, or AGP upgrade could silently flip the winner with no build-time
        // signal. This task fails the build if that ever happens.
        val mergedNativeLibsDir = variant.artifacts.get(SingleArtifact.MERGED_NATIVE_LIBS)
        val litertAarFiles = project.configurations.getByName("debugRuntimeClasspath")
            .incoming.artifactView {
                attributes {
                    attribute(ArtifactTypeDefinition.ARTIFACT_TYPE_ATTRIBUTE, "aar")
                }
                componentFilter { id ->
                    id is ModuleComponentIdentifier &&
                        id.group == "com.google.ai.edge.litert" &&
                        id.module == "litert"
                }
            }.files
        val litertVersion = libs.versions.litert.get()
        val provenanceTask = project.tasks.register("checkNativeLibProvenance") {
            inputs.dir(mergedNativeLibsDir)
            inputs.files(litertAarFiles)
            doLast {
                fun sha256(bytes: ByteArray): String =
                    MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }

                val packagedSo = mergedNativeLibsDir.get().asFile.walkTopDown()
                    .firstOrNull { it.isFile && it.invariantSeparatorsPath.endsWith("arm64-v8a/libLiteRt.so") }
                    ?: error(
                        "Could not find a packaged arm64-v8a/libLiteRt.so under " +
                            "${mergedNativeLibsDir.get().asFile} - did the merged-native-libs output layout change?"
                    )
                val packagedHash = sha256(packagedSo.readBytes())

                val litertAar = litertAarFiles.singleFile
                val aarEntryPath = "jni/arm64-v8a/libLiteRt.so"
                val aarHash = ZipFile(litertAar).use { zip ->
                    val entry = zip.getEntry(aarEntryPath)
                        ?: error("litert AAR ($litertAar) has no entry $aarEntryPath")
                    sha256(zip.getInputStream(entry).readBytes())
                }

                check(packagedHash == aarHash) {
                    "packaged libLiteRt.so no longer comes from litert $litertVersion; the pickFirsts winner " +
                        "flipped - the classifier's device-verified parity no longer covers this binary " +
                        "(packaged sha256=$packagedHash from $packagedSo; litert $litertVersion AAR " +
                        "sha256=$aarHash from $litertAar)"
                }
                println(
                    "checkNativeLibProvenance: packaged arm64-v8a/libLiteRt.so sha256=$packagedHash " +
                        "matches litert $litertVersion AAR - device-verified parity still covers this binary."
                )
            }
        }
        project.tasks.named("check") { dependsOn(provenanceTask) }
    }
}

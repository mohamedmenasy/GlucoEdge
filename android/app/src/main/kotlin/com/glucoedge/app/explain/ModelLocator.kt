package com.glucoedge.app.explain

import java.io.File

/** Any *.litertlm in the app's external-files dir enables the feature. */
object ModelLocator {
    fun findModel(dir: File?): File? =
        dir?.listFiles { f -> f.isFile && f.name.endsWith(".litertlm") }
            ?.minByOrNull { it.name }
}

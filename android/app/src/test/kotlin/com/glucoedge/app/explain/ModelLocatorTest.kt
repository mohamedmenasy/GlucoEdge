package com.glucoedge.app.explain

import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class ModelLocatorTest {
    @get:Rule val tmp = TemporaryFolder()

    @Test fun nullDirYieldsNull() = assertNull(ModelLocator.findModel(null))

    @Test fun emptyDirYieldsNull() {
        assertNull(ModelLocator.findModel(tmp.root))
    }

    @Test fun ignoresOtherExtensions() {
        tmp.newFile("model.tflite"); tmp.newFile("notes.txt")
        assertNull(ModelLocator.findModel(tmp.root))
    }

    @Test fun findsLitertlmFile() {
        val f = tmp.newFile("gemma3-1b-it-int4.litertlm")
        assertEquals(f, ModelLocator.findModel(tmp.root))
    }

    @Test fun picksFirstAlphabeticallyWhenSeveral() {
        tmp.newFile("b.litertlm")
        val a = tmp.newFile("a.litertlm")
        assertEquals(a, ModelLocator.findModel(tmp.root))
    }
}

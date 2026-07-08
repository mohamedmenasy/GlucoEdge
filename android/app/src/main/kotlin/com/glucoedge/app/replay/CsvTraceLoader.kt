package com.glucoedge.app.replay

import java.io.Reader
import java.time.LocalDateTime
import java.time.ZoneOffset
import java.time.format.DateTimeFormatter

/** Parses `time,gl` CSV (ISO-8601 minute timestamps, mg/dL values). */
object CsvTraceLoader {
    private val format = DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm")

    fun load(reader: Reader): TraceLoadResult {
        val readings = mutableListOf<Reading>()
        var skipped = 0
        reader.buffered().useLines { lines ->
            lines.drop(1).forEach { line ->
                if (line.isBlank()) return@forEach
                val parts = line.split(',')
                val reading = runCatching {
                    val t = LocalDateTime.parse(parts[0].trim(), format)
                    Reading(t.toEpochSecond(ZoneOffset.UTC) / 60, parts[1].trim().toFloat())
                }.getOrNull()
                if (reading != null) readings.add(reading) else skipped++
            }
        }
        return TraceLoadResult(readings, skipped)
    }
}

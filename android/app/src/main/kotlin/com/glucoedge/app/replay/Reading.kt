package com.glucoedge.app.replay

/** One CGM reading. Time is epoch minutes so gap math is exact integer arithmetic. */
data class Reading(val epochMinutes: Long, val mgdl: Float)

data class TraceLoadResult(val readings: List<Reading>, val skippedRows: Int)

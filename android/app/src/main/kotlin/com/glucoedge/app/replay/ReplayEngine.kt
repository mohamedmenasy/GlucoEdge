package com.glucoedge.app.replay

import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.flow

enum class Speed(val multiplier: Int) { X1(1), X4(4), X16(16) }

/**
 * Replays a fixed trace on a virtual clock: one reading per interval while
 * playing. Decoupled from wall time - the real 5-minute CGM cadence is
 * compressed to [baseIntervalMs] per reading at 1x.
 */
class ReplayEngine(
    private val readings: List<Reading>,
    private val baseIntervalMs: Long = 5_000L,
) {
    private val _isPlaying = MutableStateFlow(false)
    private val _speed = MutableStateFlow(Speed.X1)
    val isPlaying: StateFlow<Boolean> = _isPlaying.asStateFlow()
    val speed: StateFlow<Speed> = _speed.asStateFlow()

    val events: Flow<Reading> = flow {
        for (reading in readings) {
            _isPlaying.first { it }                       // suspend until playing
            delay(baseIntervalMs / _speed.value.multiplier)
            _isPlaying.first { it }                       // re-check after delay
            emit(reading)
        }
    }

    fun play() { _isPlaying.value = true }
    fun pause() { _isPlaying.value = false }
    fun setSpeed(speed: Speed) { _speed.value = speed }
}

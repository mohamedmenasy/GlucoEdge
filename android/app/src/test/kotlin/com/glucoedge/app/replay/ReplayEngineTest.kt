package com.glucoedge.app.replay

import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.launch
import kotlinx.coroutines.test.advanceTimeBy
import kotlinx.coroutines.test.runCurrent
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Test

class ReplayEngineTest {
    private val trace = (0L..9L).map { Reading(it * 5, 100f + it) }

    @Test fun emitsNothingWhilePaused() = runTest {
        val engine = ReplayEngine(trace)
        val seen = mutableListOf<Reading>()
        val job = launch { engine.events.toList(seen) }
        advanceTimeBy(60_000); runCurrent()
        assertEquals(0, seen.size)
        job.cancel()
    }

    @Test fun emitsOnePerIntervalAtX1() = runTest {
        val engine = ReplayEngine(trace)
        val seen = mutableListOf<Reading>()
        val job = launch { engine.events.toList(seen) }
        engine.play()
        advanceTimeBy(15_000); runCurrent()   // 3 intervals of 5 s
        assertEquals(3, seen.size)
        assertEquals(trace[0], seen[0])
        job.cancel()
    }

    @Test fun speedChangeShortensInterval() = runTest {
        val engine = ReplayEngine(trace)
        val seen = mutableListOf<Reading>()
        val job = launch { engine.events.toList(seen) }
        engine.setSpeed(Speed.X16)            // 5000/16 = 312.5 -> 312 ms
        engine.play()
        advanceTimeBy(3_120); runCurrent()    // 10 intervals: whole trace
        assertEquals(10, seen.size)
        job.cancel()
    }

    @Test fun pauseStopsEmission() = runTest {
        val engine = ReplayEngine(trace)
        val seen = mutableListOf<Reading>()
        val job = launch { engine.events.toList(seen) }
        engine.play()
        advanceTimeBy(10_000); runCurrent()
        engine.pause()
        advanceTimeBy(60_000); runCurrent()
        assertEquals(2, seen.size)
        job.cancel()
    }
}

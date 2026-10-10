/// One CGM reading. Time is epoch minutes so gap math is exact integer arithmetic.
struct Reading: Equatable, Sendable {
    let epochMinutes: Int64
    let mgdl: Float
}

struct TraceLoadResult: Equatable {
    let readings: [Reading]
    let skippedRows: Int
}

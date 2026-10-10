import Foundation

/// Parses `time,gl` CSV (ISO-8601 minute timestamps, mg/dL values).
enum CsvTraceLoader {
    private static let formatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "yyyy-MM-dd'T'HH:mm"
        f.locale = Locale(identifier: "en_US_POSIX")
        f.timeZone = TimeZone(identifier: "UTC")
        return f
    }()

    static func load(_ text: String) -> TraceLoadResult {
        var readings: [Reading] = []
        var skipped = 0
        let lines = text.split(whereSeparator: \.isNewline)   // handles \n and \r\n
        for line in lines.dropFirst() {
            // trim whitespacesAndNewlines, not .whitespaces: a stray \r from a
            // CRLF file would otherwise poison the gl field and skip every row.
            let parts = line.split(separator: ",", omittingEmptySubsequences: false)
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            if parts.count >= 2,
               let date = formatter.date(from: parts[0]),
               let gl = Float(parts[1]) {
                readings.append(Reading(epochMinutes: Int64(date.timeIntervalSince1970) / 60, mgdl: gl))
            } else if !line.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                skipped += 1
            }
        }
        return TraceLoadResult(readings: readings, skippedRows: skipped)
    }
}

import Charts
import SwiftUI

private let trendArrows = ["↓↓", "↓", "→", "↑", "↑↑"]

#if targetEnvironment(simulator)
private let deviceLabel = "simulator"
#else
private let deviceLabel = "device"
#endif

struct MainView: View {
    let viewModel: MainViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            if let error = viewModel.errorMessage {
                Text(error).foregroundStyle(.red)
            }

            chart

            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Text(viewModel.currentMgdl.map { String(format: "%.0f mg/dL", $0) } ?? "-- mg/dL")
                    .font(.title)
                Text("(\(viewModel.traceLabel) trace)").font(.caption)
            }

            if viewModel.rebuffering {
                Text("window rebuffering…").font(.caption)
            }

            if let p = viewModel.prediction {
                Text("\(trendArrows[p.classIndex])  \(TrendClassifier.classNames[p.classIndex])")
                    .font(.largeTitle)
                Text("predicted trend, next 15 min").font(.caption)
                ForEach(Array(TrendClassifier.classNames.enumerated()), id: \.offset) { i, name in
                    HStack {
                        Text(name).font(.caption).frame(width: 96, alignment: .leading)
                        ProgressView(value: Double(p.probabilities[i]))
                    }
                }
            }

            Text(statsLine).font(.caption)

            HStack(spacing: 8) {
                Button(viewModel.playing ? "Pause" : "Play") { viewModel.onPlayPause() }
                    .buttonStyle(.borderedProminent)
                ForEach(Speed.allCases, id: \.rawValue) { s in
                    Button(s == viewModel.speed ? "[\(s.rawValue)×]" : "\(s.rawValue)×") {
                        viewModel.onSpeed(s)
                    }
                    .buttonStyle(.bordered)
                }
            }

            Button(viewModel.model == .float ? "Switch to INT8 model" : "Switch to float model") {
                viewModel.onToggleModel()
            }
            .buttonStyle(.bordered)

            explainSection

            Spacer()
            Text("Portfolio demo on public research data — not a medical device, not treatment guidance.")
                .font(.caption2)
        }
        .padding(16)
    }

    private var chart: some View {
        Chart(Array(viewModel.recentReadings.enumerated()), id: \.offset) { index, value in
            LineMark(x: .value("reading", index), y: .value("mg/dL", value))
        }
        .chartYScale(domain: chartDomain)
        .frame(height: 140)
    }

    private var chartDomain: ClosedRange<Double> {
        let readings = viewModel.recentReadings
        guard let low = readings.min(), let high = readings.max() else { return 40...410 }
        return Double(max(low - 10, 40))...Double(min(high + 10, 410))
    }

    private var statsLine: String {
        let s = viewModel.stats
        return String(format: "model: %@  ·  engine: %@  ·  inferences: %d  ·  mean %.3f ms  ·  p95 %.3f ms  ·  skipped rows: %d  ·  %@",
                      viewModel.model == .float ? "float" : "int8",
                      viewModel.engineLabel, s.count, s.meanMs, s.p95Ms, viewModel.skippedRows, deviceLabel)
    }

    @ViewBuilder
    private var explainSection: some View {
        switch viewModel.explainerState {
        case .unavailable(let reason):
            Text(reason).font(.caption).foregroundStyle(.secondary)
        case .ready, .generating, .note, .error:
            Button("Explain") { viewModel.onExplain() }
                .buttonStyle(.bordered)
                .disabled(viewModel.explainerState == .generating)
            switch viewModel.explainerState {
            case .generating:
                Text("generating…").font(.caption)
            case .note(let text):
                VStack(alignment: .leading, spacing: 4) {
                    Text("On-device demo note — not medical guidance.").font(.caption2)
                    Text(text).font(.body)
                }
            case .error(let message):
                Text(message).font(.caption).foregroundStyle(.red)
            default:
                EmptyView()
            }
        }
    }
}

import SwiftUI

@main
struct GlucoEdgeApp: App {
    @State private var viewModel = GlucoEdgeApp.makeViewModel()
    @Environment(\.scenePhase) private var scenePhase

    var body: some Scene {
        WindowGroup {
            MainView(viewModel: viewModel)
        }
        .onChange(of: scenePhase) { _, phase in
            // Android's onResume check: Apple Intelligence may have been
            // enabled (or its model finished downloading) while backgrounded.
            if phase == .active {
                viewModel.refreshExplainAvailability(FoundationModelsNoteGenerator.currentAvailability())
            }
        }
    }

    @MainActor
    private static func makeViewModel() -> MainViewModel {
        // Prefer the (gitignored, local-only) real trace, else the committed
        // synthetic one — same preference order as the Android app.
        let trace: TraceSource
        if let text = assetText("real_trace", "csv") {
            let loaded = CsvTraceLoader.load(text)
            trace = TraceSource(readings: loaded.readings, skippedRows: loaded.skippedRows, label: "real")
        } else if let text = assetText("synthetic_trace", "csv") {
            let loaded = CsvTraceLoader.load(text)
            trace = TraceSource(readings: loaded.readings, skippedRows: loaded.skippedRows, label: "synthetic")
        } else {
            fatalError("no trace CSV in bundle assets/ — the android assets folder reference broke")
        }

        return MainViewModel(
            trace: trace,
            classifierFactory: { model in
                guard let url = Bundle.main.url(forResource: model.resourceName,
                                                withExtension: "tflite", subdirectory: "assets") else {
                    // Spec: a missing bundled model fails loud at launch.
                    fatalError("\(model.resourceName).tflite missing from bundle assets/")
                }
                return try TrendClassifier(modelURL: url, quantized: model == .int8)
            },
            explainAvailability: FoundationModelsNoteGenerator.currentAvailability(),
            noteGenerator: FoundationModelsNoteGenerator())
    }

    private static func assetText(_ name: String, _ ext: String) -> String? {
        guard let url = Bundle.main.url(forResource: name, withExtension: ext, subdirectory: "assets") else {
            return nil
        }
        return try? String(contentsOf: url, encoding: .utf8)
    }
}

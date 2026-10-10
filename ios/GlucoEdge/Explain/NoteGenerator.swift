/// Main-actor isolated, like Classifier: only the main-actor view model calls it.
@MainActor
protocol NoteGenerator {
    /// Returns the finished note text, trimmed. Throws NoteError with a
    /// user-readable message on failure.
    func generate(_ prompt: String) async throws -> String
}

struct NoteError: Error, Equatable {
    let message: String
}

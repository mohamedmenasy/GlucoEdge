enum ExplainAvailability: Equatable {
    case available
    /// Short status line explaining why Explain is off (shown in the UI).
    case unavailable(String)
}

enum ExplainerState: Equatable {
    case unavailable(String)
    case ready
    case generating
    case note(String)
    case error(String)
}

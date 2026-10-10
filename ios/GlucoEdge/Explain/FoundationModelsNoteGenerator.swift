import FoundationModels

/// Writes the Explain note with Apple's on-device model (SystemLanguageModel
/// only — never the Private Cloud Compute model, so no network). Every note
/// gets a fresh session, like Android's per-note conversation: a reused
/// session would carry earlier prompts and notes into the next one and, after
/// ~16 notes, exhaust the model's context window. Instructions carry the
/// standing no-advice rules so a prompt-injection-ish trace label can't
/// override them.
@MainActor
final class FoundationModelsNoteGenerator: NoteGenerator {
    private static let instructions = """
        You write short captions for a demo app screen that replays recorded \
        glucose data. Describe only the replayed pattern in 2-3 plain sentences. \
        Never give advice, recommendations, or dosing. Never address the reader \
        as a patient. Never mention insulin or treatment.
        """

    static func currentAvailability() -> ExplainAvailability {
        switch SystemLanguageModel.default.availability {
        case .available:
            return .available
        case .unavailable(.deviceNotEligible):
            return .unavailable("Explain needs Apple Intelligence, which this device does not support.")
        case .unavailable(.appleIntelligenceNotEnabled):
            return .unavailable("Explain needs Apple Intelligence — enable it in Settings.")
        case .unavailable(.modelNotReady):
            return .unavailable("Apple Intelligence model is still getting ready — Explain unavailable.")
        case .unavailable:
            return .unavailable("Explain is unavailable on this device.")
        }
    }

    func generate(_ prompt: String) async throws -> String {
        let session = LanguageModelSession(instructions: Self.instructions)
        do {
            let response = try await session.respond(
                to: prompt,
                options: GenerationOptions(temperature: 0.2, maximumResponseTokens: 200))
            return response.content.trimmingCharacters(in: .whitespacesAndNewlines)
        } catch let error as LanguageModelSession.GenerationError {
            throw NoteError(message: Self.message(for: error))
        } catch {
            // Some failures arrive as bridged NSErrors that don't cast to
            // GenerationError (seen on the simulator: guardrail model assets
            // missing, GenerationError code -1). Keep the NoteError contract.
            throw NoteError(message: "Note generation failed: \(error.localizedDescription)")
        }
    }

    private static func message(for error: LanguageModelSession.GenerationError) -> String {
        switch error {
        case .guardrailViolation:
            return "Apple's safety filter declined this note."
        case .refusal:
            return "The model declined to write this note."
        case .exceededContextWindowSize:
            return "Note request was too long for the on-device model."
        case .rateLimited:
            return "The on-device model is busy — try again in a moment."
        case .assetsUnavailable:
            return "Apple Intelligence model assets are unavailable right now."
        default:
            return "Note generation failed: \(error.localizedDescription)"
        }
    }
}

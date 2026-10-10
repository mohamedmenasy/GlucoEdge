import CryptoKit
import Foundation
import Testing

/// Anchor class for locating the test bundle (Swift Testing has no XCTestCase).
final class TestBundleLocator {}

struct SmokeTests {
    @Test func appBundleContainsSharedAssets() throws {
        // The app target bundles android/app/src/main/assets as a folder
        // reference; if this fails, the cross-directory reference broke.
        for name in ["trend_float", "trend_int8"] {
            #expect(Bundle.main.url(forResource: name, withExtension: "tflite", subdirectory: "assets") != nil,
                    "\(name).tflite missing from app bundle assets/")
        }
        #expect(Bundle.main.url(forResource: "synthetic_trace", withExtension: "csv", subdirectory: "assets") != nil)
    }

    @Test func testBundleContainsGoldenVectors() throws {
        let bundle = Bundle(for: TestBundleLocator.self)
        #expect(bundle.url(forResource: "golden_vectors", withExtension: "json", subdirectory: "assets") != nil,
                "golden_vectors.json missing from test bundle assets/")
    }
}

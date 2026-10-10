import Testing
@testable import GlucoEdge

@MainActor
struct MainViewTests {
    @Test func chartDomainPadsAndClampsNormalReadings() {
        #expect(MainView.chartDomain([100, 120]) == 90...130)
        #expect(MainView.chartDomain([35, 405]) == 40...410)
        #expect(MainView.chartDomain([]) == 40...410)
    }

    @Test func chartDomainStaysValidWhenReadingsSitOutsideTheClamp() {
        // All readings above 420 (or below 30) made the clamped lower bound
        // exceed the upper one — a ClosedRange precondition crash. A sensor
        // that reports up to 500 mg/dL would hit this.
        let high = MainView.chartDomain([450, 460])
        #expect(high.lowerBound < high.upperBound)
        #expect(high.contains(450) && high.contains(460))
        let low = MainView.chartDomain([20, 25])
        #expect(low.lowerBound < low.upperBound)
        #expect(low.contains(20) && low.contains(25))
    }
}

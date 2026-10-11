import MWDATCore
import SwiftUI

@main
struct BrainRotApp: App {
    @State private var model: AppModel
    @Environment(\.scenePhase) private var scenePhase

    init() {
        do {
            try Wearables.configure()
        } catch {
            print("[BrainRot] Wearables.configure failed: \(error)")
        }
        _model = State(wrappedValue: AppModel())
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(model)
                .onAppear { model.boot() }
                .onChange(of: scenePhase) { _, phase in
                    if phase == .active { model.didBecomeActive() }
                }
                .onOpenURL { url in
                    guard let items = URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems,
                          items.contains(where: { $0.name == "metaWearablesAction" }) else { return }
                    model.glasses.handleURL(url)
                }
        }
    }
}

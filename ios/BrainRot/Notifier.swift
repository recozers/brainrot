import UserNotifications

/// Local notifications only — the app stays alive via background audio, so no push server.
final class Notifier: NSObject, UNUserNotificationCenterDelegate {
    static let shared = Notifier()
    private let center = UNUserNotificationCenter.current()
    private let deadManID = "brainrot.deadman"

    func setUp() {
        center.delegate = self
    }

    func requestPermission() async -> Bool {
        (try? await center.requestAuthorization(options: [.alert, .sound, .badge])) ?? false
    }

    func roast(_ body: String, region: Region?) {
        let content = UNMutableNotificationContent()
        content.title = region.map { "🧠 \($0.name)" } ?? "🧠 BrainRot"
        content.body = body
        content.sound = .default
        content.interruptionLevel = .timeSensitive
        center.add(UNNotificationRequest(identifier: UUID().uuidString, content: content, trigger: nil))
    }

    /// Re-armed after every capture. If iOS kills the app, this is what fires.
    func armDeadMan(after seconds: TimeInterval) {
        center.removePendingNotificationRequests(withIdentifiers: [deadManID])
        let content = UNMutableNotificationContent()
        content.title = "🧠 BrainRot stopped watching"
        content.body = "You killed me. Your brain is now rotting unsupervised. Open the app."
        content.sound = .default
        let trigger = UNTimeIntervalNotificationTrigger(timeInterval: max(60, seconds), repeats: false)
        center.add(UNNotificationRequest(identifier: deadManID, content: content, trigger: trigger))
    }

    func disarmDeadMan() {
        center.removePendingNotificationRequests(withIdentifiers: [deadManID])
    }

    func userNotificationCenter(_ center: UNUserNotificationCenter, willPresent notification: UNNotification) async -> UNNotificationPresentationOptions {
        [.banner, .sound]
    }
}

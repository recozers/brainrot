import Observation
import SwiftUI
import UIKit

@Observable
@MainActor
final class AppModel {
    let glasses = GlassesManager()
    @ObservationIgnored let audio = AudioKeeper()

    // Live state
    var tracking = false
    var capturing = false
    var live: [Region: Double] = Dictionary(uniqueKeysWithValues: Region.allCases.map { ($0, 0.3) })
    var lastRoast: String?
    var lastRoastRegion: Region?
    var lastCaptureAt: Date?
    var nextCaptureAt: Date?
    var captureCount = 0
    var healPulse: [Region: Date] = [:]
    var logLines: [String] = []

    // Replay
    var timeline: [CaptureResult] = []
    var replayIndex: Int?
    var playing = false

    // Local thumbnails for captures taken this session
    var thumbs: [String: UIImage] = [:]

    // Settings
    static let defaultURL = Bundle.main.object(forInfoDictionaryKey: "BrainRotBackendURL") as? String ?? ""
    var baseURL = (UserDefaults.standard.string(forKey: "baseURL").flatMap { $0.isEmpty ? nil : $0 }) ?? AppModel.defaultURL {
        didSet { UserDefaults.standard.set(baseURL, forKey: "baseURL") }
    }
    var intervalMinutes = UserDefaults.standard.object(forKey: "intervalMinutes") as? Double ?? 3 {
        didSet { UserDefaults.standard.set(intervalMinutes, forKey: "intervalMinutes") }
    }

    @ObservationIgnored private var loopTask: Task<Void, Never>?
    @ObservationIgnored private var loopGeneration = 0
    @ObservationIgnored private var lastReconnectNag: Date = .distantPast
    @ObservationIgnored private var playTask: Task<Void, Never>?

    var api: BrainAPI { BrainAPI(baseURL: baseURL) }

    /// What the brain should show right now: a replay snapshot, or live.
    var displayed: [Region: Double] {
        if let i = replayIndex, timeline.indices.contains(i) {
            let item = timeline[i]
            return Dictionary(uniqueKeysWithValues: Region.allCases.map { ($0, item.rot($0)) })
        }
        return live
    }

    var displayedRoast: (String, Region?)? {
        if let i = replayIndex, timeline.indices.contains(i) {
            // Most recent roast at or before the scrubbed capture.
            for item in timeline[...i].reversed() {
                if let r = item.roast { return (r, item.roast_region.flatMap(Region.init(rawValue:))) }
            }
            return nil
        }
        return lastRoast.map { ($0, lastRoastRegion) }
    }

    var overallRot: Double {
        let values = displayed.values
        return values.reduce(0, +) / Double(max(1, values.count))
    }

    init() {
        glasses.onLog = { [weak self] in self?.log($0) }
        audio.onLog = { [weak self] msg in Task { @MainActor in self?.log(msg) } }
    }

    func boot() {
        Notifier.shared.setUp()
        if ProcessInfo.processInfo.arguments.contains("-demoRot") {
            for (r, v) in zip(Region.allCases, [0.1, 0.5, 0.72, 0.95, 0.3, 0.9]) { live[r] = v }
            lastRoast = "Your parahippocampal gyrus has seen more drywall than daylight. Go find a tree, you houseplant."
            lastRoastRegion = .naturePlaces
        }
        glasses.start()
        Task { await refreshTimeline() }
    }

    func log(_ message: String) {
        let stamp = Date().formatted(date: .omitted, time: .standard)
        logLines.append("\(stamp)  \(message)")
        if logLines.count > 200 { logLines.removeFirst(logLines.count - 200) }
        print("[BrainRot] \(message)")
    }

    // MARK: - Tracking loop

    func toggleTracking() {
        tracking ? stopTracking() : startTracking()
    }

    func startTracking() {
        guard !tracking else { return }
        Task {
            _ = await Notifier.shared.requestPermission()
            guard await audio.requestPermission() else {
                log("Microphone permission denied — background capture won't survive")
                return
            }
            do {
                try audio.start()
            } catch {
                log("Audio start failed: \(error.localizedDescription)")
                return
            }
            tracking = true
            replayIndex = nil
            log("Tracking started, every \(Int(intervalMinutes)) min")
            await openStreamIfNeeded()
            loopGeneration += 1
            let generation = loopGeneration
            loopTask = Task { [weak self] in await self?.runLoop(generation: generation) }
        }
    }

    func stopTracking() {
        tracking = false
        // Don't cancel the task: an in-flight upload should still land. The loop exits on its own.
        loopGeneration += 1
        loopTask = nil
        nextCaptureAt = nil
        audio.stop()
        glasses.endSession()
        Notifier.shared.disarmDeadMan()
        log("Tracking stopped")
    }

    /// The glasses only start a stream while we're in the foreground; once open it survives a lock.
    func openStreamIfNeeded() async {
        guard glasses.continuousStream, !glasses.isStreaming,
              UIApplication.shared.applicationState != .background else { return }
        do {
            try await glasses.openStream()
            log("Glasses stream open — safe to lock the phone")
        } catch {
            log("Couldn't open glasses stream: \(error.localizedDescription)")
        }
    }

    func didBecomeActive() {
        guard tracking else { return }
        Task { await openStreamIfNeeded() }
    }

    private func runLoop(generation: Int) async {
        // Give the mic a few seconds to fill the buffer before the first capture.
        try? await Task.sleep(for: .seconds(5))
        while tracking && generation == loopGeneration {
            let interval = intervalMinutes * 60
            Notifier.shared.armDeadMan(after: interval * 3 + 60)
            await captureOnce()
            nextCaptureAt = Date().addingTimeInterval(interval)
            try? await Task.sleep(for: .seconds(interval))
        }
    }

    func captureOnce() async {
        guard !capturing else { return }
        capturing = true
        defer { capturing = false }
        let captureID = UUID().uuidString
        let timestamp = Date()
        let state = UIApplication.shared.applicationState == .background ? "bg" : "fg"

        var frames: [Data] = []
        if state == "bg", glasses.continuousStream, !glasses.isStreaming {
            log("[bg] Glasses stream dropped — audio only until you open the app")
            if Date().timeIntervalSince(lastReconnectNag) > 600 {
                lastReconnectNag = Date()
                Notifier.shared.roast("Your glasses stopped watching. Tap me so I can keep judging you.", region: nil)
            }
        } else { do {
            frames = try await glasses.captureClip()
            log("[\(state)] Got \(frames.count) frames")
        } catch {
            log("[\(state)] Glasses capture failed: \(error.localizedDescription) — sending audio only")
        } }
        let wav = audio.snapshotWAV(seconds: 15)
        if let mid = frames[safe: frames.count / 2], let img = UIImage(data: mid) {
            thumbs[captureID] = img
        }

        // capture_id makes uploads idempotent server-side, so retrying is safe.
        for attempt in 1...3 {
            do {
                let result = try await api.upload(captureID: captureID, timestamp: timestamp, frames: frames, fps: 7, audio: wav, battery: glasses.battery)
                apply(result)
                return
            } catch {
                log("Upload attempt \(attempt) failed: \(error.localizedDescription)")
                if attempt < 3 { try? await Task.sleep(for: .seconds(5)) }
            }
        }
    }

    private func apply(_ result: CaptureResult) {
        captureCount += 1
        lastCaptureAt = Date()
        for region in Region.allCases {
            guard let score = result.regions[region.rawValue] else { continue }
            if let old = live[region], old - score.rot >= 0.1 { healPulse[region] = Date() }
            live[region] = score.rot
        }
        timeline.append(result)
        if let roast = result.roast {
            let region = result.roast_region.flatMap(Region.init(rawValue:))
            lastRoast = roast
            lastRoastRegion = region
            Notifier.shared.roast(roast, region: region)
        }
        let summary = Region.allCases.map { "\($0.rawValue.prefix(4)) \(Int((live[$0] ?? 0) * 100))" }.joined(separator: " ")
        log("Rot: \(summary)")
    }

    // MARK: - Replay

    func refreshTimeline() async {
        guard !baseURL.isEmpty else { return }
        do {
            let items = try await api.timeline()
            timeline = items
            Task { await prefetchThumbs() }
            if let last = items.last, !tracking {
                for region in Region.allCases { live[region] = last.rot(region) }
                if let roast = items.last(where: { $0.roast != nil }) {
                    lastRoast = roast.roast
                    lastRoastRegion = roast.roast_region.flatMap(Region.init(rawValue:))
                }
            }
        } catch {
            log("Timeline: \(error.localizedDescription)")
        }
    }

    /// The capture currently on screen (scrubbed, or the latest one live).
    var displayedCapture: CaptureResult? {
        if let i = replayIndex { return timeline[safe: i] }
        return timeline.last
    }

    /// Thumbnails are tiny; fetch them all up front so scrubbing and playback never flicker.
    private func prefetchThumbs() async {
        for item in timeline where thumbs[item.capture_id] == nil {
            guard item.thumb_url != nil, let url = thumbURL(for: item),
                  let (data, _) = try? await URLSession.shared.data(from: url),
                  let image = UIImage(data: data) else { continue }
            thumbs[item.capture_id] = image
        }
    }

    /// Jump the replay to capture `i`; the last capture means live. Stepping forward over a
    /// heal fires the green pulse, so dragging through the day shows the brain recovering.
    func scrub(to i: Int) {
        playing = false
        playTask?.cancel()
        let from = replayIndex ?? timeline.count - 1
        if i == from + 1, let a = timeline[safe: from], let b = timeline[safe: i] {
            for region in Region.allCases where a.rot(region) - b.rot(region) >= 0.1 {
                healPulse[region] = Date()
            }
        }
        replayIndex = i >= timeline.count - 1 ? nil : i
    }

    func togglePlay() {
        if playing {
            playing = false
            playTask?.cancel()
            return
        }
        guard timeline.count > 1 else { return }
        playing = true
        if replayIndex == nil || replayIndex == timeline.count - 1 { replayIndex = 0 }
        playTask = Task { [weak self] in
            while let self, self.playing, let i = self.replayIndex, i < self.timeline.count - 1 {
                try? await Task.sleep(for: .milliseconds(700))
                guard self.playing else { return }
                let next = i + 1
                for region in Region.allCases where self.timeline[i].rot(region) - self.timeline[next].rot(region) >= 0.1 {
                    self.healPulse[region] = Date()
                }
                self.replayIndex = next
            }
            self?.playing = false
        }
    }

    func goLive() {
        playing = false
        playTask?.cancel()
        replayIndex = nil
    }

    func resetBackend() {
        Task {
            do {
                try await api.reset()
                timeline = []
                thumbs = [:]
                for r in Region.allCases { live[r] = 0.3 }
                lastRoast = nil
                log("Backend reset")
            } catch {
                log("Reset failed: \(error.localizedDescription)")
            }
        }
    }

    func thumbURL(for item: CaptureResult) -> URL? {
        item.thumb_url.flatMap { api.url($0) } ?? api.url("thumb/\(item.capture_id)")
    }

    func latestCapture(for region: Region) -> CaptureResult? {
        let upto = replayIndex.map { Array(timeline.prefix($0 + 1)) } ?? timeline
        return upto.last(where: { $0.roast_region == region.rawValue }) ?? upto.last
    }
}

extension Array {
    subscript(safe i: Int) -> Element? { indices.contains(i) ? self[i] : nil }
}

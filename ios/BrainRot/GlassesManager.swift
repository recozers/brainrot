import MWDATCamera
import MWDATCore
import Observation
import UIKit

enum GlassesError: LocalizedError {
    case noSession, sessionTimeout, cameraUnavailable, streamTimeout, noFrames, permission

    var errorDescription: String? {
        switch self {
        case .noSession: "Couldn't create a glasses session"
        case .sessionTimeout: "Glasses session didn't start"
        case .cameraUnavailable: "Glasses camera unavailable"
        case .streamTimeout: "Glasses stream didn't start"
        case .noFrames: "No frames arrived from the glasses"
        case .permission: "Camera permission not granted in Meta AI"
        }
    }
}

/// Thread-safe sink for frames arriving on the SDK's callback queue. Downsamples to `fps`
/// and JPEG-encodes only while a capture window is open.
final class FrameCollector: @unchecked Sendable {
    private let lock = NSLock()
    private var collecting = false
    private var frames: [Data] = []
    private var lastKept: CFAbsoluteTime = 0
    private var minGap: CFAbsoluteTime = 1.0 / 7.0
    private(set) var framesSeen = 0

    func begin(fps: Double) {
        lock.lock()
        frames = []
        lastKept = 0
        minGap = 1.0 / fps
        collecting = true
        lock.unlock()
    }

    func end() -> [Data] {
        lock.lock()
        collecting = false
        let out = frames
        frames = []
        lock.unlock()
        return out
    }

    func offer(_ frame: VideoFrame) {
        lock.lock()
        framesSeen += 1
        let now = CFAbsoluteTimeGetCurrent()
        guard collecting, now - lastKept >= minGap else { lock.unlock(); return }
        lastKept = now
        lock.unlock()
        guard let jpeg = frame.makeUIImage()?.jpegData(compressionQuality: 0.6) else { return }
        lock.lock()
        if collecting { frames.append(jpeg) }
        lock.unlock()
    }
}

@Observable
@MainActor
final class GlassesManager {
    var registration: RegistrationState = .unavailable
    var hasDevice = false
    var sessionState: DeviceSessionState = .idle
    var streamState: StreamState = .stopped
    var battery: Int?
    var donned: Bool?
    var cameraGranted = false
    var link = "unknown"
    var compatibility = "unknown"
    var knownDevices = 0
    var lastError: String?
    /// Keep the camera stream open between captures. Costs battery, but is the fallback if
    /// starting a stream from the background turns out not to work.
    var continuousStream = UserDefaults.standard.object(forKey: "continuousStream2") as? Bool ?? true {
        didSet { UserDefaults.standard.set(continuousStream, forKey: "continuousStream2") }
    }

    @ObservationIgnored private let wearables: WearablesInterface = Wearables.shared
    @ObservationIgnored private lazy var selector = AutoDeviceSelector(wearables: wearables)
    @ObservationIgnored private var session: DeviceSession?
    @ObservationIgnored private var camera: MWDATCamera.Camera?
    @ObservationIgnored private let sessionBag = ListenerTokenBag()
    @ObservationIgnored private let streamBag = ListenerTokenBag()
    @ObservationIgnored private var deviceToken: (any AnyListenerToken)?
    @ObservationIgnored private var watchedDevice: DeviceIdentifier?
    @ObservationIgnored let collector = FrameCollector()
    @ObservationIgnored var onLog: ((String) -> Void)?

    var isRegistered: Bool { registration == .registered }

    func start() {
        registration = wearables.registrationState
        Task {
            for await state in wearables.registrationStateStream() {
                registration = state
                if state == .registered { await refreshPermission() }
            }
        }
        Task {
            for await ids in wearables.devicesStream() {
                knownDevices = ids.count
                if let first = ids.first { watchDevice(first) }
            }
        }
        Task {
            for await id in selector.activeDeviceStream() {
                hasDevice = id != nil
                if let id { watchDevice(id) }
            }
        }
    }

    private func watchDevice(_ id: DeviceIdentifier) {
        guard watchedDevice != id, let device = wearables.deviceForIdentifier(id) else { return }
        watchedDevice = id
        deviceToken = device.addDeviceStateListener { [weak self] state in
            Task { @MainActor in
                self?.battery = state.batteryLevel
                self?.donned = state.donState == .unknown ? nil : state.donState == .donned
                self?.link = String(describing: state.linkState)
                self?.compatibility = String(describing: state.compatibility)
            }
        }
    }

    func handleURL(_ url: URL) {
        Task {
            do { _ = try await wearables.handleUrl(url) } catch { log("handleUrl: \(error.localizedDescription)") }
        }
    }

    func connect() {
        Task {
            do { try await wearables.startRegistration() } catch { log("Registration: \(error.localizedDescription)") }
        }
    }

    func openFirmwareUpdate() {
        Task {
            do { try await wearables.openFirmwareUpdate() } catch { log("Firmware update: \(error.localizedDescription)") }
        }
    }

    func openGlassesAppUpdate() {
        Task {
            do { try await wearables.openDATGlassesAppUpdate() } catch { log("Glasses app update: \(error.localizedDescription)") }
        }
    }

    func refreshPermission() async {
        cameraGranted = (try? await wearables.checkPermissionStatus(.camera)) == .granted
    }

    func requestCameraPermission() {
        Task {
            do {
                cameraGranted = try await wearables.requestPermission(.camera) == .granted
            } catch {
                log("Camera permission: \(error.localizedDescription)")
            }
        }
    }

    // MARK: - Session

    private func ensureSession() async throws -> DeviceSession {
        if let session, session.state == .started { return session }
        if session == nil || session?.state == .stopped || session?.state == .idle {
            let s: DeviceSession
            do { s = try wearables.createSession(deviceSelector: selector) } catch { throw GlassesError.noSession }
            session = s
            s.statePublisher.listen { [weak self] state in
                Task { @MainActor in self?.handleSessionState(state) }
            }.store(in: sessionBag)
            s.errorPublisher.listen { [weak self] error in
                Task { @MainActor in self?.log("Session error: \(error.localizedDescription) [\(String(describing: error))]") }
            }.store(in: sessionBag)
            do { try s.start() } catch { throw GlassesError.noSession }
        }
        guard let s = session else { throw GlassesError.noSession }
        try await waitUntil(timeout: 15) { s.state == .started }
        return s
    }

    private func handleSessionState(_ state: DeviceSessionState) {
        sessionState = state
        if state == .stopped {
            sessionBag.clear()
            streamBag.clear()
            session = nil
            camera = nil
            streamState = .stopped
        }
    }

    private func waitUntil(timeout: Double, _ condition: @escaping () -> Bool) async throws {
        let deadline = Date().addingTimeInterval(timeout)
        while !condition() {
            if Date() > deadline { throw GlassesError.sessionTimeout }
            try await Task.sleep(for: .milliseconds(100))
        }
    }

    // MARK: - Capture

    /// Starts the camera stream and waits until frames flow. Must be called while the app is in
    /// the foreground: the glasses refuse to *start* a stream from the background, but an open
    /// stream keeps delivering frames after the phone is locked.
    func openStream() async throws {
        if !cameraGranted { await refreshPermission() }
        guard cameraGranted else { throw GlassesError.permission }
        let session = try await ensureSession()

        if camera == nil {
            // Continuous mode runs all day, so keep it cheap: 7 fps is plenty for TRIBE.
            let config = StreamConfiguration(videoCodec: .raw, resolution: .low, frameRate: continuousStream ? 7 : 15)
            let cam: MWDATCamera.Camera?
            do { cam = try session.addCamera(config: config) } catch { throw GlassesError.cameraUnavailable }
            guard let cam else { throw GlassesError.cameraUnavailable }
            camera = cam
            let collector = self.collector
            cam.stream.videoFramePublisher.listen { frame in
                collector.offer(frame)
            }.store(in: streamBag)
            cam.stream.statePublisher.listen { [weak self] state in
                Task { @MainActor in
                    self?.streamState = state
                    if state == .stopped { self?.camera = nil; self?.streamBag.clear() }
                }
            }.store(in: streamBag)
            cam.stream.errorPublisher.listen { [weak self] error in
                Task { @MainActor in self?.log("Stream error: \(error.localizedDescription)") }
            }.store(in: streamBag)
            cam.stream.start()
        }
        guard let cam = camera else { throw GlassesError.cameraUnavailable }
        do {
            try await waitUntil(timeout: 12) { cam.stream.state == .streaming }
        } catch {
            closeCamera()
            throw GlassesError.streamTimeout
        }
    }

    var isStreaming: Bool { camera != nil && streamState == .streaming }

    /// Collects `seconds` of frames at `fps` from the (opened-if-needed) stream, then closes it
    /// unless continuous mode is on. Returns JPEG frames in order.
    func captureClip(seconds: Double = 3.5, fps: Double = 7) async throws -> [Data] {
        try await openStream()
        collector.begin(fps: fps)
        try? await Task.sleep(for: .seconds(seconds))
        let frames = collector.end()
        if !continuousStream { closeCamera() }
        if frames.isEmpty { throw GlassesError.noFrames }
        return frames
    }

    func closeCamera() {
        camera?.stop()
        camera = nil
        streamBag.clear()
        streamState = .stopped
    }

    func endSession() {
        closeCamera()
        session?.stop()
    }

    private func log(_ message: String) {
        lastError = message
        onLog?(message)
    }
}

import AVFoundation

/// Records the phone mic continuously into a 30 s ring buffer (16 kHz mono).
/// Doubles as the background keep-alive: iOS won't suspend an app that is actively recording.
final class AudioKeeper: @unchecked Sendable {
    static let sampleRate: Double = 16_000
    private let capacity = Int(sampleRate * 30)

    private let engine = AVAudioEngine()
    private let lock = NSLock()
    private var ring: [Float]
    private var writeIndex = 0
    private var filled = 0
    private var converter: AVAudioConverter?
    private let outFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: sampleRate, channels: 1, interleaved: false)!
    private var observers: [NSObjectProtocol] = []
    private(set) var isRunning = false
    var onLog: ((String) -> Void)?

    init() {
        ring = [Float](repeating: 0, count: capacity)
    }

    func requestPermission() async -> Bool {
        await AVAudioApplication.requestRecordPermission()
    }

    func start() throws {
        guard !isRunning else { return }
        let session = AVAudioSession.sharedInstance()
        // No .allowBluetoothHFP: keep input on the phone mic so the glasses battery is spared.
        try session.setCategory(.playAndRecord, mode: .default, options: [.mixWithOthers, .allowBluetoothA2DP, .defaultToSpeaker])
        try session.setActive(true)
        try startEngine()
        isRunning = true
        installObservers()
    }

    func stop() {
        isRunning = false
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        observers.forEach(NotificationCenter.default.removeObserver)
        observers.removeAll()
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
    }

    private func startEngine() throws {
        let input = engine.inputNode
        input.removeTap(onBus: 0)
        let inFormat = input.outputFormat(forBus: 0)
        converter = AVAudioConverter(from: inFormat, to: outFormat)
        input.installTap(onBus: 0, bufferSize: 4096, format: inFormat) { [weak self] buffer, _ in
            self?.ingest(buffer)
        }
        engine.prepare()
        try engine.start()
    }

    private func restart(reason: String) {
        guard isRunning else { return }
        onLog?("Audio restart: \(reason)")
        engine.stop()
        do {
            try AVAudioSession.sharedInstance().setActive(true)
            try startEngine()
        } catch {
            onLog?("Audio restart failed: \(error.localizedDescription)")
        }
    }

    private func installObservers() {
        let nc = NotificationCenter.default
        observers.append(nc.addObserver(forName: AVAudioSession.interruptionNotification, object: nil, queue: .main) { [weak self] note in
            guard let raw = note.userInfo?[AVAudioSessionInterruptionTypeKey] as? UInt,
                  AVAudioSession.InterruptionType(rawValue: raw) == .ended else { return }
            self?.restart(reason: "interruption ended")
        })
        observers.append(nc.addObserver(forName: .AVAudioEngineConfigurationChange, object: engine, queue: .main) { [weak self] _ in
            self?.restart(reason: "route/config change")
        })
        observers.append(nc.addObserver(forName: AVAudioSession.mediaServicesWereResetNotification, object: nil, queue: .main) { [weak self] _ in
            self?.restart(reason: "media services reset")
        })
    }

    private func ingest(_ buffer: AVAudioPCMBuffer) {
        guard let converter else { return }
        let ratio = outFormat.sampleRate / buffer.format.sampleRate
        let outCapacity = AVAudioFrameCount(Double(buffer.frameLength) * ratio) + 32
        guard let out = AVAudioPCMBuffer(pcmFormat: outFormat, frameCapacity: outCapacity) else { return }
        var consumed = false
        var error: NSError?
        converter.convert(to: out, error: &error) { _, status in
            if consumed {
                status.pointee = .noDataNow
                return nil
            }
            consumed = true
            status.pointee = .haveData
            return buffer
        }
        guard error == nil, let ch = out.floatChannelData?[0] else { return }
        let n = Int(out.frameLength)
        lock.lock()
        for i in 0..<n {
            ring[writeIndex] = ch[i]
            writeIndex = (writeIndex + 1) % capacity
        }
        filled = min(capacity, filled + n)
        lock.unlock()
    }

    /// Last `seconds` of audio as a 16-bit PCM WAV file. Nil if we have under a second.
    func snapshotWAV(seconds: Double) -> Data? {
        lock.lock()
        let count = min(filled, Int(seconds * Self.sampleRate))
        var samples = [Float](repeating: 0, count: count)
        let start = (writeIndex - count + capacity) % capacity
        for i in 0..<count { samples[i] = ring[(start + i) % capacity] }
        lock.unlock()
        guard count > Int(Self.sampleRate) else { return nil }
        return Self.wav(samples)
    }

    private static func wav(_ samples: [Float]) -> Data {
        var data = Data()
        func u32(_ v: UInt32) { var x = v.littleEndian; data.append(Data(bytes: &x, count: 4)) }
        func u16(_ v: UInt16) { var x = v.littleEndian; data.append(Data(bytes: &x, count: 2)) }
        let byteCount = UInt32(samples.count * 2)
        data.append("RIFF".data(using: .ascii)!); u32(36 + byteCount)
        data.append("WAVE".data(using: .ascii)!)
        data.append("fmt ".data(using: .ascii)!); u32(16); u16(1); u16(1)
        u32(UInt32(sampleRate)); u32(UInt32(sampleRate) * 2); u16(2); u16(16)
        data.append("data".data(using: .ascii)!); u32(byteCount)
        var pcm = samples.map { Int16(max(-1, min(1, $0)) * Float(Int16.max)).littleEndian }
        data.append(Data(bytes: &pcm, count: pcm.count * 2))
        return data
    }
}

import Foundation

struct RegionScore: Codable, Hashable {
    var rot: Double
    var activation_z: Double?
    var delta: Double?
}

struct CaptureResult: Codable, Identifiable, Hashable {
    var capture_id: String
    var timestamp: String
    var regions: [String: RegionScore]
    var roast: String?
    var roast_kind: String?
    var roast_region: String?
    var transcript: String?
    var thumb_url: String?

    var id: String { capture_id }

    var date: Date? {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        if let d = f.date(from: timestamp) { return d }
        f.formatOptions = [.withInternetDateTime]
        return f.date(from: timestamp)
    }

    func rot(_ region: Region) -> Double { regions[region.rawValue]?.rot ?? 0 }
}

enum APIError: LocalizedError {
    case badURL, http(Int, String)

    var errorDescription: String? {
        switch self {
        case .badURL: "Backend URL not set"
        case .http(let code, let body): "HTTP \(code): \(body.prefix(200))"
        }
    }
}

struct BrainAPI {
    var baseURL: String

    private var base: URL? {
        URL(string: baseURL.trimmingCharacters(in: .whitespacesAndNewlines).trimmingCharacters(in: CharacterSet(charactersIn: "/")))
    }

    func url(_ path: String) -> URL? {
        if path.hasPrefix("http") { return URL(string: path) }
        return base?.appendingPathComponent(path.trimmingCharacters(in: CharacterSet(charactersIn: "/")))
    }

    private static let session: URLSession = {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 180
        config.timeoutIntervalForResource = 300
        return URLSession(configuration: config)
    }()

    func upload(captureID: String, timestamp: Date, frames: [Data], fps: Int, audio: Data?, battery: Int?) async throws -> CaptureResult {
        guard let url = url("capture") else { throw APIError.badURL }
        var form = Multipart()
        form.field("capture_id", captureID)
        form.field("timestamp", ISO8601DateFormatter().string(from: timestamp))
        form.field("fps", String(fps))
        if let battery { form.field("glasses_battery", String(battery)) }
        for (i, frame) in frames.enumerated() {
            form.file("frames", filename: String(format: "frame_%03d.jpg", i), mime: "image/jpeg", data: frame)
        }
        if let audio { form.file("audio", filename: "audio.wav", mime: "audio/wav", data: audio) }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("multipart/form-data; boundary=\(form.boundary)", forHTTPHeaderField: "Content-Type")
        let (data, response) = try await Self.session.upload(for: request, from: form.finish())
        try check(response, data)
        return try JSONDecoder().decode(CaptureResult.self, from: data)
    }

    func timeline() async throws -> [CaptureResult] {
        guard let url = url("timeline") else { throw APIError.badURL }
        let (data, response) = try await Self.session.data(from: url)
        try check(response, data)
        if let list = try? JSONDecoder().decode([CaptureResult].self, from: data) { return list }
        struct Wrapped: Codable { var captures: [CaptureResult]?; var timeline: [CaptureResult]? }
        let wrapped = try JSONDecoder().decode(Wrapped.self, from: data)
        return wrapped.captures ?? wrapped.timeline ?? []
    }

    func reset() async throws {
        guard let url = url("reset") else { throw APIError.badURL }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        let (data, response) = try await Self.session.data(for: request)
        try check(response, data)
    }

    private func check(_ response: URLResponse, _ data: Data) throws {
        guard let http = response as? HTTPURLResponse else { return }
        guard (200..<300).contains(http.statusCode) else {
            throw APIError.http(http.statusCode, String(data: data, encoding: .utf8) ?? "")
        }
    }
}

struct Multipart {
    let boundary = "brainrot-\(UUID().uuidString)"
    private var body = Data()

    mutating func field(_ name: String, _ value: String) {
        body.append("--\(boundary)\r\nContent-Disposition: form-data; name=\"\(name)\"\r\n\r\n\(value)\r\n".data(using: .utf8)!)
    }

    mutating func file(_ name: String, filename: String, mime: String, data: Data) {
        body.append("--\(boundary)\r\nContent-Disposition: form-data; name=\"\(name)\"; filename=\"\(filename)\"\r\nContent-Type: \(mime)\r\n\r\n".data(using: .utf8)!)
        body.append(data)
        body.append("\r\n".data(using: .utf8)!)
    }

    func finish() -> Data {
        var out = body
        out.append("--\(boundary)--\r\n".data(using: .utf8)!)
        return out
    }
}

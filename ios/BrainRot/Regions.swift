import SwiftUI

/// The six cortical regions the backend scores. Order matches shader slots (rotA.xyzw, rotB.xy)
/// and the `regions` list in brain_mesh.json.
enum Region: String, CaseIterable, Identifiable, Codable {
    case naturePlaces = "nature_places"
    case facesSocial = "faces_social"
    case movement = "movement"
    case reading = "reading"
    case soundSpeech = "sound_speech"
    case language = "language"

    var id: String { rawValue }
    var slot: Int { Region.allCases.firstIndex(of: self)! }

    var name: String {
        switch self {
        case .naturePlaces: "Nature & Places"
        case .facesSocial: "Faces & Social"
        case .movement: "Movement"
        case .reading: "Reading"
        case .soundSpeech: "Sound & Speech"
        case .language: "Language"
        }
    }

    var anatomy: String {
        switch self {
        case .naturePlaces: "Parahippocampal place area · retrosplenial cortex"
        case .facesSocial: "Fusiform face area · superior temporal sulcus"
        case .movement: "MT+ · extrastriate body area"
        case .reading: "Visual word form area"
        case .soundSpeech: "Heschl's gyrus · superior temporal gyrus"
        case .language: "Broca's area · temporal language network"
        }
    }

    var cure: String {
        switch self {
        case .naturePlaces: "Go outside. Look at a tree."
        case .facesSocial: "Look at an actual human face."
        case .movement: "Move. Watch things move."
        case .reading: "Read something that isn't a notification."
        case .soundSpeech: "Listen to the world. Music counts."
        case .language: "Have a conversation with a person."
        }
    }

    var icon: String {
        switch self {
        case .naturePlaces: "tree.fill"
        case .facesSocial: "face.smiling"
        case .movement: "figure.walk"
        case .reading: "book.fill"
        case .soundSpeech: "waveform"
        case .language: "bubble.left.and.bubble.right.fill"
        }
    }
}

enum Palette {
    static let bg = Color(red: 0.985, green: 0.98, blue: 0.97)
    static let card = Color(red: 0.95, green: 0.94, blue: 0.92)
    static let healthy = Color(red: 0.91, green: 0.53, blue: 0.58)
    static let mold = Color(red: 0.45, green: 0.58, blue: 0.16)
    static let rot = Color(red: 0.42, green: 0.33, blue: 0.13)
    static let text = Color(red: 0.10, green: 0.10, blue: 0.09)
    static let dim = Color(red: 0.55, green: 0.53, blue: 0.49)

    static func forRot(_ r: Double) -> Color {
        if r < 0.3 { return healthy }
        if r < 0.6 { return Color(red: 0.62, green: 0.50, blue: 0.58) }
        if r < 0.85 { return mold }
        return rot
    }
}

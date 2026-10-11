import SwiftUI

struct SettingsView: View {
    @Environment(AppModel.self) private var model
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        @Bindable var model = model
        @Bindable var glasses = model.glasses
        NavigationStack {
            Form {
                Section("Glasses") {
                    LabeledContent("Registration", value: model.glasses.registration.description)
                    LabeledContent("Device", value: model.glasses.hasDevice ? "Connected" : "Not found")
                    if let b = model.glasses.battery { LabeledContent("Battery", value: "\(b)%") }
                    if let d = model.glasses.donned { LabeledContent("Worn", value: d ? "Yes" : "No") }
                    LabeledContent("Known devices", value: "\(model.glasses.knownDevices)")
                    LabeledContent("Link", value: model.glasses.link)
                    LabeledContent("Compatibility", value: model.glasses.compatibility)
                    LabeledContent("Session", value: model.glasses.sessionState.description)
                    Button(model.glasses.isRegistered ? "Re-register with Meta AI" : "Connect glasses (Meta AI)") {
                        model.glasses.connect()
                    }
                    Button(model.glasses.cameraGranted ? "Camera permission ✓" : "Grant camera permission") {
                        model.glasses.requestCameraPermission()
                    }
                    .disabled(!model.glasses.isRegistered)
                    Toggle("Keep stream open (needed when locked)", isOn: $glasses.continuousStream)
                    Button("Update glasses firmware") { model.glasses.openFirmwareUpdate() }
                    Button("Install/update glasses app (DAT)") { model.glasses.openGlassesAppUpdate() }
                }
                Section("Capture") {
                    Stepper("Every \(Int(model.intervalMinutes)) min", value: $model.intervalMinutes, in: 1...15, step: 1)
                    Button("Capture now") { Task { await model.captureOnce() } }
                        .disabled(model.capturing)
                }
                Section("Backend") {
                    TextField("https://…modal.run", text: $model.baseURL)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .keyboardType(.URL)
                    Button("Reload timeline") { Task { await model.refreshTimeline() } }
                    Button("Reset brain (new demo run)", role: .destructive) { model.resetBackend() }
                }
                Section("Log") {
                    ForEach(Array(model.logLines.suffix(60).reversed().enumerated()), id: \.offset) { _, line in
                        Text(line).font(.caption2.monospaced())
                    }
                }
            }
            .navigationTitle("Settings")
            .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } } }
        }
        .preferredColorScheme(.light)
    }
}

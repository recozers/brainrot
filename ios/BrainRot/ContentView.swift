import SwiftUI

struct ContentView: View {
    @Environment(AppModel.self) private var model
    @State private var selected: Region?
    @State private var showSettings = false

    var body: some View {
        ZStack {
            Palette.bg.ignoresSafeArea()
            BrainSceneView(values: model.displayed, healPulse: model.healPulse) { selected = $0 }
                .padding(.top, 60)
                .padding(.bottom, 200)
                .ignoresSafeArea()

            VStack(spacing: 0) {
                header
                Spacer()
                VStack(spacing: 18) {
                    if let (roast, region) = model.displayedRoast {
                        RoastQuote(text: roast, region: region)
                            .transition(.opacity.combined(with: .move(edge: .bottom)))
                            .id(roast)
                    }
                    RegionRow(values: model.displayed) { selected = $0 }
                    trackButton
                    ReplayTimeline()
                }
            }
            .padding(.horizontal, 24)
            .padding(.bottom, 8)
            .animation(.spring(duration: 0.4), value: model.displayedRoast?.0)
        }
        .sheet(item: $selected) { RegionSheet(region: $0) }
        .sheet(isPresented: $showSettings) { SettingsView() }
        .preferredColorScheme(.light)
    }

    private var header: some View {
        HStack(alignment: .firstTextBaseline) {
            VStack(alignment: .leading, spacing: 0) {
                Text("\(Int(model.overallRot * 100))%")
                    .font(.system(size: 56, weight: .heavy, design: .rounded))
                    .foregroundStyle(Palette.text)
                    .contentTransition(.numericText())
                    .animation(.default, value: Int(model.overallRot * 100))
                Text(model.replayIndex == nil ? "of your brain is rotting" : "rotting at \(model.displayedCapture?.date?.formatted(date: .omitted, time: .shortened) ?? "")")
                    .font(.system(.subheadline, design: .rounded))
                    .foregroundStyle(Palette.dim)
            }
            Spacer()
            VStack(alignment: .trailing, spacing: 12) {
                Button { showSettings = true } label: {
                    Image(systemName: "slider.horizontal.3")
                        .font(.title3)
                        .foregroundStyle(Palette.text)
                }
                LastMeal()
            }
        }
        .padding(.top, 8)
    }

    private var trackButton: some View {
        VStack(spacing: 8) {
            Button { model.toggleTracking() } label: {
                HStack(spacing: 8) {
                    if model.capturing {
                        ProgressView().tint(model.tracking ? Palette.text : .white)
                    }
                    Text(model.tracking ? "Stop watching" : "Start watching")
                        .font(.system(.headline, design: .rounded))
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 15)
                .background(model.tracking ? Palette.card : Palette.text, in: Capsule())
                .foregroundStyle(model.tracking ? Palette.text : .white)
            }
            StatusLine()
        }
    }
}

/// What the brain was last fed — the glasses' view at the displayed capture.
struct LastMeal: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        if let capture = model.displayedCapture, let image = model.thumbs[capture.capture_id] {
            VStack(alignment: .trailing, spacing: 4) {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFill()
                    .frame(width: 72, height: 96)
                    .clipShape(RoundedRectangle(cornerRadius: 10))
                    .overlay(RoundedRectangle(cornerRadius: 10).stroke(.black.opacity(0.08)))
                    .id(capture.capture_id)
                    .transition(.opacity)
                Text(capture.date?.formatted(date: .omitted, time: .shortened) ?? "")
                    .font(.caption2.monospacedDigit())
                    .foregroundStyle(Palette.dim)
            }
            .animation(.easeInOut(duration: 0.25), value: capture.capture_id)
        }
    }
}

struct StatusLine: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        TimelineView(.periodic(from: .now, by: 1)) { ctx in
            Text(text(now: ctx.date))
                .font(.caption.monospacedDigit())
                .foregroundStyle(Palette.dim)
        }
    }

    private func text(now: Date) -> String {
        var parts: [String] = []
        if model.glasses.hasDevice {
            parts.append("Glasses" + (model.glasses.battery.map { " \($0)%" } ?? ""))
        } else {
            parts.append("Glasses not connected")
        }
        if model.capturing {
            parts.append("feeding…")
        } else if model.tracking, let next = model.nextCaptureAt {
            let s = max(0, Int(next.timeIntervalSince(now)))
            parts.append(String(format: "next look in %d:%02d", s / 60, s % 60))
        }
        return parts.joined(separator: "  ·  ")
    }
}

struct RoastQuote: View {
    let text: String
    let region: Region?

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            if let region {
                Label(region.name, systemImage: region.icon)
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(Palette.mold)
            }
            Text(text)
                .font(.system(.title3, design: .serif))
                .foregroundStyle(Palette.text)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// One quiet row of six rings — tap for detail.
struct RegionRow: View {
    let values: [Region: Double]
    let onTap: (Region) -> Void

    var body: some View {
        HStack {
            ForEach(Region.allCases) { region in
                let rot = values[region] ?? 0
                Button { onTap(region) } label: {
                    ZStack {
                        Circle().stroke(Palette.card, lineWidth: 3)
                        Circle()
                            .trim(from: 0, to: 1 - rot)
                            .stroke(Palette.forRot(rot), style: StrokeStyle(lineWidth: 3, lineCap: .round))
                            .rotationEffect(.degrees(-90))
                        Image(systemName: region.icon)
                            .font(.system(size: 15))
                            .foregroundStyle(Palette.text.opacity(0.75))
                    }
                    .frame(width: 44, height: 44)
                    .animation(.easeInOut(duration: 0.6), value: rot)
                }
                .accessibilityLabel("\(region.name), \(Int((1 - rot) * 100)) percent healthy")
                if region != Region.allCases.last { Spacer(minLength: 0) }
            }
        }
    }
}

/// The whole session as a strip of bars — taller and pinker means healthier. Drag anywhere to
/// scrub; letting go on the last bar returns to live.
struct ReplayTimeline: View {
    @Environment(AppModel.self) private var model
    @State private var haptics = UISelectionFeedbackGenerator()

    var body: some View {
        let items = model.timeline
        if items.count > 1 {
            let current = model.replayIndex ?? items.count - 1
            let rots = items.map { item in Region.allCases.map(item.rot).reduce(0, +) / Double(Region.allCases.count) }
            let lo = rots.min() ?? 0, hi = max((rots.max() ?? 1), lo + 0.05)
            GeometryReader { geo in
                let step = geo.size.width / CGFloat(items.count)
                ZStack(alignment: .bottomLeading) {
                    HStack(alignment: .bottom, spacing: 2) {
                        ForEach(Array(items.enumerated()), id: \.element.id) { i, item in
                            let rot = rots[i]
                            RoundedRectangle(cornerRadius: 1.5)
                                .fill(Palette.forRot(rot).opacity(i <= current ? 1 : 0.3))
                                .frame(maxWidth: .infinity)
                                .frame(height: 4 + 26 * (hi - rot) / (hi - lo))
                        }
                    }
                    .frame(height: 30, alignment: .bottom)
                    Capsule()
                        .fill(Palette.text)
                        .frame(width: 2.5, height: 38)
                        .offset(x: step * (CGFloat(current) + 0.5) - 1.25)
                        .animation(.easeOut(duration: 0.12), value: current)
                }
                .frame(height: 38, alignment: .bottom)
                .contentShape(Rectangle())
                .gesture(DragGesture(minimumDistance: 0).onChanged { value in
                    let i = min(items.count - 1, max(0, Int(value.location.x / step)))
                    if i != current { haptics.selectionChanged() }
                    model.scrub(to: i)
                })
            }
            .frame(height: 38)
            .accessibilityElement()
            .accessibilityLabel("Session timeline")
            .accessibilityAdjustableAction { direction in
                switch direction {
                case .increment: model.scrub(to: min(items.count - 1, current + 1))
                case .decrement: model.scrub(to: max(0, current - 1))
                @unknown default: break
                }
            }
        }
    }
}

struct RegionSheet: View {
    @Environment(AppModel.self) private var model
    let region: Region

    var body: some View {
        let rot = model.displayed[region] ?? 0
        let capture = model.latestCapture(for: region)
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                HStack {
                    Image(systemName: region.icon).font(.title)
                    VStack(alignment: .leading) {
                        Text(region.name).font(.system(.title2, design: .rounded).weight(.heavy))
                        Text(region.anatomy).font(.caption).foregroundStyle(Palette.dim)
                    }
                }
                HStack(alignment: .firstTextBaseline) {
                    Text("\(Int(rot * 100))%").font(.system(size: 54, weight: .black, design: .rounded))
                        .foregroundStyle(Palette.forRot(rot))
                    Text("rotten").font(.title3.weight(.semibold)).foregroundStyle(Palette.dim)
                }
                Label(region.cure, systemImage: "cross.case.fill")
                    .font(.headline).foregroundStyle(Palette.mold)
                if let capture {
                    if let img = model.thumbs[capture.capture_id] {
                        Image(uiImage: img).resizable().scaledToFit().clipShape(RoundedRectangle(cornerRadius: 12))
                    } else if let url = model.thumbURL(for: capture) {
                        AsyncImage(url: url) { $0.resizable().scaledToFit() } placeholder: { ProgressView() }
                            .clipShape(RoundedRectangle(cornerRadius: 12))
                    }
                    if let roast = capture.roast, capture.roast_region == region.rawValue {
                        RoastQuote(text: roast, region: nil)
                    }
                    if let t = capture.transcript, !t.isEmpty {
                        Text("Heard: “\(t)”").font(.footnote).foregroundStyle(Palette.dim)
                    }
                }
            }
            .padding(20)
            .foregroundStyle(Palette.text)
        }
        .background(Palette.bg)
        .presentationDetents([.medium, .large])
    }
}

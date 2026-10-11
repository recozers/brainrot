import SceneKit
import SwiftUI

struct BrainMesh: Decodable {
    var vertices: [Float]
    var normals: [Float]?
    var faces: [Int32]
    var region: [Int]
    var regions: [String]?

    static func load() -> BrainMesh? {
        guard let url = Bundle.main.url(forResource: "brain_mesh", withExtension: "json"),
              let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(BrainMesh.self, from: data)
    }
}

/// Region rot values are shader uniforms (rotA = slots 0-3, rotB = slots 4-5), so animating rot
/// never touches the 20k-vertex geometry. Each vertex carries its region slot in texcoord.x.
enum RotShader {
    static let geometry = """
    #pragma arguments
    float4 rotA;
    float4 rotB;
    float4 healA;
    float4 healB;
    float rotBase;

    #pragma varyings
    float vrot;
    float vheal;
    float3 opos;

    #pragma body
    int rid = int(round(_geometry.texcoords[0].x));
    float r = rotBase;
    float h = 0.0;
    if (rid >= 0 && rid < 4) { r = rotA[rid]; h = healA[rid]; }
    else if (rid >= 4 && rid < 6) { r = rotB[rid - 4]; h = healB[rid - 4]; }
    out.vrot = r;
    out.vheal = h;
    out.opos = _geometry.position.xyz;
    // Rotting tissue sags and shrivels inward.
    float3 p = _geometry.position.xyz;
    float wobble = 0.6 + 0.4 * sin(p.x * 41.0 + p.y * 29.0) * cos(p.z * 37.0);
    float sag = smoothstep(0.5, 1.0, r) * 0.03 * wobble;
    _geometry.position.xyz -= _geometry.normal * sag;
    """

    static let surface = """
    #pragma body
    #define HASH(c) fract(sin(dot((c), float3(127.1, 311.7, 74.7))) * 43758.5453)
    float r = clamp(in.vrot, 0.0, 1.0);
    float h = clamp(in.vheal, 0.0, 1.0);
    float t = scn_frame.time;

    // fbm value noise, two fields: blotches and mould colonies (slowly creeping).
    float n = 0.0;
    float n2 = 0.0;
    float amp = 0.5;
    float3 q = in.opos * 6.0;
    float3 q2 = in.opos * 11.0 + float3(0.0, t * 0.03, 0.0);
    for (int o = 0; o < 4; o++) {
        float3 i = floor(q); float3 f = fract(q); f = f * f * (3.0 - 2.0 * f);
        float a = mix(mix(mix(HASH(i), HASH(i + float3(1,0,0)), f.x), mix(HASH(i + float3(0,1,0)), HASH(i + float3(1,1,0)), f.x), f.y),
                      mix(mix(HASH(i + float3(0,0,1)), HASH(i + float3(1,0,1)), f.x), mix(HASH(i + float3(0,1,1)), HASH(i + float3(1,1,1)), f.x), f.y), f.z);
        float3 i2 = floor(q2); float3 g = fract(q2); g = g * g * (3.0 - 2.0 * g);
        float b = mix(mix(mix(HASH(i2), HASH(i2 + float3(1,0,0)), g.x), mix(HASH(i2 + float3(0,1,0)), HASH(i2 + float3(1,1,0)), g.x), g.y),
                      mix(mix(HASH(i2 + float3(0,0,1)), HASH(i2 + float3(1,0,1)), g.x), mix(HASH(i2 + float3(0,1,1)), HASH(i2 + float3(1,1,1)), g.x), g.y), g.z);
        n += a * amp; n2 += b * amp;
        q *= 2.03; q2 *= 2.07; amp *= 0.5;
    }

    float3 healthy = float3(0.96, 0.60, 0.64) * (0.88 + 0.24 * n);
    float3 grey = float3(0.56, 0.52, 0.52) * (0.85 + 0.3 * n);
    float3 bruise = mix(float3(0.40, 0.26, 0.44), float3(0.66, 0.60, 0.28), smoothstep(0.35, 0.7, n2));
    float3 mould = mix(float3(0.22, 0.19, 0.09), float3(0.46, 0.58, 0.18), smoothstep(0.45, 0.75, n));

    float3 c = healthy;
    c = mix(c, grey, smoothstep(0.05, 0.35, r));
    c = mix(c, bruise, smoothstep(0.3, 0.6, r) * smoothstep(0.4, 0.62, n2));
    // Mould colonies: thresholds slide down as rot rises so spots grow and merge.
    float colony = smoothstep(0.78 - r * 0.45, 0.86 - r * 0.45, n2) * smoothstep(0.5, 0.75, r);
    c = mix(c, mould, colony);
    // Fuzz: high-frequency speckle on the worst bits.
    float speck = HASH(floor(in.opos * 420.0));
    c = mix(c, float3(0.80, 0.84, 0.70), colony * smoothstep(0.8, 1.0, r) * step(0.82, speck) * 0.8);

    _surface.diffuse = float4(c, 1.0);
    _surface.roughness = mix(0.32, 0.98, r);
    _surface.metalness = 0.0;
    float glow = h * (0.55 + 0.45 * sin(t * 10.0 - in.opos.y * 40.0));
    _surface.emission = float4(float3(0.25, 1.0, 0.45) * glow, 1.0);
    """
}

@MainActor
final class BrainRenderer: NSObject, SCNSceneRendererDelegate {
    let scene = SCNScene()
    let brainNode = SCNNode()
    let cameraNode = SCNNode()
    nonisolated(unsafe) private var material: SCNMaterial?
    private var regionCentroids: [Int: SCNVector3] = [:]
    private var swarms: [Int: SCNNode] = [:]

    // Written from SwiftUI, read on the render thread.
    nonisolated(unsafe) var target: [Float] = Array(repeating: 0.3, count: 6)
    nonisolated(unsafe) var shown: [Float] = Array(repeating: 0.3, count: 6)
    nonisolated(unsafe) var heal: [Float] = Array(repeating: 0, count: 6)
    nonisolated(unsafe) private var lastTime: TimeInterval = 0

    override init() {
        super.init()
        build()
    }

    private func build() {
        scene.background.contents = UIColor(red: 0.985, green: 0.98, blue: 0.97, alpha: 1)

        let camera = cameraNode
        camera.camera = SCNCamera()
        camera.camera?.fieldOfView = 40
        camera.camera?.wantsHDR = false
        camera.position = SCNVector3(0, -0.15, 3.7)
        scene.rootNode.addChildNode(camera)

        let key = SCNNode()
        key.light = SCNLight()
        key.light?.type = .directional
        key.light?.intensity = 1000
        key.light?.color = UIColor(red: 1, green: 0.95, blue: 0.88, alpha: 1)
        key.eulerAngles = SCNVector3(-0.7, 0.5, 0)
        scene.rootNode.addChildNode(key)

        let rim = SCNNode()
        rim.light = SCNLight()
        rim.light?.type = .directional
        rim.light?.intensity = 450
        rim.light?.color = UIColor(red: 0.85, green: 0.9, blue: 1.0, alpha: 1)
        rim.eulerAngles = SCNVector3(0.3, .pi + 0.6, 0)
        scene.rootNode.addChildNode(rim)

        let ambient = SCNNode()
        ambient.light = SCNLight()
        ambient.light?.type = .ambient
        ambient.light?.intensity = 420
        scene.rootNode.addChildNode(ambient)

        let material = SCNMaterial()
        material.lightingModel = .physicallyBased
        material.shaderModifiers = [.geometry: RotShader.geometry, .surface: RotShader.surface]
        material.isDoubleSided = false
        self.material = material
        applyUniforms()

        let geometry: SCNGeometry
        if let mesh = BrainMesh.load() {
            geometry = makeGeometry(mesh)
        } else {
            // Placeholder until brain_mesh.json lands: a lumpy sphere with fake regions.
            geometry = SCNSphere(radius: 0.8)
        }
        geometry.materials = [material]
        let mesh = SCNNode(geometry: geometry)
        // fsaverage is RAS (z up); SceneKit is y up. Then turn to show the left hemisphere side-on.
        mesh.eulerAngles = SCNVector3(-Float.pi / 2, 0, 0)
        brainNode.addChildNode(mesh)
        brainNode.eulerAngles.y = -Float.pi / 2 + 0.5
        brainNode.runAction(.repeatForever(.rotateBy(x: 0, y: CGFloat.pi * 2, z: 0, duration: 60)))
        scene.rootNode.addChildNode(brainNode)
    }

    private func makeGeometry(_ mesh: BrainMesh) -> SCNGeometry {
        let count = mesh.vertices.count / 3
        var verts = [SCNVector3](); verts.reserveCapacity(count)
        var uvs = [CGPoint](); uvs.reserveCapacity(count)
        var sums: [Int: (SCNVector3, Int)] = [:]
        for i in 0..<count {
            let v = SCNVector3(mesh.vertices[i * 3], mesh.vertices[i * 3 + 1], mesh.vertices[i * 3 + 2])
            verts.append(v)
            let rid = i < mesh.region.count ? mesh.region[i] : -1
            uvs.append(CGPoint(x: CGFloat(rid), y: 0))
            if rid >= 0 {
                let s = sums[rid] ?? (SCNVector3Zero, 0)
                sums[rid] = (SCNVector3(s.0.x + v.x, s.0.y + v.y, s.0.z + v.z), s.1 + 1)
            }
        }
        for (rid, s) in sums {
            regionCentroids[rid] = SCNVector3(s.0.x / Float(s.1), s.0.y / Float(s.1), s.0.z / Float(s.1))
        }
        var sources = [SCNGeometrySource(vertices: verts), SCNGeometrySource(textureCoordinates: uvs)]
        if let n = mesh.normals, n.count == mesh.vertices.count {
            sources.append(SCNGeometrySource(normals: (0..<count).map { SCNVector3(n[$0 * 3], n[$0 * 3 + 1], n[$0 * 3 + 2]) }))
        } else {
            sources.append(SCNGeometrySource(normals: computeNormals(verts, mesh.faces)))
        }
        let element = SCNGeometryElement(indices: mesh.faces, primitiveType: .triangles)
        return SCNGeometry(sources: sources, elements: [element])
    }

    private func computeNormals(_ v: [SCNVector3], _ f: [Int32]) -> [SCNVector3] {
        var n = [SIMD3<Float>](repeating: .zero, count: v.count)
        for t in stride(from: 0, to: f.count - 2, by: 3) {
            let a = Int(f[t]), b = Int(f[t + 1]), c = Int(f[t + 2])
            let pa = SIMD3(v[a].x, v[a].y, v[a].z), pb = SIMD3(v[b].x, v[b].y, v[b].z), pc = SIMD3(v[c].x, v[c].y, v[c].z)
            let fn = simd_cross(pb - pa, pc - pa)
            n[a] += fn; n[b] += fn; n[c] += fn
        }
        return n.map { let u = simd_normalize($0); return SCNVector3(u.x, u.y, u.z) }
    }

    func region(at point: CGPoint, in view: SCNView) -> Region? {
        guard let hit = view.hitTest(point, options: [.searchMode: SCNHitTestSearchMode.closest.rawValue]).first else { return nil }
        let rid = Int(hit.textureCoordinates(withMappingChannel: 0).x.rounded())
        return Region.allCases[safe: rid]
    }

    func setTargets(_ values: [Region: Double], healed: [Region]) {
        for r in Region.allCases { target[r.slot] = Float(values[r] ?? 0) }
        for r in healed { heal[r.slot] = 1 }
        updateSwarms()
    }

    nonisolated private func applyUniforms() {
        guard let material else { return }
        material.setValue(NSValue(scnVector4: SCNVector4(shown[0], shown[1], shown[2], shown[3])), forKey: "rotA")
        material.setValue(NSValue(scnVector4: SCNVector4(shown[4], shown[5], 0, 0)), forKey: "rotB")
        material.setValue(NSValue(scnVector4: SCNVector4(heal[0], heal[1], heal[2], heal[3])), forKey: "healA")
        material.setValue(NSValue(scnVector4: SCNVector4(heal[4], heal[5], 0, 0)), forKey: "healB")
        material.setValue(NSNumber(value: shown.reduce(0, +) / 6 * 0.7), forKey: "rotBase")
    }

    nonisolated func renderer(_ renderer: any SCNSceneRenderer, updateAtTime time: TimeInterval) {
        let dt = Float(lastTime == 0 ? 0 : min(0.1, time - lastTime))
        lastTime = time
        var changed = false
        for i in 0..<6 {
            let diff = target[i] - shown[i]
            if abs(diff) > 0.0005 { shown[i] += diff * min(1, dt * 1.5); changed = true }
            if heal[i] > 0 { heal[i] = max(0, heal[i] - dt / 3.0); changed = true }
        }
        if changed {
            applyUniforms()
        }
    }

    /// Flies orbit any region past 80% rot.
    private func updateSwarms() {
        guard let meshNode = brainNode.childNodes.first else { return }
        for r in Region.allCases {
            let rid = r.slot
            let rotten = target[rid] > 0.8
            if rotten, swarms[rid] == nil, let c = regionCentroids[rid] {
                let swarm = SCNNode()
                let len = max(0.001, sqrt(c.x * c.x + c.y * c.y + c.z * c.z))
                let push: Float = 0.18
                swarm.position = SCNVector3(c.x + c.x / len * push, c.y + c.y / len * push, c.z + c.z / len * push)
                for k in 0..<5 {
                    let pivot = SCNNode()
                    pivot.eulerAngles = SCNVector3(Float.random(in: 0...(.pi)), Float.random(in: 0...(.pi)), 0)
                    let fly = SCNNode(geometry: SCNSphere(radius: 0.012))
                    fly.geometry?.firstMaterial?.diffuse.contents = UIColor(white: 0.03, alpha: 1)
                    fly.geometry?.firstMaterial?.lightingModel = .constant
                    fly.position = SCNVector3(0.08 + Float(k) * 0.025, 0, 0)
                    let wingL = SCNNode(geometry: SCNPlane(width: 0.02, height: 0.012))
                    wingL.geometry?.firstMaterial?.diffuse.contents = UIColor(white: 0.85, alpha: 0.45)
                    wingL.geometry?.firstMaterial?.isDoubleSided = true
                    wingL.position = SCNVector3(0, 0.012, 0)
                    wingL.runAction(.repeatForever(.sequence([.rotateBy(x: 0.9, y: 0, z: 0, duration: 0.03), .rotateBy(x: -0.9, y: 0, z: 0, duration: 0.03)])))
                    fly.addChildNode(wingL)
                    pivot.addChildNode(fly)
                    let spin = SCNAction.rotateBy(x: CGFloat.random(in: -1...1), y: CGFloat.pi * 2, z: CGFloat.random(in: -1...1), duration: Double.random(in: 0.9...1.8))
                    pivot.runAction(.repeatForever(spin))
                    let bob = SCNAction.sequence([.moveBy(x: 0, y: 0.03, z: 0, duration: 0.2), .moveBy(x: 0, y: -0.03, z: 0, duration: 0.25)])
                    fly.runAction(.repeatForever(bob))
                    swarm.addChildNode(pivot)
                }
                meshNode.addChildNode(swarm)
                swarms[rid] = swarm
            } else if !rotten, target[rid] < 0.75, let swarm = swarms[rid] {
                swarm.runAction(.sequence([.fadeOut(duration: 0.8), .removeFromParentNode()]))
                swarms[rid] = nil
            }
        }
    }
}

struct BrainSceneView: UIViewRepresentable {
    var values: [Region: Double]
    var healPulse: [Region: Date]
    var onTapRegion: (Region) -> Void

    func makeCoordinator() -> Coordinator { Coordinator(onTap: onTapRegion) }

    func makeUIView(context: Context) -> SCNView {
        let view = SCNView()
        let renderer = context.coordinator.renderer
        view.scene = renderer.scene
        view.pointOfView = renderer.cameraNode
        view.delegate = renderer
        view.allowsCameraControl = true
        view.defaultCameraController.interactionMode = .orbitTurntable
        view.antialiasingMode = .multisampling4X
        view.isPlaying = true
        view.preferredFramesPerSecond = 60
        view.backgroundColor = .clear
        let tap = UITapGestureRecognizer(target: context.coordinator, action: #selector(Coordinator.tapped(_:)))
        view.addGestureRecognizer(tap)
        return view
    }

    func updateUIView(_ view: SCNView, context: Context) {
        let c = context.coordinator
        c.onTap = onTapRegion
        let fresh = healPulse.filter { $0.value > c.lastHealSeen }.map(\.key)
        if let newest = healPulse.values.max(), newest > c.lastHealSeen { c.lastHealSeen = newest }
        c.renderer.setTargets(values, healed: fresh)
    }

    @MainActor
    final class Coordinator: NSObject {
        let renderer = BrainRenderer()
        var onTap: (Region) -> Void
        var lastHealSeen = Date.distantPast

        init(onTap: @escaping (Region) -> Void) { self.onTap = onTap }

        @objc func tapped(_ g: UITapGestureRecognizer) {
            guard let view = g.view as? SCNView else { return }
            if let region = renderer.region(at: g.location(in: view), in: view) { onTap(region) }
        }
    }
}

import CoreAudio

/// Detects whether any app is currently using a microphone, i.e. we're probably in a call.
enum Microphone {
    /// Bundle IDs of processes recording audio input. Only input counts: a headset is a single
    /// device for both directions, so device-level "running" is also true while just playing audio.
    static var users: [String] {
        if #available(macOS 14, *) {
            return objects(kAudioHardwarePropertyProcessObjectList).filter { process in
                var running: UInt32 = 0
                var size = UInt32(MemoryLayout<UInt32>.size)
                var addr = address(kAudioProcessPropertyIsRunningInput)
                return AudioObjectGetPropertyData(process, &addr, 0, nil, &size, &running) == noErr && running != 0
            }.map(bundleID)
        }
        // Pre-14 fallback: input-only devices that are running (can't tell input from output on a headset).
        return objects(kAudioHardwarePropertyDevices).filter { device in
            var running: UInt32 = 0
            var size = UInt32(MemoryLayout<UInt32>.size)
            var addr = address(kAudioDevicePropertyDeviceIsRunningSomewhere)
            return streamCount(device, kAudioObjectPropertyScopeInput) > 0
                && streamCount(device, kAudioObjectPropertyScopeOutput) == 0
                && AudioObjectGetPropertyData(device, &addr, 0, nil, &size, &running) == noErr && running != 0
        }.map { "device \($0)" }
    }

    private static func objects(_ selector: AudioObjectPropertySelector) -> [AudioObjectID] {
        var addr = address(selector)
        var size: UInt32 = 0
        let system = AudioObjectID(kAudioObjectSystemObject)
        guard AudioObjectGetPropertyDataSize(system, &addr, 0, nil, &size) == noErr else { return [] }
        var ids = [AudioObjectID](repeating: 0, count: Int(size) / MemoryLayout<AudioObjectID>.size)
        guard AudioObjectGetPropertyData(system, &addr, 0, nil, &size, &ids) == noErr else { return [] }
        return ids
    }

    private static func streamCount(_ device: AudioObjectID, _ scope: AudioObjectPropertyScope) -> UInt32 {
        var addr = address(kAudioDevicePropertyStreams, scope: scope)
        var size: UInt32 = 0
        guard AudioObjectGetPropertyDataSize(device, &addr, 0, nil, &size) == noErr else { return 0 }
        return size / UInt32(MemoryLayout<AudioStreamID>.size)
    }

    @available(macOS 14, *)
    private static func bundleID(_ process: AudioObjectID) -> String {
        var addr = address(kAudioProcessPropertyBundleID)
        var value: Unmanaged<CFString>?
        var size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
        guard AudioObjectGetPropertyData(process, &addr, 0, nil, &size, &value) == noErr,
              let id = value?.takeRetainedValue() as String?, !id.isEmpty else { return "process \(process)" }
        return id
    }

    private static func address(_ selector: AudioObjectPropertySelector,
                                scope: AudioObjectPropertyScope = kAudioObjectPropertyScopeGlobal) -> AudioObjectPropertyAddress {
        AudioObjectPropertyAddress(mSelector: selector, mScope: scope, mElement: kAudioObjectPropertyElementMain)
    }
}

# IQ.Pilot User Changelog

## IQ.Pilot 1.0c Changelog

### Features

#### Navigate on IQ.Pilot

- Added Navigate on IQ.Pilot as a complete on-device navigation experience.
- Added destination search with Home, Work, and Recent shortcuts.
- Added live turn-by-turn guidance with automatic rerouting after a missed turn or route deviation.
- Added traffic-aware online routing with automatic traffic refresh, delay tracking, closure awareness, and fresh-route metadata.
- Added AMap destination search and routing for configured mainland China devices.
- Added an interactive off-road map with panning and current-location display.
- Added a split on-road map view so the camera, model path, route, current position, and upcoming maneuver remain visible together.
- Added route-aware longitudinal planning for turns, highway exits, and highway forks.
- Added route-aware lane-change guidance for supported highway exits.
- Added route-commanded automatic turn signals on supported vehicles.
- Added optional exit-lane-change assistance with Blind Spot Monitoring awareness.
- Improved low-speed maneuver guidance so the model keeps the requested turn path while waiting and carries it through the committed turn.
- Added route cancellation and saved-destination management from the Navigate screen in the Konn3kt app.

#### IQ Speed Assist

- Added the new IQ Speed Assist architecture.
- Added TomTom speed-limit data alongside dashboard, Mapbox, and offline OpenStreetMap sources.
- Added percentage-based offset zones for low, medium, and higher speed ranges.
- Added direct integration of upcoming speed limits into the longitudinal cruise envelope for earlier and smoother reactions.

#### Construction Zone Assist

- Added optional camera-based Construction Zone Assist.
- Added road-camera detection of bright orange work-zone barrels and markers.
- Added an adjustable work-zone target speed with a 60 mph default.
- Added daylight, road-speed, and active-zone state checks to reduce detections from unrelated reflective objects.

#### Camera Alerts

- Added direct Flock/ALPR hardware detection from nearby Bluetooth and Wi-Fi radio signatures.
- Added warnings for mapped speed cameras, red-light cameras, and ALPR/Flock Safety cameras.
- Direct radio detection works without internet access or preexisting map data, and can warn about a nearby unit before appearing in the map database.
- Added optional speed-camera slowdown using the detected camera limit and a configurable safety factor.
- Added optional haptic feedback on supported Hyundai/Kia/Genesis vehicles when approaching a speed camera.

#### IQ.Dynamic

- Added configurable IQ.Dynamic activation for curves, low road speeds, slower or stopped lead vehicles, and model-predicted stops.
- Added separate road-speed, lead-speed, and model-stop timing controls.
- Added on-device IQ.Dynamic configuration by double-tapping IQ.Dynamic in the longitudinal mode selector.
- Added on-road IQ longitudinal-mode cycling through the nucleus icon on BIG UI devices.
- Added optional stock-radar blending on supported Volkswagen PQ vehicles for more stable highway following.
- Added IQ Force Stops for model-predicted stop lights and stop signs when no lead vehicle is present.
- Added adjustable minimum stop length and stopping distance.

#### General Longitudinal Updates

- Added end-to-end cruise convergence so the vehicle returns toward the selected cruise speed as the road opens in IQ.Pilot (E2E) mode.
- Added Smooth Stops for gentler final braking at regular and model-predicted stops.
- Added smoother pull-away behavior and departure chimes when a lead moves or the path opens.
- Added an optional Experimental Lead MPC mode.
- Added earlier reactions to upcoming curves.

#### IQ Steering Assistance Behavior (SAB)

- Added a distinct lateral-only engagement border separate from full lateral-and-longitudinal engagement.
- Added Always-On Lateral support through compatible Hyundai LFA buttons.
- Added an optional mode that pauses steering torque when the driver takes the wheel and resumes after release.

#### Lane Changes

- Added an optional model-based lane edge guard that blocks lane changes when a road edge is detected on the target side.

#### Lateral Tuning

- Added configurable steering smoothing, slew limiting, and curvature lookahead.
- Improved curve entry and reduced abrupt steering changes while preserving quick avoidance responses.
- Added angle-based steering and optional torque blending for Volkswagen vehicles with ALC.
- Added Volkswagen PQ HCA7 steering support and live-learned curvature correction on supported MEB vehicles.

#### Driving Models

- Added automatic model refresh and redownload when an installed model needs an update.
- Added a clear Driving Model Updating state, and engagement now waits until the selected model is ready.

#### IQ eMac

- Added IQ eMac for running supported big driving models on an Apple Silicon Mac over USB.
- Added a Big Model selector with Off, BRH, Lebowski, and RDF choices.
- Added one-time model download and compilation on the Mac, with cached models for later drives.
- Added automatic USB network setup with a one-time Mac administrator prompt when required.
- Added USB AMD eGPU-dock support with automatic detection and recovery tools.

#### IQ eMac App

- Added a native Mac app for starting, monitoring, restarting, and stopping IQ eMac.
- Added live connection, model, download, compilation, inference rate, latency, and session status.
- Added persistent menu-bar operation with compact live statistics.

#### Home Screen and Off-Road UI

- Added a new IQ.Pilot home screen with dedicated Routes, Navigation, Video, and status views.
- Added a selectable home-panel widget and an expanded status bar.
- Added 60 fps BIG UI presentation and improved Comma 4 visuals.
- Added connected Wi-Fi, vehicle state, temperature, Konn3kt status, and installed IQ.OS version displays.
- Added automatic mph or km/h selection based on the device location.
- Added smoother navigation, transitions, animations, and controls.
- Added Polish and expanded translations throughout the Comma 4 UI and IQ.Pilot settings.

#### On-Road UI

- Added a glowing orb for the primary lead vehicle.
- Added live Konn3kt accent colors across borders, lane lines, controls, and sliders.
- Updated the acceleration bar with IQ.Pilot's teal and pink visual style.
- Added on-road longitudinal personality selection.
- Added a Silent Mode bell control.
- Added Night Mode for automatic display sleep after sunset.
- Added a gradual volume ramp for immediate warning alerts.
- Added screen recording through Konn3kt.

#### Dashcam and Routes

- Increased dashcam (qcam) video resolution by 5x.
- Added a master dashcam control for route logging, video, and audio.
- Added crash-safe recording with recovery after power loss or an interrupted route.
- Added an on-device Routes screen with drive details and upload status.
- Added model path, steering angle, driver-monitoring state, speed, and cruise-speed overlays to the route viewer in the Konn3kt app.

#### Live View, Audio, and WebSSH

- Added live on-road video over cellular/Wi-Fi with the model path overlay.
- Added full-resolution HDR driver-camera video on Comma 4.
- Added microphone audio and two-way voice communication through Konn3kt.
- Added an on-road indicator while Live View is active.
- Added road, wide, and driver-camera snapshots.
- Added faster camera switching, adaptive video quality, and dual-camera picture-in-picture.
- Improved Konn3kt WebSSH connection reliability.

#### Konn3kt Services

- Konn3kt now remains available when IQ.Pilot is stopped or cannot open its main UI.
- Added remote recovery access over Wi-Fi and cellular.
- Added faster reconnection after network or IP-address changes.
- Added dedicated route, log, and crash-log uploading.
- Added Volkswagen and Tesla odometer display.
- Added encrypted device backup and restore.
- Added supported Volkswagen coding, diagnostic controls, and EPS flashing through Konn3kt.

#### Konn3kt Bluetooth Control

- Added direct Bluetooth control from the Konn3kt app across supported IQ.OS devices.
- Added Bluetooth synchronization for vehicle, display, network, navigation, and driving settings.
- Added automatic discovery and pairing without manual network configuration.
- Added automatic Bluetooth fallback when Wi-Fi or cellular service is unavailable.
- Added setup-stage Wi-Fi configuration, channel selection, and installation control.

#### Volkswagen PQ and MQB

- Expanded Volkswagen PQ and MQB vehicle support.
- Added Volkswagen Passat B7/NMS and SEAT Alhambra Stop-and-Go support.
- Added Stop-and-Go and automatic-resume improvements across supported PQ and MQB vehicles.
- Added stock-radar blending with IQ.Dynamic on supported PQ vehicles.
- Added automatic PQ steering-patch detection and minimum-steering-speed handling.
- Added Volkswagen PQ firmware backup, patching, programming, and recovery tools.
- Added PQ and MQB steering coding and compatibility checks through Konn3kt.
- Added continued PQ and MQB lateral control during cruise faults.
- Added an MQB Steering Lockout toggle that reduces low-speed steering torque to prevent LKAS faults on sensitive MQB vehicles.

#### Volkswagen MEB and MQBevo

- Added official Volkswagen MEB and MQBevo support.
- Added supported Volkswagen ID.3, ID.4, ID.5, and Golf Mk8 configurations through model year 2025.
- Added platform-specific steering, zero-speed steering, and ACC display support.
- Added vehicle Drive/Park state handling and reliable device wake support.

#### Toyota and Lexus

- Added Toyota and Lexus Stop-and-Go support with an optional compatibility mode.
- Added SDSU support.
- Fixed ignition handling so the device returns off-road in Park.

#### Hyundai, Kia, and Genesis

- Added more supported Hyundai, Kia, and Genesis variants.
- Added expanded CAN-FD, HDA2, Camera SCC, radar-track, and corner-radar support.
- Added Auto Cruise Control and Auto Engage options on compatible vehicles.
- Added custom steering maximum and steering-rate controls.
- Added lane-change-specific steering-rate controls.
- Added speed-camera haptics and Always-On Lateral support through compatible LFA buttons.

#### Honda, Subaru, and Tesla

- Added smoother final braking on supported Honda vehicles.
- Added Subaru Creep from Standstill.
- Added support for additional Tesla configurations and Model Y steering firmware.
- Fixed Tesla stock-DAS cancellation so cruise speed remains available for IQ.Pilot longitudinal control.
- Added an optional Tesla FSD/Autosteer visualization mode while IQ.Pilot steers.

#### IQ.OS

- Updated device firmware to IQ.OS 4.9.7.
- Added support for Comma 3, Comma 3X, Comma 4, Konik A1/M, and Mr.One C3/C3 Lite.
- Added per-unit Comma 4 display calibration and HDR camera color support.
- Reduced Comma 3 cold-boot time from 39 seconds to 21 seconds.
- Improved USB link recovery for IQ eMac and eGPU configurations.
- Improved Konik A1 audio reliability.
- Added assisted GPS acquisition through Konn3kt.

#### FastSleep and Power Management

- Added FastSleep deep standby after the vehicle is parked.
- Added faster standby when vehicle battery voltage begins to drop.
- Added staged low-voltage shutdown protection.
- Kept Konn3kt recovery access available while high-power IQ.Pilot services sleep.
- Added immediate wake when ignition or charging is detected.

#### Bluetooth Setup and Controller Support

- Added zero-touch Bluetooth onboarding with Konn3kt pairing and setup progress.
- Added IQ.OS update confirmation through Konn3kt.
- Kept Bluetooth controls available independently from the main IQ.Pilot process.

#### Network, Cellular, and eSIM

- Added a new Network settings experience on Comma 3 and Comma 4.
- Added a direct Wi-Fi Disconnect action.
- Added automatic cellular reconnection after APN changes.
- Added SIM recovery for Comma 3X devices with a worn tray-presence switch.
- Added experimental eSIM setup and profile management through QR or manual activation codes.

#### Device and Recovery Controls

- Added Force On-Road for a temporary ten-minute diagnostic session while parked.
- Added Update & Reboot on the crash and recovery screen.
- Added USB Storage mode to access device storage over USB.

#### Updater and Installation

- Added a new IQ.Pilot and IQ.OS update workflow.
- Added Predownload Only and Predownload + Preinstall modes.
- Added interrupted-installation recovery.

#### Reliability and Camera Fault Recovery

- Added independent recovery for navigation and map services.
- Added automatic wide-camera fault detection so the road and driver cameras can continue operating, with an on-road warning after fallback.
- Improved calibration recovery and protected driving services from map-service communication stalls.

---

### Technical Changes

- Reduced the IQ.Pilot total installation size to just 139.48 MiB.
- Removed the legacy openpilot source mirror, compatibility directories, and obsolete source aliases.
- Removed every Git submodule from the IQ.Pilot repository.
- Updated release builds to vendor the exact pinned component sources and required LFS assets into a self-contained installation without nested Git repositories or private source URLs.
- Added navigation-memory handling so important fork and exit guidance remains available when a model output briefly omits it.
- Added on-device map rendering and route-state services designed specifically for IQ.Pilot.
- Reduced navigation CPU, GPU, and memory overhead so the map can remain open for long drives without competing with the driving model.
- Added full offline routing through a packaged Valhalla runtime.
- Added offline navigation that can operate without Mapbox or an active internet connection.
- Added support for keeping multiple offline regions installed simultaneously.
- Added offline raster map tiles for the on-screen map, not only route calculation and road metadata.
- Added separate Online On-Screen Maps and Offline On-Screen Maps controls.
- Added resumable regional map downloads for unstable cellular and hotspot connections.
- Added combined download progress for routing databases and rendered map tiles.
- Added automatic recognition of already installed map regions.
- Added automatic restoration of missing offline-map data.
- Added a pinned IQ.Pilot mapd v2 fork with binary verification and automatic quarantine of incompatible mapd builds.
- Added a hosted regional tile-bundle service with a secondary fallback source.
- Added background tile decoding and bounded texture caching to keep map work off the UI render path.
- Added automatic stock-radar set-speed and following-gap synchronization on supported Volkswagen vehicles.
- Hardened Experimental Lead MPC with model-horizon validation and automatic radar-trajectory fallback when model lead data is missing, malformed, or non-finite.
- Added Bluetooth-controller commands for testing or controlling Always-On Lateral in Joystick Mode.
- Added the unified IQModeld runtime.
- Added a native IQModeld bridge for current combined models and legacy split models.
- Added fused vision-and-policy execution for supported supercombo bundles.
- Added zero-copy camera-frame handling through the current tinygrad runner.
- Added combined-artifact, combined-split, fused, tinygrad, and ONNX runner support under one manager.
- Added signed and notarized Mac packaging with improved USB transport diagnostics.
- Added stable border-crossing detection and last-known-location fallback for IQ Auto Units.
- Added tappable branch information in the BIG UI header.
- Fixed live language updates and several previously untranslated or malformed translations.
- Added qlog-only route visualization, allowing the model path and driving telemetry to be viewed even when only a qlog and qcam are uploaded, and no rlog is available.
- Hardened Qualcomm encoder polling and H.264 filtering for more reliable route recording.
- Fixed upload queues that could stall behind a missing build-version file.
- Fixed route-upload cache invalidation when a file is replaced at the same path, preventing stale upload state from being reused.
- Added automatic LocalAPI deployment and configuration.
- Fixed periodic Konn3kt disconnects caused by numeric heartbeat parameters terminating the connection writer.
- Added authenticated BLE requests, replay protection, and a dedicated settings RPC dispatcher.
- Added live propagation of BLE setting changes to the device UI and active IQ.Pilot services.
- Added TRW450 ACC handling for the Volkswagen Passat B7/NMS.
- Added MQB standstill handling for supported non-EPB ACC vehicles.
- Added dedicated PQ radar engagement, cancellation, set-speed, acceleration, and following-gap management.
- Added model-year ECU fingerprint checks and expanded Passat identification.
- Added explicit Volkswagen car-readiness state handling.
- Added on-car MLB longitudinal refinements and HCA steering-status configuration for compatible Audi and Porsche platforms.
- Corrected Porsche Macan vehicle selection and Volkswagen-group settings presentation.
- Added MEB camera-harness support for lateral control with stock ACC and compatible gateway-harness support for IQ.Pilot longitudinal control.
- Added broader Hyundai/Kia/Genesis vehicle parameter and fingerprint diagnostics.
- Added guarded Tesla vehicle-bus parsing for supported harness configurations.
- Added kernel-level USB 3 logging support.
- Added updated USB 3 receive equalization and VGA calibration for improved link stability.
- Added 2 GB compressed zram swap for additional memory headroom.
- Fixed GPS clock synchronization to interpret and apply timestamps in UTC.
- Forced the Qualcomm camera BPS pipeline to its maximum clock for more consistent frame processing.
- Disabled unsupported EGL zero-copy paths on Comma 4 while retaining them on compatible Comma 3 and Comma 3X hardware.
- Added in-process audio-stream retries so an unavailable Konik A1 audio DSP does not crash-loop the alert service.
- Added cached Konn3kt-assisted GPS data with freshness checks for faster GNSS acquisition when no local AssistNow token is configured.
- Added measured-voltage power decisions while FastSleep is active.
- Added an authenticated Bluetooth GATT transport and RPC dispatcher.
- Added live remote CAN streaming from a device into Cabana through Konn3kt.
- Updated Cabana and Jotpluggler to current upstream tool foundations while retaining Konn3kt device, route, DBC, and direct remote-stream support.
- Added replay support for IQ.Pilot's crash-safe H.264 fragmented-MP4 recordings, including correct container seeking, decoder-delay handling, and multi-frame packet output.
- Added a precompiled release pipeline for comma 3, comma 3X, and comma 4.
- Added automatic runtime-package bootstrap, revision pinning, and credential reuse for fresh devices and updates.
- Added the iq command center for setup, environment checks, builds, quality checks, package synchronization, status, branch switching, updates, service control, fast restart, and optional reboot.
- Added an IQ.Pilot process layout built around independent model, navigation, map, uploader, backup, and perception services.
- Added clearer on-device build, process-state, and diagnostic output.
- Improved Panda SPI NACK handling for more reliable device-to-vehicle communication.
- Hardened calibration by clearing invalid saved calibration, preserving stable solutions during excessive spread on supported comma 3/3X hardware, and rejecting implausible camera-height data before model or path projection.
- Added validation for every public vehicle platform and route plus expanded deterministic and ARM64 process-replay coverage.

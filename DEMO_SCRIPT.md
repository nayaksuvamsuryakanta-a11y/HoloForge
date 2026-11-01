# HoloForge Two-Minute Demonstration

## 0:00–0:15 — Idle cube

“HoloForge is a webcam-driven spatial computing prototype. This frame shows its
wireframe cube and the live HUD. The display is an ordinary monitor; the
holographic treatment is a rendering style. No camera is needed to reproduce
these demo stills.”

## 0:15–0:35 — Grab and rotate

“Pinch and move one hand to rotate. Releasing leaves angular momentum, and
pinching again catches the current motion. The arrow keys provide a keyboard
fallback: the four arrows rotate, plus and minus change scale, and `r` resets.”

## 0:35–0:50 — Two-hand transform

“Two hands enter transform mode after a short confirmation window. Moving
palms apart scales the object; moving them together pans it. This mode takes
precedence over one-hand rotation.”

## 0:50–1:05 — Analysis and export

“Press `a` to show mesh measurements and topology checks. Press `e` to export
the current rotation and scale as an STL. Screen pan and explode displacement
are view operations and are not written into the exported mesh.”

## 1:05–1:25 — Explode and measure

“Press `x` to animate the exploded view. With the hand open, bring landmark 8
near a vertex to highlight it, then pinch to anchor a measurement. Move to a
second vertex and release to freeze the distance. Press `m` to clear saved
measurements.”

## 1:25–1:45 — Audio, voice, and telemetry

“Audio cues are synthesized locally; `v` mutes them. Voice is opt-in with
`--voice` and recognizes analyze, export, reset, explode, and quit. The `Voice`
HUD line reports whether it is available. Press `l` to toggle JSONL telemetry;
records flush every 150 frames and when the session stops.”

## 1:45–2:00 — Configuration and close

“Press `c` to save the current tuning values to the ignored local `config.json`.
CLI tuning flags override that file, which overrides built-in defaults. Press
`q` to exit. The demo stills, benchmark, and tests are generated without opening
a camera, window, or audio device.”

## Keymap

| Key | Action |
|---|---|
| `q` | Quit |
| `a` | Toggle analysis panel |
| `e` | Export STL |
| `x` | Toggle exploded view |
| `m` | Clear measurements |
| `l` | Toggle telemetry recording |
| `c` | Write current tuning config |
| `v` | Toggle audio mute |
| `r` | Reset pose |
| Arrows | Rotate model |
| `+` / `=` / `-` | Change scale |

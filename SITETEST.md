# Site test kit

One file drives everything: `python sitetest.py <command> [flags]` (use `.venv/bin/python` on the Pi).
Every test writes `data/sitetest/<session>/results/NNN_<test>_<VERDICT>.json` + `.md`
(PASS / FAIL / WARN / INFO, with reasons and sample frames). `--help` works on every command.

## Run order on site

| Step | Command |
|---|---|
| 1. Start | `sitetest.py session start --name <site>` → `sitetest.py check` |
| 2. App + logging | `sitetest.py app start --scene-source compose` (or `precomposed`; `--strict` = no fallback) · `sitetest.py telemetry start` |
| 3. Panel | `sitetest.py patterns start` → judge each on screen → `sitetest.py note --test display --label geometry --verdict pass "circle round"` → `patterns stop` (app comes back on its last settings) |
| 4. Seed visitors | visitor faces the display, then `sitetest.py seed --name Asha --gender female --height 160 --from-capture --proxy-from auto` · list: `sitetest.py gallery` |
| 5. Range | per floor mark: `sitetest.py scenario spot --expect varun --label 2m-centre` |
| 6. Scenarios | `sitetest.py scenario list`, then e.g. `scenario empty`, `scenario single --expect varun --label glasses`, `scenario unknown`, `scenario duo --expect varun,asha`, `mixed`, `crowd`, `return`, `walkby`, `queue --expect a,b,c`, `spoof --expect varun` |
| 7. Repeat 6 with the other scene source | `sitetest.py app start --scene-source precomposed` |
| 8. Load / faults | `sitetest.py stress --minutes 20` · `fault kill` · `fault disk --fill-to 95` · `fault display` · `fault power --arm` → pull plug → `fault power --verify` |
| 9. Soak | leave it running; `sitetest.py telemetry check` at the end |
| 10. Finish | `sitetest.py report` → `sitetest.py collect --out /media/<usb>` → `sitetest.py app restore` |

`sitetest.py status` shows everything at a glance; `sitetest.py note "..."` records anything manual
(mount, light, glare).

## Guidelines

- Scenarios are judged from the running app's own captures and logs: keep the app started through
  `app start` (captures on), and keep people out of view who are not part of the scenario.
- Seed with `--from-capture` (same camera, mount and light); a newly seeded visitor is recognised
  within ~5 s. No sketch server on site: `--proxy-from auto` borrows another person's cutouts.
- Run `scenario unknown` with several unseeded people: it is what the report's threshold suggestion
  is computed from.
- Never unplug the camera ribbon while powered (not hot-pluggable). Video slides are out of scope.
- `app restore` puts the units back exactly as installed; config/config.yaml is never edited.
- Captures (full frames + face crops) go to `data/captures/`. On the laptop, run
  `python sitetest.py report --session <unpacked bundle>/session`.

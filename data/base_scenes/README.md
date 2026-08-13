# Base-mode scene assets

Backgrounds and placement configs used by base (server-recognition) mode to
compose each visitor's slides **on the Pi**, from the two raw alpha cutouts the
server serves per person:

- **SAU** — body cutout, placed by scale + a bottom-center anchor.
- **FRU** — face cutout, placed by an eye-anchored warp (scale + rotate + translate).

SAU and FRU are different mechanics and are never cross-composed; each produces
its own separate slides.

> **The `scenes_config.json` files here are placeholders.** Their values are the
> server's real single-person configs (`app/sketch/scenes/{body,face}/crops.json`),
> with the duo variants derived from them by splitting each position left/right.
> They are a working reference for the file format and a starting point for
> tuning — not final art direction. The duo anchors in particular are arithmetic,
> not hand-placed, so expect to adjust them once you can see the result.

## Directories

| Directory | Composes | Config shape |
|---|---|---|
| `sau_single/` | one body | `{"<id>": {"scale": f, "anchor": [x, y]}}` |
| `sau_duo/` | two bodies | `{"<id>": {"scale_1": f, "anchor_1": [x, y], "scale_2": f, "anchor_2": [x, y]}}` |
| `fru_single/` | one face | `{"<id>": {"gender": [...], "face_anchor": {...}}}` |
| `fru_duo/` | two faces | `{"<id>": {"gender": [...], "face_anchor_1": {...}, "face_anchor_2": {...}}}` |

`fru_duo/` uses the same shape as the diagnostic `data/duo_scenes/scenes_config.json`,
just in its own directory. These four dirs are entirely separate from
`data/duo_scenes/` and `data/duo_output/`, which stay diagnostic-mode-only.

## Backgrounds

**No background images are checked in yet** — drop them in beside each
`scenes_config.json`. Until then these configs compose nothing, base mode falls
back to the server's pre-composed images, and nothing breaks.

- Filename stem must equal the JSON key: key `"3"` → `3.png` (`.jpg`/`.jpeg` also work).
- A key with no matching image is skipped; an image with no matching key is
  ignored. Both are normal — the server's own `face/crops.json` deliberately
  omits scenes `4` and `9`.
- All four dirs are independent; a scene id may appear in some and not others.

## Field reference

### Body (`scale` / `anchor`)

```jsonc
"1": { "scale": 0.3922, "anchor": [936, 2539] }
```

- `scale` — multiplier applied to the cutout before pasting.
- `anchor` — `[x, y]` in scene pixels marking where the subject's **feet** land.
  The resized cutout is centred horizontally on `x`, with its **bottom edge** at `y`.

An anchor may legitimately fall outside the scene: the example above sits at
`y = 2539` on a 2240px-tall background, deliberately cropping the feet. The
compositor clips the paste rectangle to the scene bounds, so partially- and
fully-offscreen placements are safe.

Duo uses `scale_1`/`anchor_1` and `scale_2`/`anchor_2` with identical semantics.
A duo scene is only written when **both** people can be placed.

### Face (`face_anchor`)

```jsonc
"2": {
  "gender": ["male"],
  "face_anchor": {
    "target_eye_midpoint": [946, 538],
    "target_eye_distance": 50,
    "target_tilt_angle": 0
  }
}
```

- `target_eye_midpoint` — `[x, y]` the midpoint between the eyes should land on.
- `target_eye_distance` — pixel distance between the eyes after warping; this is
  what sets the apparent face size.
- `target_tilt_angle` — degrees of head tilt, `0` for upright.

Placement is driven entirely by the eyes, detected in the cutout with the same
YuNet model the recognition loop already has loaded. A cutout with no detectable
eyes composes no face slides.

### `gender`

Optional list of genders a scene accepts; defaults to
`["male", "female", "unknown"]`. Used by the FRU paths only — body configs have
no gender filter, matching the server's `body/crops.json`.

Matching is lowercase. A person whose gender is unknown is **never** excluded
(the server may not send the field yet). For duo, a scene is skipped only when
both people's genders are known and both are disallowed — so a mixed pair can
still land on a gender-restricted scene.

## Output

Composed slides are cached under `data/base_assets/` (gitignored) as JPEG:

```
data/base_assets/<registration_id>/display/sau_scene_<id>.jpg, fru_scene_<id>.jpg
data/base_assets/duo/<id_a>__<id_b>/display/...
```

That directory is size-capped by `base_assets_max_bytes` and LRU-evicted, duo
pairs first. Deleting it is always safe — everything in it is regenerable.

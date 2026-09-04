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

Every entry also carries a generated `name` — see the field reference below.

| Directory | Composes | Config shape |
|---|---|---|
| `sau_single/` | one body | `{"<id>": {"scale": f, "anchor": [x, y]}}` |
| `sau_duo/` | two bodies | `{"<id>": {"scale_1": f, "anchor_1": [x, y], "scale_2": f, "anchor_2": [x, y]}}` |
| `fru_single/` | one face | `{"<id>": {"gender": [...], "face_anchor": {...}}}` |
| `fru_duo/` | two faces | `{"<id>": {"gender": [...], "face_anchor_1": {...}, "face_anchor_2": {...}}}` |

Both modes compose from these same four directories. Base mode uses the
cutouts it downloads; diagnostic mode uses the cutouts in each person's
`data/embeddings/<slug>/sketches/` folder. Nothing about the scenes differs
between them.

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

### `name`

A readable label for the entry, derived from its background's filename
(`winter_market.png` -> `"Winter Market"`). It has no effect on composition —
scenes are matched by id — but it is what log lines quote when a slide is
written, and what makes a config readable without cross-referencing the
directory listing.

**Do not hand-edit it.** It is generated, and it drifts the moment a background
is renamed. Regenerate after adding, removing or renaming any background:

```bash
python scripts/sync_scene_names.py            # rewrite the configs
python scripts/sync_scene_names.py --check    # report drift, change nothing
```

The test suite runs `--check`, so a config that drifts fails CI. Today's
backgrounds are numbered (`4.png`), so their names read as `"4"`; the field
starts earning its keep once scenes are named for what they show.

### Body (`scale` / `anchor`)

```jsonc
"1": { "name": "1", "scale": 0.3922, "anchor": [936, 2539] }
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
  "name": "2",
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

Optional list of genders a scene accepts. Used by the FRU paths only — body
configs have no gender filter, matching the server's `body/crops.json`. Omit it
and the scene accepts anyone. Matching is lowercase.

**Single (`fru_single/`)** — the list is the set of genders the scene suits:

| Value | Accepts |
|---|---|
| `["male"]` | a man |
| `["female"]` | a woman |
| `["male", "female"]` | either |

**Duo (`fru_duo/`)** — a duo scene is art with two people posed in it, so the
list constrains the **pair**, not each person:

| Value | Accepts |
|---|---|
| `["male"]` | two men |
| `["female"]` | two women |
| `["male", "female"]` | two men **or** two women — same-gender either way |
| `["mixed"]` | one man and one woman |
| `["male", "mixed"]` | two men **or** a mixed pair (values combine) |

`["male", "female"]` does **not** mean "any pair". Art staged for two men
rarely reads correctly for a man and a woman, so mixed pairs are their own
opt-in via `"mixed"`. `"mixed"` only makes sense in a duo config — in
`fru_single/` it can never match, and the test suite rejects it there.

A person whose gender is unknown is **never** excluded (the server may not send
the field yet); a pair with either gender unknown matches every scene.

## Output

Composed slides are cached as JPEG under one root per mode (both gitignored):

```
data/base_assets/<registration_id>/display/sau_scene_<id>.jpg, fru_scene_<id>.jpg
data/base_assets/duo/<id_a>__<id_b>/display/...
data/diagnostic_assets/<slug>/display/...
data/diagnostic_assets/duo/<slug_a>__<slug_b>/display/...
```

Each root is size-capped by `base_assets_max_bytes` and LRU-evicted, duo pairs
first. Deleting either is always safe — everything in them is regenerable, and
that is also how you force a recompose after retuning anchors.

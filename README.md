# FSAE Michigan EV 2026 Endurance Map

Interactive satellite overlay comparing the official **FSAE Electric 2026 Endurance** course at Michigan International Speedway with georeferenced course data and a **standalone Assetto Corsa track** for the 2026 EV layout.

**Live map:** [https://kithek.github.io/fsae-ev-endurance-map/](https://kithek.github.io/fsae-ev-endurance-map/)

Open the page, toggle layers (official EV line, AC roads/cones/walls, IC comparison), adjust opacity, and nudge placement with the on-screen controls or arrow keys. The layers panel is collapsible (click the header; state is remembered).

**Assetto Corsa map (reference):** [top-down render](https://kithek.github.io/fsae-ev-endurance-map/ac_topdown.png) · [track GeoJSON](https://kithek.github.io/fsae-ev-endurance-map/ac_track.geojson)

## Install: Assetto Corsa track (`fsae_michigan_2026_ev`)

This repository includes a **from-scratch** AC track (no dependency on `fsae_michigan18`):

| Install option | Path |
| --- | --- |
| **Folder** | Copy [`assetto_corsa/fsae_michigan_2026_ev/`](assetto_corsa/fsae_michigan_2026_ev/) into `%Steam%/assettocorsa/content/tracks/` |
| **Zip** | Extract [`artifacts/fsae_michigan_2026_ev.zip`](artifacts/fsae_michigan_2026_ev.zip) into `content/tracks/` |

In Content Manager, select **FSAE Michigan 2026 EV Endurance** (~1.22 km lap, 20 pit boxes).

### Rebuild from source

Requirements: Python 3.10+, `numpy`, `Pillow`, `shapely`.

```bash
pip install numpy pillow shapely
python3 scripts/extract_ev_cones.py      # reads official_ev_endurance_2026.png
python3 scripts/build_fsae_ev_track.py   # writes assetto_corsa/fsae_michigan_2026_ev/
```

Validation images are written to [`artifacts/`](artifacts/) (`check_satellite_and_ac_world.png`, `check_road_on_official_map.png`). The generated `.kn5` is re-read with [`tools/kn5.py`](tools/kn5.py).

### Track contents & limitations

- **Programmatic kn5 (v5)** via [`scripts/kn5_writer.py`](scripts/kn5_writer.py): physics `1ROAD_0`, `1GRASS_0`, `1WALL_0`; visuals `ROAD_0`, `GRASS_0`; cones from the official EV map; satellite ground texture from Esri mosaic `tools/mis_z18.jpg`.
- **Spots:** `AC_START_0`–`19`, `AC_PIT_0`–`19`, `AC_TIME_0` / `AC_TIME_1` beams near start-finish and mid-course (staging aligned to traced start-finish on the official map).
- **`data/surfaces.ini`**, **`data/map.ini`**, **`data/ai/fast_lane.ai`** (centreline-based, optional AI support — verify in-game).
- **Flat track:** no elevation mesh; Y-up AC coordinates; north from WGS84 ↔ AC affine fit (~2 mm at community-track spot anchors).
- **Not verified in-game here:** physics grip, AI line quality, pit/timing gate exact placement vs rules, CSP grass shader. Please report issues after a test drive.

## What you can see (web map)

- Official EV 2026 endurance map georeferenced on Esri World Imagery
- Traced EV 2026 line (~1.22 km; ~18 laps for 22 km)
- Legacy Assetto Corsa `fsae_michigan18` geometry in `ac_track.geojson` (comparison only)
- Optional IC 2026 map/line for comparison

## Local preview (web map)

Serve the folder over HTTP (browsers may block local file access for tiles/scripts):

```powershell
py -m http.server 8000
```

Then open [http://localhost:8000](http://localhost:8000). Needs internet for Leaflet and Esri tiles.

## Georeference notes

- **EV map:** 30 ft grid cells ≈ 18.0 px → 0.508 m/px. Map left is NNE (Gate 50), map right is SSW (Gate 12), bearing 193.5°. Fit ±5 m to back-straight pavement edges.
- **AC track (new):** WGS84 → local ENU → AC world using the same ~1 m scale / orientation as `fsae_michigan18` spot fit; course centreline from `official_line.geojson`.
- **Imagery:** Esri World Imagery, GeoEye-1 (Vantor/Vivid), 30 Jun 2019, 0.46 m resolution.

## Files

| File | Description |
| --- | --- |
| `assetto_corsa/fsae_michigan_2026_ev/` | Installable AC track package |
| `artifacts/*.png`, `artifacts/fsae_michigan_2026_ev.zip` | QA overlays and release zip |
| `scripts/build_fsae_ev_track.py` | Track generator |
| `index.html` / `data.js` | Interactive Leaflet viewer |
| `official_ev_endurance_2026*.png` | Official EV 2026 endurance map |
| `official_endurance_2026*.png` | IC 2026 map (comparison layer) |
| `official_line.geojson` | Traced EV endurance line |
| `ac_track.geojson` / `fsae_michigan_all.geojson` | Legacy AC `fsae_michigan18` geometry |
| `fsae_michigan_2026_endurance.kml` | KML export of the endurance line |
| `overlay_*.png` / `ac_topdown.png` | Static overlay renders |
| `tools/` | Python georef helpers (`geo.py`, `kn5.py`, imagery) |

## License / attribution

Official FSAE endurance map imagery from [fsaeonline MapViewer](https://www.fsaeonline.com/). Satellite imagery © Esri, Maxar/Vantor, Earthstar Geographics. Legacy Assetto Corsa data from community track `fsae_michigan18` (comparison only; not required to drive this mod).
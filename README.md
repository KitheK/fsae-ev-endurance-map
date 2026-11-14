# FSAE Michigan EV 2026 Endurance Map

Interactive satellite overlay comparing the official **FSAE Electric 2026 Endurance** course at Michigan International Speedway with the Assetto Corsa `fsae_michigan18` track.

**Live map:** [https://kithek.github.io/fsae-ev-endurance-map/](https://kithek.github.io/fsae-ev-endurance-map/)

Open the page, toggle layers (official EV line, AC roads/cones/walls, IC comparison), adjust opacity, and nudge placement with the on-screen controls or arrow keys.

## What you can see

- Official EV 2026 endurance map georeferenced on Esri World Imagery
- Traced EV 2026 line (~1.22 km; ~18 laps for 22 km)
- Assetto Corsa road, physics, cones, walls, and start/pit/timing spots
- Optional IC 2026 map/line for comparison

## Local preview

Serve the folder over HTTP (browsers may block local file access for tiles/scripts):

```powershell
py -m http.server 8000
```

Then open [http://localhost:8000](http://localhost:8000). Needs internet for Leaflet and Esri tiles.

## Georeference notes

- **EV map:** 30 ft grid cells ≈ 18.0 px → 0.508 m/px. Map left is NNE (Gate 50), map right is SSW (Gate 12), bearing 193.5°. Fit ±5 m to back-straight pavement edges.
- **AC track:** placed from the kn5 embedded GoogleMap ground texture; walls match the real MIS oval (SIFT fit ~1.6 m RMS).
- **Imagery:** Esri World Imagery, GeoEye-1 (Vantor/Vivid), 30 Jun 2019, 0.46 m resolution.

## Files

| File | Description |
| --- | --- |
| `index.html` / `data.js` | Interactive Leaflet viewer |
| `official_ev_endurance_2026*.png` | Official EV 2026 endurance map |
| `official_endurance_2026*.png` | IC 2026 map (comparison layer) |
| `official_line.geojson` | Traced EV endurance line |
| `ac_track.geojson` / `fsae_michigan_all.geojson` | AC track geometry |
| `fsae_michigan_2026_endurance.kml` | KML export of the endurance line |
| `overlay_*.png` / `ac_topdown.png` | Static overlay renders |

## License / attribution

Official FSAE endurance map imagery from [fsaeonline MapViewer](https://www.fsaeonline.com/). Satellite imagery © Esri, Maxar/Vantor, Earthstar Geographics. Assetto Corsa track data from `fsae_michigan18`.

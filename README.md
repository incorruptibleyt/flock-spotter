# Flock Spotter

A phone web page that watches the road through your camera and logs Flock license-plate cameras it spots, with GPS, without you touching the screen.

- Runs entirely on the phone. The AI model runs in the browser; spotted locations and thumbnails are stored only on the phone.
- Spots are tagged "on map" when a camera is already mapped in OpenStreetMap near that point, or "new" when it isn't, so you can add new ones to [DeFlock](https://deflock.me).
- Export gives a GeoJSON file of your spots.
- It detects cameras only. It does not detect or track vehicles or people.

## Use it

Open the GitHub Pages link on your phone, tap Start, and allow camera and location. Mount the phone so the back camera faces the road. Don't handle the phone while driving.

## What's inside

| File | What it is |
| --- | --- |
| `index.html`, `app.js` | The page and its logic |
| `model.onnx` | Small single-class YOLO detector trained for Flock camera heads |
| `known.js` | Flock cameras already mapped in OpenStreetMap around Kansas (© OpenStreetMap contributors, ODbL) |
| `ort/` | ONNX Runtime Web (MIT), vendored so the page works offline |
| `sw.js` | Caches the page so it works with weak signal |

The model was trained from the [Outflock](https://huggingface.co/aseylys/Outflock) dataset plus street-level photos at mapped camera locations. Training photos are not included in this repo.

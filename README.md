# RfDetr.Preprocessing

A platform-agnostic .NET 10 library for **pre- and postprocessing** images for the RF-DETR-Seg-Nano ONNX model. Does **not** run inference — it only prepares inputs and interprets outputs.

Designed for use in **.NET MAUI** apps (iOS, Android, Windows, macOS) but has no platform dependencies beyond SkiaSharp.

---

## Context

This library is the mobile-side companion to the [rf-detr-document](https://github.com/Agredo/rf-detr-document) training pipeline.  
The ONNX model detects and segments documents (IDs, contracts, letters, ...) in photos. This library handles everything around the inference call:

```
Camera/File bytes
      │
      ▼
┌─────────────────────┐
│  ImagePreprocessor  │  → float[] tensor  [1, 3, 312, 312]
└─────────────────────┘
      │
      ▼
  ONNX Runtime (caller)
      │
      ▼
┌─────────────────────┐
│   PostProcessor     │  ← dets [100,4] + labels [100,2] + masks [100,78,78]
└─────────────────────┘
      │
      ▼
List<DetectionResult>
      │
      ├── BoundingBox          (pixel coords, original size)
      ├── NormalizedBoundingBox (0–1)
      ├── Mask                 (byte[], original size, 0/255 per pixel)
      ├── GetCorners(n)        → SKPoint[] pixel coords
      └── GetNormalizedCorners(n) → SKPoint[] 0–1
```

---

## Installation

```xml
<PackageReference Include="SkiaSharp" Version="3.116.1" />
<!-- Add project reference to RfDetr.Preprocessing -->
```

Requirements: **.NET 10**, **SkiaSharp 3.116.1**

---

## Usage

### Complete Example — 1920×1080 Image

This example shows the full pipeline for a high-resolution photo. **You never resize manually** — the library handles the 312×312 model input internally and maps everything back to original pixel space automatically.

```
1920×1080 JPEG
      │
      │  ImagePreprocessor.Prepare()
      │  → scales to 312×312 internally
      │  → remembers 1920 / 1080 via out params
      ▼
float[291888]  (= 1×3×312×312, NCHW)
      │
      │  ONNX Runtime  (your code)
      ▼
dets   float[400]      (= 100×4)
labels float[200]      (= 100×2)
masks  float[608400]   (= 100×78×78)
      │
      │  PostProcessor.Process(origW: 1920, origH: 1080)
      │  → converts cxcywh → pixel xyxy  (×1920 / ×1080)
      │  → upscales each 78×78 mask → 1920×1080
      ▼
DetectionResult
  ├── Mask        byte[2073600]   (= 1920×1080, 0/255)
  ├── BoundingBox SKRectI         in 1920×1080 pixel space
  └── GetCorners(4) SKPoint[]     in 1920×1080 pixel space
```

```csharp
using RfDetr.Preprocessing;
using Microsoft.ML.OnnxRuntime;

// ── 1. Preprocess ────────────────────────────────────────────────────────
byte[] imageBytes = File.ReadAllBytes("photo_1920x1080.jpg"); // JPEG or PNG

float[] tensor = ImagePreprocessor.Prepare(imageBytes, out int origW, out int origH);
// origW = 1920, origH = 1080  (read from the actual image, no hardcoding needed)
// tensor has 1×3×312×312 = 291 888 floats — ready for the model

// ── 2. Inference ─────────────────────────────────────────────────────────
using var session = new InferenceSession("model_int8.onnx");
using var inputTensor = OrtValue.CreateTensorValueFromMemory(
    tensor, new long[] { 1, 3, 312, 312 });

var outputs = session.Run(
    new RunOptions(),
    inputNames:  ["input"],
    inputValues: [inputTensor],
    outputNames: ["dets", "labels", "masks"]);

float[] dets   = outputs[0].GetTensorDataAsSpan<float>().ToArray(); // 100×4
float[] labels = outputs[1].GetTensorDataAsSpan<float>().ToArray(); // 100×2
float[] masks  = outputs[2].GetTensorDataAsSpan<float>().ToArray(); // 100×78×78

// ── 3. Postprocess ────────────────────────────────────────────────────────
// Pass origW/origH — the library maps everything back to 1920×1080 for you
var results = PostProcessor.Process(dets, labels, masks, origW, origH, threshold: 0.5f);

if (results.Count == 0)
{
    Console.WriteLine("No document detected.");
    return;
}

DetectionResult best = results[0]; // sorted by score descending

// ── Mask in original resolution ──────────────────────────────────────────
byte[] mask = best.Mask;
// Length = 1920 × 1080 = 2 073 600 bytes
// 255 = document pixel, 0 = background
// Access pixel at (x, y):  mask[y * origW + x]

// ── 4 corners in original pixel space ────────────────────────────────────
SKPoint[] corners = best.GetCorners(4, CornerMethod.ContourApproximation);
// corners[0..3] are SKPoint with X/Y in [0, 1920] / [0, 1080]
// Typical order returned by contour tracing (clockwise from top-left)

Console.WriteLine($"Score:   {best.Score:P1}");
Console.WriteLine($"BBox:    {best.BoundingBox}");  // e.g. {240,135,1680,945}
Console.WriteLine($"Corners: {string.Join(", ", corners.Select(p => $"({p.X:F0},{p.Y:F0})"))}");

// ── Use corners for perspective correction ────────────────────────────────
// corners[] can be passed directly to a homography/warp function
// e.g. SkiaSharp SKMatrix.CreatePerspective or OpenCV warpPerspective
```

> **Why does this work without manual rescaling?**
> `ImagePreprocessor.Prepare()` resizes to 312×312 internally but returns the original dimensions.
> `PostProcessor` uses those dimensions to:
> - multiply normalized bbox coordinates by `origW`/`origH`
> - bilinearly upscale each 78×78 logit mask to `origW × origH` before binarizing

---

### Minimal Usage (quick start)

```csharp
using RfDetr.Preprocessing;

byte[] imageBytes = File.ReadAllBytes("photo.jpg"); // or from camera stream

float[] tensor = ImagePreprocessor.Prepare(imageBytes, out int origW, out int origH);
// Pass as ONNX input named "input"
```

### Step 2 — Run Inference (your code)

```csharp
// Using Microsoft.ML.OnnxRuntime:
using var session = new InferenceSession("model_int8.onnx");
using var inputTensor = OrtValue.CreateTensorValueFromMemory(tensor, new long[] { 1, 3, 312, 312 });

var outputs = session.Run(new RunOptions(), ["input"], [inputTensor], ["dets", "labels", "masks"]);

float[] dets   = outputs[0].GetTensorDataAsSpan<float>().ToArray(); // [100 * 4]
float[] labels = outputs[1].GetTensorDataAsSpan<float>().ToArray(); // [100 * 2]
float[] masks  = outputs[2].GetTensorDataAsSpan<float>().ToArray(); // [100 * 78 * 78]
```

### Step 3 — Postprocess (after inference)

```csharp
var results = PostProcessor.Process(dets, labels, masks, origW, origH, threshold: 0.5f);

DetectionResult best = results[0];

SKPoint[] corners = best.GetCorners(4, CornerMethod.ContourApproximation); // pixel coords
byte[]    mask    = best.Mask;   // length = origW * origH, values 0 or 255
SKRectI   bbox    = best.BoundingBox;
```

---

## ONNX Model: Input & Output

### Input

| Name    | Shape              | Type    | Description                              |
|---------|--------------------|---------|------------------------------------------|
| `input` | `[1, 3, 312, 312]` | float32 | NCHW, RGB, pixel values divided by 255.0 |

> No ImageNet mean subtraction — raw normalization to [0, 1] only.

### Outputs

| Name     | Shape               | Type    | Description                                           |
|----------|---------------------|---------|-------------------------------------------------------|
| `dets`   | `[1, 100, 4]`       | float32 | Bounding boxes **cxcywh normalized** (center x/y + w/h, range 0–1) |
| `labels` | `[1, 100, 2]`       | float32 | Raw logits: `[logit_document, logit_no-object]`        |
| `masks`  | `[1, 100, 78, 78]`  | float32 | Raw logits for segmentation masks (78×78 per query)   |

### Important Details

- **100 queries** are always output. Most have near-zero scores — filter by threshold.
- **Class index 1 = no-object.** Score for the document class must be computed via softmax over both logits:
  ```
  score = softmax([logit_document, logit_no_object])[0]
  ```
- **Masks are raw logits** (not sigmoid). The library applies sigmoid + threshold (0.5) internally.
- **Masks are 78×78** and are bilinearly upscaled to original image dimensions by `MaskProcessor`.
- **BBoxes are cxcywh** (not xyxy). The library converts to pixel xyxy automatically.

---

## API Reference

### `ImagePreprocessor`

```csharp
// Decode JPEG/PNG → CHW float tensor [1*3*312*312]
static float[] Prepare(byte[] imageBytes, out int originalWidth, out int originalHeight)
static float[] Prepare(ReadOnlySpan<byte> imageBytes, out int originalWidth, out int originalHeight)

// Decode + resize only (returns SKBitmap, caller must dispose)
static SKBitmap DecodeAndResize(ReadOnlySpan<byte> imageBytes)

const int ModelSize = 312
```

### `PostProcessor`

```csharp
static IReadOnlyList<DetectionResult> Process(
    float[] dets,          // [100 * 4]
    float[] labels,        // [100 * 2]
    float[] masks,         // [100 * 78 * 78]
    int originalWidth,
    int originalHeight,
    float threshold = 0.5f)
// Returns results sorted descending by score
```

### `DetectionResult`

```csharp
float  Score                          // 0–1 confidence
SKRectI BoundingBox                   // pixel coords (xyxy)
SKRect  NormalizedBoundingBox         // 0–1 (xyxy)
byte[]  Mask                          // 0/255 per pixel, row-major, length = origW * origH
int     OriginalWidth
int     OriginalHeight

SKPoint[] GetCorners(int cornerCount = 4, CornerMethod method = ContourApproximation)
SKPoint[] GetNormalizedCorners(int cornerCount = 4, CornerMethod method = ContourApproximation)
```

### `CornerMethod`

| Value | Description |
|---|---|
| `ContourApproximation` | Traces mask boundary, simplifies with Douglas-Peucker to N points. Best for actual document corners. |
| `BoundingBox` | Returns 4 corners of the axis-aligned bounding box. Always a rectangle. |

---

## Related

- **Training & ONNX pipeline**: [github.com/Agredo/rf-detr-document](https://github.com/Agredo/rf-detr-document)
- **Model**: RF-DETR-Seg-Nano (Apache 2.0) — single class `document`, input 312×312

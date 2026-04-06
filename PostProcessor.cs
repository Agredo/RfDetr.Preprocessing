using SkiaSharp;

namespace RfDetr.Preprocessing;

/// <summary>
/// Verarbeitet den rohen ONNX-Output des RF-DETR-Modells und erzeugt
/// eine Liste von <see cref="DetectionResult"/>-Objekten.
///
/// Erwartete ONNX-Ausgaben (flache Arrays, wie von Microsoft.ML.OnnxRuntime geliefert):
///   - <c>dets</c>   : float[1 * 100 * 4]  — cxcywh normiert
///   - <c>labels</c> : float[1 * 100 * 2]  — Logits; Klasse 0 = Dokument, Klasse 1 = kein Objekt
///   - <c>masks</c>  : float[1 * 100 * 78 * 78] — Logits
/// </summary>
public static class PostProcessor
{
    private const int NumQueries  = 100;
    private const int NumClasses  = 2;   // [dokument, no-object]
    private const int MaskSize    = 78;

    /// <summary>
    /// Verarbeitet den ONNX-Output und gibt alle Detektionen über dem Schwellwert zurück.
    /// </summary>
    /// <param name="dets">Flaches float-Array, Länge 100×4 (oder mit Batch: 1×100×4).</param>
    /// <param name="labels">Flaches float-Array, Länge 100×2 (oder 1×100×2).</param>
    /// <param name="masks">Flaches float-Array, Länge 100×78×78 (oder 1×100×78×78).</param>
    /// <param name="originalWidth">Originalbreite aus <see cref="ImagePreprocessor.Prepare"/>.</param>
    /// <param name="originalHeight">Originalhöhe aus <see cref="ImagePreprocessor.Prepare"/>.</param>
    /// <param name="threshold">Score-Schwellwert (Standard 0,5).</param>
    /// <returns>Alle Detektionen mit Score ≥ threshold, absteigend nach Score sortiert.</returns>
    public static IReadOnlyList<DetectionResult> Process(
        float[] dets,
        float[] labels,
        float[] masks,
        int originalWidth,
        int originalHeight,
        float threshold = 0.5f)
    {
        // Batch-Dimension überspringen wenn nötig
        int detOffset   = dets.Length   == 1 * NumQueries * 4              ? 0 : 4;               // safety
        int labelOffset = labels.Length == 1 * NumQueries * NumClasses      ? 0 : NumClasses;
        int maskOffset  = masks.Length  == 1 * NumQueries * MaskSize * MaskSize ? 0 : MaskSize * MaskSize;
        _ = detOffset; _ = labelOffset; _ = maskOffset; // reserved for future batch support

        var results = new List<DetectionResult>();

        for (int q = 0; q < NumQueries; q++)
        {
            // ── Score via Softmax (nur Klasse 0, Klasse 1 = no-object) ────────
            int labelBase = q * NumClasses;
            float logit0 = labels[labelBase];
            float logit1 = labels[labelBase + 1];
            float score  = Softmax2(logit0, logit1); // P(class=document)

            if (score < threshold)
                continue;

            // ── Bounding-Box: cxcywh normiert → xyxy Pixel ───────────────────
            int detBase = q * 4;
            float cx = dets[detBase];
            float cy = dets[detBase + 1];
            float w  = dets[detBase + 2];
            float h  = dets[detBase + 3];

            float x1n = cx - w * 0.5f;
            float y1n = cy - h * 0.5f;
            float x2n = cx + w * 0.5f;
            float y2n = cy + h * 0.5f;

            int x1 = Clamp((int)(x1n * originalWidth),  0, originalWidth  - 1);
            int y1 = Clamp((int)(y1n * originalHeight), 0, originalHeight - 1);
            int x2 = Clamp((int)(x2n * originalWidth),  0, originalWidth  - 1);
            int y2 = Clamp((int)(y2n * originalHeight), 0, originalHeight - 1);

            var bboxPixel = new SKRectI(x1, y1, x2, y2);
            var bboxNorm  = new SKRect(
                Math.Clamp(x1n, 0f, 1f),
                Math.Clamp(y1n, 0f, 1f),
                Math.Clamp(x2n, 0f, 1f),
                Math.Clamp(y2n, 0f, 1f));

            // ── Maske upscalen ────────────────────────────────────────────────
            int maskBase = q * MaskSize * MaskSize;
            var maskSlice = masks.AsSpan(maskBase, MaskSize * MaskSize);
            byte[] maskBytes = MaskProcessor.Upscale(maskSlice, originalWidth, originalHeight);

            results.Add(new DetectionResult(
                score,
                bboxPixel,
                bboxNorm,
                maskBytes,
                originalWidth,
                originalHeight));
        }

        results.Sort((a, b) => b.Score.CompareTo(a.Score));
        return results;
    }

    // ── Hilfsfunktionen ───────────────────────────────────────────────────────

    /// <summary>Softmax über 2 Logits, gibt P(logit0) zurück.</summary>
    private static float Softmax2(float a, float b)
    {
        float maxVal = MathF.Max(a, b);
        float ea = MathF.Exp(a - maxVal);
        float eb = MathF.Exp(b - maxVal);
        return ea / (ea + eb);
    }

    private static int Clamp(int value, int min, int max)
        => value < min ? min : value > max ? max : value;
}

using SkiaSharp;
using System.Runtime.InteropServices;

namespace RfDetr.Preprocessing;

/// <summary>
/// Bereitet ein Bild für die RF-DETR ONNX-Inferenz vor.
///
/// Das Modell erwartet:
///   - Input-Name : "input"
///   - Shape      : [1, 3, 312, 312]  (NCHW, float32)
///   - Normalisierung: Pixelwert / 255.0  (keine ImageNet-Mean-Subtraktion)
/// </summary>
public static class ImagePreprocessor
{
    /// <summary>Modelleingangsgröße (Breite = Höhe).</summary>
    public const int ModelSize = 312;

    /// <summary>
    /// Dekodiert ein JPEG- oder PNG-kodiertes Bild und erzeugt den Float-Tensor
    /// für das RF-DETR ONNX-Modell.
    /// </summary>
    /// <param name="imageBytes">JPEG oder PNG als Byte-Array (z.B. aus Kamera oder Datei).</param>
    /// <param name="originalWidth">Originalbreite des Bilds in Pixeln (für Postprocessing benötigt).</param>
    /// <param name="originalHeight">Originalhöhe des Bilds in Pixeln (für Postprocessing benötigt).</param>
    /// <returns>
    /// Flacher float[]-Tensor in NCHW-Reihenfolge: [1, 3, 312, 312].
    /// Länge = 1 × 3 × 312 × 312 = 291'888 Elemente.
    /// Kann direkt als ONNX-Input übergeben werden.
    /// </returns>
    /// <exception cref="ArgumentException">Wenn das Bild nicht dekodiert werden kann.</exception>
    public static float[] Prepare(
        ReadOnlySpan<byte> imageBytes,
        out int originalWidth,
        out int originalHeight)
    {
        using var bitmap = Decode(imageBytes, out originalWidth, out originalHeight);
        return BitmapToTensor(bitmap);
    }

    /// <summary>
    /// Überladung für byte[] statt ReadOnlySpan — für Kompatibilität mit älteren APIs.
    /// </summary>
    public static float[] Prepare(
        byte[] imageBytes,
        out int originalWidth,
        out int originalHeight)
        => Prepare(imageBytes.AsSpan(), out originalWidth, out originalHeight);

    /// <summary>
    /// Dekodiert das Bild, skaliert auf ModelSize×ModelSize und gibt eine
    /// verwaltete SKBitmap (RGBA8888) zurück. Der Aufrufer muss disposen.
    /// Nützlich wenn das Bitmap für weitere Zwecke benötigt wird.
    /// </summary>
    public static SKBitmap DecodeAndResize(ReadOnlySpan<byte> imageBytes)
        => Decode(imageBytes, out _, out _);

    // ── Interna ────────────────────────────────────────────────────────────

    private static SKBitmap Decode(ReadOnlySpan<byte> imageBytes, out int origW, out int origH)
    {
        using var data = SKData.CreateCopy(imageBytes);
        using var original = SKBitmap.Decode(data)
            ?? throw new ArgumentException("Bild konnte nicht dekodiert werden (kein gültiges JPEG/PNG).");

        origW = original.Width;
        origH = original.Height;

        if (original.Width == ModelSize && original.Height == ModelSize)
            return original.Copy(SKColorType.Rgba8888)!;

        var resized = new SKBitmap(ModelSize, ModelSize, SKColorType.Rgba8888, SKAlphaType.Premul);
        using var canvas = new SKCanvas(resized);
        using var paint = new SKPaint();
        canvas.DrawBitmap(original, new SKRect(0, 0, ModelSize, ModelSize), paint);
        canvas.Flush();
        return resized;
    }

    private static unsafe float[] BitmapToTensor(SKBitmap bitmap)
    {
        const int H = ModelSize, W = ModelSize, C = 3;
        const int planeSize = H * W;
        var tensor = new float[C * planeSize]; // [3, 312, 312]

        var pixels = bitmap.GetPixels();
        if (pixels == IntPtr.Zero)
            throw new InvalidOperationException("Bitmap-Pixel konnten nicht gelesen werden.");

        byte* ptr = (byte*)pixels.ToPointer();

        // RGBA8888 → CHW float (R/G/B / 255.0, Alpha ignorieren)
        for (int y = 0; y < H; y++)
        {
            int rowOffset = y * W;
            for (int x = 0; x < W; x++)
            {
                int pxOffset  = (rowOffset + x) * 4; // 4 Bytes pro Pixel (RGBA)
                float r = ptr[pxOffset    ] / 255f;
                float g = ptr[pxOffset + 1] / 255f;
                float b = ptr[pxOffset + 2] / 255f;

                int idx = rowOffset + x;
                tensor[idx]                    = r;
                tensor[planeSize     + idx]    = g;
                tensor[planeSize * 2 + idx]    = b;
            }
        }

        return tensor;
    }
}

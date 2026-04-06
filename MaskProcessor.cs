using SkiaSharp;

namespace RfDetr.Preprocessing;

/// <summary>
/// Skaliert eine 78×78 Rohmaske (Logit-Werte) auf die Originalgröße des Bilds
/// und erzeugt ein binäres Byte-Array (0 oder 255).
/// </summary>
internal static class MaskProcessor
{
    /// <summary>
    /// Verarbeitet eine 78×78-Maske.
    /// </summary>
    /// <param name="maskLogits">78×78 Rohlogits (Index: [row * 78 + col]).</param>
    /// <param name="originalWidth">Zielbreite in Pixeln.</param>
    /// <param name="originalHeight">Zielhöhe in Pixeln.</param>
    /// <param name="threshold">Sigmoid-Schwellwert für Binärisierung (Standard 0,5).</param>
    /// <returns>
    /// Byte-Array der Länge originalWidth × originalHeight.
    /// 255 = Maske gesetzt, 0 = Hintergrund. Zeilenweise (row-major).
    /// </returns>
    internal static byte[] Upscale(
        ReadOnlySpan<float> maskLogits,
        int originalWidth,
        int originalHeight,
        float threshold = 0.5f)
    {
        const int SrcSize = 78;

        // Sigmoid auf Logits anwenden → Wahrscheinlichkeit in [0, 1]
        using var srcBitmap = new SKBitmap(SrcSize, SrcSize, SKColorType.Gray8, SKAlphaType.Opaque);
        unsafe
        {
            byte* dst = (byte*)srcBitmap.GetPixels().ToPointer();
            for (int i = 0; i < SrcSize * SrcSize; i++)
            {
                float prob = Sigmoid(maskLogits[i]);
                dst[i] = (byte)(prob >= threshold ? 255 : 0);
            }
        }

        // Bilineares Upscaling auf Originalgröße
        var info = new SKImageInfo(originalWidth, originalHeight, SKColorType.Gray8, SKAlphaType.Opaque);
        using var dstBitmap = srcBitmap.Resize(info, new SKSamplingOptions(SKFilterMode.Linear))
            ?? throw new InvalidOperationException("Masken-Upscale fehlgeschlagen.");

        var result = new byte[originalWidth * originalHeight];
        unsafe
        {
            byte* src = (byte*)dstBitmap.GetPixels().ToPointer();
            for (int i = 0; i < result.Length; i++)
                result[i] = (byte)(src[i] >= 128 ? 255 : 0);
        }
        return result;
    }

    private static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));
}

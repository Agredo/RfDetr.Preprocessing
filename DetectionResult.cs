using SkiaSharp;

namespace RfDetr.Preprocessing;

/// <summary>
/// Ein einzelnes Detektionsergebnis des RF-DETR Segmentierungsmodells.
/// Alle Koordinaten beziehen sich auf das Originalbild.
/// </summary>
public sealed class DetectionResult
{
    internal DetectionResult(
        float score,
        SKRectI boundingBox,
        SKRect normalizedBoundingBox,
        byte[] mask,
        int originalWidth,
        int originalHeight)
    {
        Score = score;
        BoundingBox = boundingBox;
        NormalizedBoundingBox = normalizedBoundingBox;
        Mask = mask;
        OriginalWidth = originalWidth;
        OriginalHeight = originalHeight;
    }

    /// <summary>Konfidenz-Score (0–1).</summary>
    public float Score { get; init; }

    /// <summary>Bounding-Box in Originalpixeln (xyxy).</summary>
    public SKRectI BoundingBox { get; init; }

    /// <summary>Bounding-Box normalisiert 0–1 (xyxy).</summary>
    public SKRect NormalizedBoundingBox { get; init; }

    /// <summary>
    /// Segmentierungsmaske in Originalgröße (1 Byte pro Pixel, 0 oder 255).
    /// Breite = OriginalWidth, Höhe = OriginalHeight, row-major.
    /// </summary>
    public byte[] Mask { get; init; } = [];

    /// <summary>Originalbreite des Eingangsbilds in Pixeln.</summary>
    public int OriginalWidth { get; init; }

    /// <summary>Originalhöhe des Eingangsbilds in Pixeln.</summary>
    public int OriginalHeight { get; init; }

    /// <summary>
    /// Extrahiert N Eckpunkte aus der Maske via Douglas-Peucker-Kontur-Approximation.
    /// Gibt Pixelkoordinaten im Originalbild zurück.
    /// </summary>
    /// <param name="cornerCount">Gewünschte Anzahl Ecken (3–8, Standard: 4).</param>
    /// <param name="method">Extraktionsmethode.</param>
    public SKPoint[] GetCorners(int cornerCount = 4, CornerMethod method = CornerMethod.ContourApproximation)
        => CornerExtractor.Extract(Mask, OriginalWidth, OriginalHeight, cornerCount, method);

    /// <summary>
    /// Wie <see cref="GetCorners"/> aber normalisiert (0–1).
    /// </summary>
    public SKPoint[] GetNormalizedCorners(int cornerCount = 4, CornerMethod method = CornerMethod.ContourApproximation)
    {
        var pts = GetCorners(cornerCount, method);
        for (int i = 0; i < pts.Length; i++)
            pts[i] = new SKPoint(pts[i].X / OriginalWidth, pts[i].Y / OriginalHeight);
        return pts;
    }
}

/// <summary>Methode zur Eckpunkt-Extraktion.</summary>
public enum CornerMethod
{
    /// <summary>
    /// Verfolgt die Maske-Kontur und approximiert sie mit Douglas-Peucker
    /// auf die gewünschte Eckanzahl. Liefert die echten Ecken des Dokuments.
    /// </summary>
    ContourApproximation,

    /// <summary>
    /// Vier Ecken der Bounding-Box — immer ein Rechteck.
    /// Ignoriert <c>cornerCount</c> (gibt immer 4 Punkte zurück).
    /// </summary>
    BoundingBox,
}

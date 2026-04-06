using SkiaSharp;

namespace RfDetr.Preprocessing;

/// <summary>
/// Extrahiert Eckpunkte aus einer binären Maske.
///
/// Zwei Methoden stehen zur Verfügung:
/// <list type="bullet">
///   <item><see cref="CornerMethod.ContourApproximation"/> — Konturverfolgung + Douglas-Peucker-Vereinfachung</item>
///   <item><see cref="CornerMethod.BoundingBox"/> — immer genau 4 Ecken der Bounding-Box</item>
/// </list>
/// </summary>
internal static class CornerExtractor
{
    /// <summary>
    /// Extrahiert N Eckpunkte in Pixelkoordinaten.
    /// </summary>
    /// <param name="maskBytes">Binärmaske (0 oder 255), row-major, Länge = width × height.</param>
    /// <param name="width">Breite der Maske (= Originalbildbreite).</param>
    /// <param name="height">Höhe der Maske (= Originalbildhöhe).</param>
    /// <param name="cornerCount">Gewünschte Anzahl Eckpunkte (Standard 4).</param>
    /// <param name="method">Algorithmus.</param>
    /// <returns>SKPoint[] in Pixelkoordinaten. Kann leer sein wenn keine Maske gefunden.</returns>
    internal static SKPoint[] Extract(
        byte[] maskBytes,
        int width,
        int height,
        int cornerCount,
        CornerMethod method)
    {
        return method switch
        {
            CornerMethod.BoundingBox            => BoundingBoxCorners(maskBytes, width, height),
            CornerMethod.ContourApproximation   => ContourCorners(maskBytes, width, height, cornerCount),
            _                                   => BoundingBoxCorners(maskBytes, width, height)
        };
    }

    // ── BoundingBox ───────────────────────────────────────────────────────────

    private static SKPoint[] BoundingBoxCorners(byte[] maskBytes, int width, int height)
    {
        int minX = int.MaxValue, minY = int.MaxValue;
        int maxX = int.MinValue, maxY = int.MinValue;

        for (int y = 0; y < height; y++)
        {
            int row = y * width;
            for (int x = 0; x < width; x++)
            {
                if (maskBytes[row + x] == 0) continue;
                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
            }
        }

        if (minX == int.MaxValue)
            return [];

        return
        [
            new SKPoint(minX, minY),
            new SKPoint(maxX, minY),
            new SKPoint(maxX, maxY),
            new SKPoint(minX, maxY),
        ];
    }

    // ── Kontur-Approximation (Douglas-Peucker) ────────────────────────────────

    private static SKPoint[] ContourCorners(byte[] maskBytes, int width, int height, int targetCorners)
    {
        var contour = TraceContour(maskBytes, width, height);
        if (contour.Count < 3)
            return BoundingBoxCorners(maskBytes, width, height);

        // Epsilon schrittweise erhöhen bis wir ≤ targetCorners Punkte haben
        float epsilon = 1f;
        List<SKPoint> simplified = contour;

        while (simplified.Count > targetCorners && epsilon < 10_000f)
        {
            simplified = DouglasPeucker(contour, epsilon);
            epsilon *= 1.5f;
        }

        // Falls nötig auf genau targetCorners reduzieren (gleichmäßig samplen)
        if (simplified.Count > targetCorners)
            simplified = SampleEvenly(simplified, targetCorners);

        return [.. simplified];
    }

    /// <summary>
    /// Einfache Randverfolgung (Moore-Neighborhood, Schachbrettmuster).
    /// Gibt den äußeren Konturzug als geordnete Punktliste zurück.
    /// </summary>
    private static List<SKPoint> TraceContour(byte[] maskBytes, int width, int height)
    {
        // Startpunkt: erster gesetzter Pixel von oben-links
        int startX = -1, startY = -1;
        for (int y = 0; y < height && startX < 0; y++)
        {
            for (int x = 0; x < width; x++)
            {
                if (maskBytes[y * width + x] != 0) { startX = x; startY = y; break; }
            }
        }

        if (startX < 0)
            return [];

        // 8-Richtungen (Uhrzeigersinn, Start: links)
        ReadOnlySpan<(int dx, int dy)> dirs = stackalloc (int, int)[]
        {
            (-1,  0), (-1, -1), (0, -1), (1, -1),
            ( 1,  0), ( 1,  1), (0,  1), (-1, 1)
        };

        var contour = new List<SKPoint> { new(startX, startY) };
        int cx = startX, cy = startY;
        int prevDir = 0; // aus welcher Richtung wir kamen (Gegen-Richtung durchsuchen zuerst)

        const int MaxSteps = 40_000;
        for (int step = 0; step < MaxSteps; step++)
        {
            // Suche nächsten gesetzten Nachbarn, beginnend links vom Einkommens-Vektor
            int searchStart = (prevDir + 6) % 8; // 180°+90° zurück
            bool found = false;
            for (int d = 0; d < 8; d++)
            {
                int dir = (searchStart + d) % 8;
                int nx = cx + dirs[dir].dx;
                int ny = cy + dirs[dir].dy;
                if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
                if (maskBytes[ny * width + nx] == 0) continue;

                cx = nx;
                cy = ny;
                prevDir = dir;
                found = true;

                if (cx == startX && cy == startY)
                    goto Done;

                contour.Add(new SKPoint(cx, cy));
                break;
            }
            if (!found) break;
        }
        Done:
        return contour;
    }

    /// <summary>Rekursiver Douglas-Peucker-Algorithmus.</summary>
    private static List<SKPoint> DouglasPeucker(List<SKPoint> points, float epsilon)
    {
        if (points.Count <= 2)
            return new List<SKPoint>(points);

        // Maximalen Abstand zur Linie start→end finden
        float maxDist = 0f;
        int maxIdx  = 0;

        SKPoint start = points[0];
        SKPoint end   = points[^1];

        for (int i = 1; i < points.Count - 1; i++)
        {
            float d = PerpendicularDistance(points[i], start, end);
            if (d > maxDist) { maxDist = d; maxIdx = i; }
        }

        if (maxDist <= epsilon)
            return [start, end];

        var left  = DouglasPeucker(points[..( maxIdx + 1)], epsilon);
        var right = DouglasPeucker(points[maxIdx..],        epsilon);

        // Zusammenführen (Mittelpunkt nicht doppelt)
        left.RemoveAt(left.Count - 1);
        left.AddRange(right);
        return left;
    }

    private static float PerpendicularDistance(SKPoint p, SKPoint a, SKPoint b)
    {
        float dx = b.X - a.X;
        float dy = b.Y - a.Y;
        float len = MathF.Sqrt(dx * dx + dy * dy);
        if (len < 1e-6f) return MathF.Sqrt((p.X - a.X) * (p.X - a.X) + (p.Y - a.Y) * (p.Y - a.Y));
        return MathF.Abs(dy * p.X - dx * p.Y + b.X * a.Y - b.Y * a.X) / len;
    }

    private static List<SKPoint> SampleEvenly(List<SKPoint> points, int n)
    {
        if (points.Count <= n) return points;
        var result = new List<SKPoint>(n);
        float step = (float)(points.Count - 1) / (n - 1);
        for (int i = 0; i < n; i++)
            result.Add(points[(int)MathF.Round(i * step)]);
        return result;
    }
}

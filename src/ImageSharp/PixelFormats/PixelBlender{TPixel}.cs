// Copyright (c) Six Labors.
// Licensed under the Six Labors Split License.

using System.Buffers;
using System.Numerics;
using System.Runtime.InteropServices;
using SixLabors.ImageSharp.PixelFormats.PixelBlenders;

namespace SixLabors.ImageSharp.PixelFormats;

/// <summary>
/// Defines operations that blend pixels and pixel rows.
/// </summary>
/// <typeparam name="TPixel">The destination pixel format.</typeparam>
public abstract class PixelBlender<TPixel>
    where TPixel : unmanaged, IPixel<TPixel>
{
    // PixelBlenders selects shared implementations of this type. This base class manages row buffers and composition calls.
    // Shared implementations supply composition arithmetic and select the alpha representation required by each equation.
    // PixelOperations owns pixel-buffer conversion, including format-specific overrides. Pixel formats own rounding and storage layout.

    /// <summary>
    /// Gets a value indicating whether an opaque constant source at amount 1 blends to the source pixel itself at
    /// coverage 1, for every backdrop, so that such pixels can be written without the blend.
    /// </summary>
    /// <remarks>
    /// Only a blender whose equations and pixel format make that exact may return <see langword="true"/>.
    /// </remarks>
    private protected virtual bool CopiesOpaqueSourceAtFullCoverage => false;

    /// <summary>
    /// Blend 2 pixels together.
    /// </summary>
    /// <param name="background">The background color.</param>
    /// <param name="source">The source color.</param>
    /// <param name="amount">
    /// A value between 0 and 1 indicating the weight of the second source vector.
    /// At amount = 0, "background" is returned, at amount = 1, "source" is returned.
    /// </param>
    /// <returns>The final pixel value after composition.</returns>
    public abstract TPixel Blend(TPixel background, TPixel source, float amount);

    /// <summary>
    /// Blends 2 rows together
    /// </summary>
    /// <typeparam name="TPixelSrc">the pixel format of the source span</typeparam>
    /// <param name="configuration"><see cref="Configuration"/> to use internally</param>
    /// <param name="destination">the destination span</param>
    /// <param name="background">the background span</param>
    /// <param name="source">the source span</param>
    /// <param name="amount">
    /// A value between 0 and 1 indicating the weight of the second source vector.
    /// At amount = 0, "background" is returned, at amount = 1, "source" is returned.
    /// </param>
    public void Blend<TPixelSrc>(
        Configuration configuration,
        Span<TPixel> destination,
        ReadOnlySpan<TPixel> background,
        ReadOnlySpan<TPixelSrc> source,
        float amount)
        where TPixelSrc : unmanaged, IPixel<TPixelSrc>
    {
        int maxLength = destination.Length;
        Guard.MustBeGreaterThanOrEqualTo(background.Length, maxLength, nameof(background.Length));
        Guard.MustBeGreaterThanOrEqualTo(source.Length, maxLength, nameof(source.Length));
        Guard.MustBeBetweenOrEqualTo(amount, 0, 1, nameof(amount));

        using IMemoryOwner<Vector4> buffer = configuration.MemoryAllocator.Allocate<Vector4>(maxLength * 3);
        this.Blend(
            configuration,
            destination,
            background,
            source,
            amount,
            buffer.Memory.Span[..(maxLength * 3)]);
    }

    /// <summary>
    /// Blends 2 rows together using caller-provided temporary vector scratch.
    /// </summary>
    /// <typeparam name="TPixelSrc">the pixel format of the source span</typeparam>
    /// <param name="configuration"><see cref="Configuration"/> to use internally</param>
    /// <param name="destination">the destination span</param>
    /// <param name="background">the background span</param>
    /// <param name="source">the source span</param>
    /// <param name="amount">
    /// A value between 0 and 1 indicating the weight of the second source vector.
    /// At amount = 0, "background" is returned, at amount = 1, "source" is returned.
    /// </param>
    /// <param name="workingBuffer">Reusable temporary vector scratch with capacity for at least 3 rows.</param>
    public void Blend<TPixelSrc>(
        Configuration configuration,
        Span<TPixel> destination,
        ReadOnlySpan<TPixel> background,
        ReadOnlySpan<TPixelSrc> source,
        float amount,
        Span<Vector4> workingBuffer)
        where TPixelSrc : unmanaged, IPixel<TPixelSrc>
    {
        int maxLength = destination.Length;
        Guard.MustBeGreaterThanOrEqualTo(background.Length, maxLength, nameof(background.Length));
        Guard.MustBeGreaterThanOrEqualTo(source.Length, maxLength, nameof(source.Length));
        Guard.MustBeBetweenOrEqualTo(amount, 0, 1, nameof(amount));
        Guard.MustBeGreaterThanOrEqualTo(workingBuffer.Length, maxLength * 3, nameof(workingBuffer.Length));

        Span<Vector4> destinationVectors = workingBuffer[..maxLength];
        Span<Vector4> backgroundVectors = workingBuffer.Slice(maxLength, maxLength);
        Span<Vector4> sourceVectors = workingBuffer.Slice(maxLength * 2, maxLength);

        this.ToBlendVector4(configuration, background[..maxLength], backgroundVectors);
        this.ToBlendVector4(configuration, source[..maxLength], sourceVectors);

        this.BlendFunction(destinationVectors, backgroundVectors, sourceVectors, amount);

        this.FromBlendVector4(configuration, destinationVectors, destination);
    }

    /// <summary>
    /// Blends a row against a constant source color.
    /// </summary>
    /// <param name="configuration"><see cref="Configuration"/> to use internally</param>
    /// <param name="destination">the destination span</param>
    /// <param name="background">the background span</param>
    /// <param name="source">the source color</param>
    /// <param name="amount">
    /// A value between 0 and 1 indicating the weight of the second source vector.
    /// At amount = 0, "background" is returned, at amount = 1, "source" is returned.
    /// </param>
    public void Blend(
        Configuration configuration,
        Span<TPixel> destination,
        ReadOnlySpan<TPixel> background,
        TPixel source,
        float amount)
    {
        int maxLength = destination.Length;
        Guard.MustBeGreaterThanOrEqualTo(background.Length, maxLength, nameof(background.Length));
        Guard.MustBeBetweenOrEqualTo(amount, 0, 1, nameof(amount));

        using IMemoryOwner<Vector4> buffer = configuration.MemoryAllocator.Allocate<Vector4>(maxLength * 2);
        this.Blend(
            configuration,
            destination,
            background,
            source,
            amount,
            buffer.Memory.Span[..(maxLength * 2)]);
    }

    /// <summary>
    /// Blends a row against a constant source color using caller-provided temporary vector scratch.
    /// </summary>
    /// <param name="configuration"><see cref="Configuration"/> to use internally</param>
    /// <param name="destination">the destination span</param>
    /// <param name="background">the background span</param>
    /// <param name="source">the source color</param>
    /// <param name="amount">
    /// A value between 0 and 1 indicating the weight of the second source vector.
    /// At amount = 0, "background" is returned, at amount = 1, "source" is returned.
    /// </param>
    /// <param name="workingBuffer">Reusable temporary vector scratch with capacity for at least 2 rows.</param>
    public void Blend(
        Configuration configuration,
        Span<TPixel> destination,
        ReadOnlySpan<TPixel> background,
        TPixel source,
        float amount,
        Span<Vector4> workingBuffer)
    {
        int maxLength = destination.Length;
        Guard.MustBeGreaterThanOrEqualTo(background.Length, maxLength, nameof(background.Length));
        Guard.MustBeBetweenOrEqualTo(amount, 0, 1, nameof(amount));
        Guard.MustBeGreaterThanOrEqualTo(workingBuffer.Length, maxLength * 2, nameof(workingBuffer.Length));

        Span<Vector4> destinationVectors = workingBuffer[..maxLength];
        Span<Vector4> backgroundVectors = workingBuffer.Slice(maxLength, maxLength);

        this.ToBlendVector4(configuration, background[..maxLength], backgroundVectors);

        this.BlendFunction(destinationVectors, backgroundVectors, this.ToBlendVector4(source), amount);

        this.FromBlendVector4(configuration, destinationVectors, destination);
    }

    /// <summary>
    /// Blends 2 rows together with per-pixel coverage.
    /// </summary>
    /// <typeparam name="TPixelSrc">the pixel format of the source span</typeparam>
    /// <param name="configuration"><see cref="Configuration"/> to use internally</param>
    /// <param name="destination">the destination span</param>
    /// <param name="background">the background span</param>
    /// <param name="source">the source span</param>
    /// <param name="amount">
    /// A value between 0 and 1 indicating the weight of the second source vector.
    /// At amount = 0, "background" is returned, at amount = 1, "source" is returned.
    /// </param>
    /// <param name="coverage">A span with coverage values between 0 and 1.</param>
    public void BlendWithCoverage<TPixelSrc>(
        Configuration configuration,
        Span<TPixel> destination,
        ReadOnlySpan<TPixel> background,
        ReadOnlySpan<TPixelSrc> source,
        float amount,
        ReadOnlySpan<float> coverage)
        where TPixelSrc : unmanaged, IPixel<TPixelSrc>
    {
        int maxLength = destination.Length;
        Guard.MustBeGreaterThanOrEqualTo(background.Length, maxLength, nameof(background.Length));
        Guard.MustBeGreaterThanOrEqualTo(source.Length, maxLength, nameof(source.Length));
        Guard.MustBeBetweenOrEqualTo(amount, 0, 1, nameof(amount));
        Guard.MustBeGreaterThanOrEqualTo(coverage.Length, maxLength, nameof(coverage.Length));

        using IMemoryOwner<Vector4> buffer = configuration.MemoryAllocator.Allocate<Vector4>(maxLength * 3);
        this.BlendWithCoverage(
            configuration,
            destination,
            background,
            source,
            amount,
            coverage,
            buffer.Memory.Span[..(maxLength * 3)]);
    }

    /// <summary>
    /// Blends 2 rows together with per-pixel coverage using caller-provided temporary vector scratch.
    /// </summary>
    /// <typeparam name="TPixelSrc">the pixel format of the source span</typeparam>
    /// <param name="configuration"><see cref="Configuration"/> to use internally</param>
    /// <param name="destination">the destination span</param>
    /// <param name="background">the background span</param>
    /// <param name="source">the source span</param>
    /// <param name="amount">
    /// A value between 0 and 1 indicating the weight of the second source vector.
    /// At amount = 0, "background" is returned, at amount = 1, "source" is returned.
    /// </param>
    /// <param name="coverage">A span with coverage values between 0 and 1.</param>
    /// <param name="workingBuffer">Reusable temporary vector scratch with capacity for at least 3 rows.</param>
    public void BlendWithCoverage<TPixelSrc>(
        Configuration configuration,
        Span<TPixel> destination,
        ReadOnlySpan<TPixel> background,
        ReadOnlySpan<TPixelSrc> source,
        float amount,
        ReadOnlySpan<float> coverage,
        Span<Vector4> workingBuffer)
        where TPixelSrc : unmanaged, IPixel<TPixelSrc>
    {
        int maxLength = destination.Length;
        Guard.MustBeGreaterThanOrEqualTo(background.Length, maxLength, nameof(background.Length));
        Guard.MustBeGreaterThanOrEqualTo(source.Length, maxLength, nameof(source.Length));
        Guard.MustBeBetweenOrEqualTo(amount, 0, 1, nameof(amount));
        Guard.MustBeGreaterThanOrEqualTo(coverage.Length, maxLength, nameof(coverage.Length));
        Guard.MustBeGreaterThanOrEqualTo(workingBuffer.Length, maxLength * 3, nameof(workingBuffer.Length));

        Span<Vector4> destinationVectors = workingBuffer[..maxLength];
        Span<Vector4> backgroundVectors = workingBuffer.Slice(maxLength, maxLength);
        Span<Vector4> sourceVectors = workingBuffer.Slice(maxLength * 2, maxLength);

        this.ToBlendVector4(configuration, background[..maxLength], backgroundVectors);
        this.ToBlendVector4(configuration, source[..maxLength], sourceVectors);

        this.BlendWithCoverageFunction(destinationVectors, backgroundVectors, sourceVectors, amount, coverage);

        this.FromBlendVector4(configuration, destinationVectors, destination);
    }

    /// <summary>
    /// Blends a row against a constant source color with per-pixel coverage.
    /// </summary>
    /// <param name="configuration"><see cref="Configuration"/> to use internally</param>
    /// <param name="destination">the destination span</param>
    /// <param name="background">the background span</param>
    /// <param name="source">the source color</param>
    /// <param name="amount">
    /// A value between 0 and 1 indicating the weight of the second source vector.
    /// At amount = 0, "background" is returned, at amount = 1, "source" is returned.
    /// </param>
    /// <param name="coverage">A span with coverage values between 0 and 1.</param>
    public void BlendWithCoverage(
        Configuration configuration,
        Span<TPixel> destination,
        ReadOnlySpan<TPixel> background,
        TPixel source,
        float amount,
        ReadOnlySpan<float> coverage)
    {
        int maxLength = destination.Length;
        Guard.MustBeGreaterThanOrEqualTo(background.Length, maxLength, nameof(background.Length));
        Guard.MustBeBetweenOrEqualTo(amount, 0, 1, nameof(amount));
        Guard.MustBeGreaterThanOrEqualTo(coverage.Length, maxLength, nameof(coverage.Length));

        using IMemoryOwner<Vector4> buffer = configuration.MemoryAllocator.Allocate<Vector4>(maxLength * 2);
        this.BlendWithCoverage(
            configuration,
            destination,
            background,
            source,
            amount,
            coverage,
            buffer.Memory.Span[..(maxLength * 2)]);
    }

    /// <summary>
    /// Blends a row against a constant source color with per-pixel coverage using caller-provided temporary vector scratch.
    /// </summary>
    /// <param name="configuration"><see cref="Configuration"/> to use internally</param>
    /// <param name="destination">the destination span</param>
    /// <param name="background">the background span</param>
    /// <param name="source">the source color</param>
    /// <param name="amount">
    /// A value between 0 and 1 indicating the weight of the second source vector.
    /// At amount = 0, "background" is returned, at amount = 1, "source" is returned.
    /// </param>
    /// <param name="coverage">A span with coverage values between 0 and 1.</param>
    /// <param name="workingBuffer">Reusable temporary vector scratch with capacity for at least 2 rows.</param>
    public void BlendWithCoverage(
        Configuration configuration,
        Span<TPixel> destination,
        ReadOnlySpan<TPixel> background,
        TPixel source,
        float amount,
        ReadOnlySpan<float> coverage,
        Span<Vector4> workingBuffer)
    {
        int maxLength = destination.Length;
        Guard.MustBeGreaterThanOrEqualTo(background.Length, maxLength, nameof(background.Length));
        Guard.MustBeBetweenOrEqualTo(amount, 0, 1, nameof(amount));
        Guard.MustBeGreaterThanOrEqualTo(coverage.Length, maxLength, nameof(coverage.Length));
        Guard.MustBeGreaterThanOrEqualTo(workingBuffer.Length, maxLength * 2, nameof(workingBuffer.Length));

        // Rows shorter than 32 pixels have little to copy, and the search for runs and the extra blend calls cost more
        // than the copy saves when the coverage has no run of 1 (measured on 15-pixel rows).
        if (maxLength >= 32 && amount == 1F && this.CopiesOpaqueSourceAtFullCoverage && source.ToScaledVector4().W == 1F)
        {
            this.BlendWithCoverageCopyingFullCoverage(configuration, destination[..maxLength], background, source, coverage, workingBuffer);
            return;
        }

        this.BlendWithCoverageCore(configuration, destination[..maxLength], background, source, amount, coverage, workingBuffer);
    }

    /// <summary>
    /// Blends a row against an opaque constant source at amount 1: groups of 4 pixels whose coverage is exactly 1 receive
    /// the source pixel, and the other pixels are blended through <see cref="BlendWithCoverageCore"/>.
    /// </summary>
    /// <remarks>
    /// The vector blend runs over the first multiple of 4 pixels of a call (512-bit; 2 with 256-bit vectors) and the
    /// scalar blend over the rest, and the two can differ in the last bit. Every blended piece therefore starts at a
    /// multiple of 4 pixels of the row, and the last piece never holds only the last 2 or 3 pixels, so every pixel is
    /// blended by the same loop as in one call for the whole row and the blended pixels keep their bytes.
    /// The pieces are disjoint, so a destination that is the background is read before it is written.
    /// </remarks>
    /// <param name="configuration">The configuration.</param>
    /// <param name="destination">The destination pixels, sliced to the row length.</param>
    /// <param name="background">The background pixels.</param>
    /// <param name="source">The opaque source pixel.</param>
    /// <param name="coverage">The coverage values.</param>
    /// <param name="workingBuffer">The vector scratch, at least 2 rows.</param>
    private void BlendWithCoverageCopyingFullCoverage(
        Configuration configuration,
        Span<TPixel> destination,
        ReadOnlySpan<TPixel> background,
        TPixel source,
        ReadOnlySpan<float> coverage,
        Span<Vector4> workingBuffer)
    {
        const uint fullCoverage = 0x3F80_0000; // The bits of 1F; other values, also -0 and NaN, have other bits.
        int length = destination.Length;

        // Groups that may be copied. A remainder of 2 or 3 pixels stays with the group before it (see the remarks).
        int groupEnd = length & ~3;
        if ((length & 3) >= 2)
        {
            groupEnd -= 4;
        }

        ReadOnlySpan<uint> bits = MemoryMarshal.Cast<float, uint>(coverage[..length]);
        int pending = 0;
        int i = 0;
        while (i < groupEnd)
        {
            int found = bits[i..groupEnd].IndexOf(fullCoverage);
            if (found < 0)
            {
                break;
            }

            int runStart = (i + found + 3) & ~3;
            if (runStart >= groupEnd)
            {
                break;
            }

            int other = bits[runStart..groupEnd].IndexOfAnyExcept(fullCoverage);
            int runEnd = other < 0 ? groupEnd : (runStart + other) & ~3;
            if (runEnd == runStart)
            {
                // The group at runStart has a value other than 1; search again after it.
                i = runStart + other + 1;
                continue;
            }

            if (runStart > pending)
            {
                this.BlendWithCoverageCore(
                    configuration,
                    destination[pending..runStart],
                    background[pending..runStart],
                    source,
                    1F,
                    coverage[pending..runStart],
                    workingBuffer);
            }

            destination[runStart..runEnd].Fill(source);
            pending = runEnd;
            i = runEnd;
        }

        if (pending < length)
        {
            this.BlendWithCoverageCore(
                configuration,
                destination[pending..],
                background[pending..length],
                source,
                1F,
                coverage[pending..length],
                workingBuffer);
        }
    }

    /// <summary>
    /// Blends a row against a constant source color with per-pixel coverage through the vector rows of the blender.
    /// </summary>
    /// <param name="configuration">The configuration.</param>
    /// <param name="destination">The destination pixels; the length is the row length.</param>
    /// <param name="background">The background pixels.</param>
    /// <param name="source">The source pixel.</param>
    /// <param name="amount">The amount, from 0 to 1.</param>
    /// <param name="coverage">The coverage values.</param>
    /// <param name="workingBuffer">The vector scratch, at least 2 rows.</param>
    private void BlendWithCoverageCore(
        Configuration configuration,
        Span<TPixel> destination,
        ReadOnlySpan<TPixel> background,
        TPixel source,
        float amount,
        ReadOnlySpan<float> coverage,
        Span<Vector4> workingBuffer)
    {
        int maxLength = destination.Length;
        Span<Vector4> destinationVectors = workingBuffer[..maxLength];
        Span<Vector4> backgroundVectors = workingBuffer.Slice(maxLength, maxLength);

        this.ToBlendVector4(configuration, background[..maxLength], backgroundVectors);

        this.BlendWithCoverageFunction(destinationVectors, backgroundVectors, this.ToBlendVector4(source), amount, coverage[..maxLength]);

        this.FromBlendVector4(configuration, destinationVectors, destination);
    }

    /// <summary>
    /// Blends 2 rows together
    /// </summary>
    /// <param name="configuration"><see cref="Configuration"/> to use internally</param>
    /// <param name="destination">the destination span</param>
    /// <param name="background">the background span</param>
    /// <param name="source">the source span</param>
    /// <param name="amount">
    /// A span with values between 0 and 1 indicating the weight of the second source vector.
    /// At amount = 0, "background" is returned, at amount = 1, "source" is returned.
    /// </param>
    public void Blend(
        Configuration configuration,
        Span<TPixel> destination,
        ReadOnlySpan<TPixel> background,
        ReadOnlySpan<TPixel> source,
        ReadOnlySpan<float> amount)
        => this.Blend<TPixel>(configuration, destination, background, source, amount);

    /// <summary>
    /// Blends 2 rows together using caller-provided temporary vector scratch.
    /// </summary>
    /// <param name="configuration"><see cref="Configuration"/> to use internally</param>
    /// <param name="destination">the destination span</param>
    /// <param name="background">the background span</param>
    /// <param name="source">the source span</param>
    /// <param name="amount">
    /// A span with values between 0 and 1 indicating the weight of the second source vector.
    /// At amount = 0, "background" is returned, at amount = 1, "source" is returned.
    /// </param>
    /// <param name="workingBuffer">Reusable temporary vector scratch with capacity for at least 3 rows.</param>
    public void Blend(
        Configuration configuration,
        Span<TPixel> destination,
        ReadOnlySpan<TPixel> background,
        ReadOnlySpan<TPixel> source,
        ReadOnlySpan<float> amount,
        Span<Vector4> workingBuffer)
        => this.Blend<TPixel>(configuration, destination, background, source, amount, workingBuffer);

    /// <summary>
    /// Blends 2 rows together
    /// </summary>
    /// <typeparam name="TPixelSrc">the pixel format of the source span</typeparam>
    /// <param name="configuration"><see cref="Configuration"/> to use internally</param>
    /// <param name="destination">the destination span</param>
    /// <param name="background">the background span</param>
    /// <param name="source">the source span</param>
    /// <param name="amount">
    /// A span with values between 0 and 1 indicating the weight of the second source vector.
    /// At amount = 0, "background" is returned, at amount = 1, "source" is returned.
    /// </param>
    public void Blend<TPixelSrc>(
        Configuration configuration,
        Span<TPixel> destination,
        ReadOnlySpan<TPixel> background,
        ReadOnlySpan<TPixelSrc> source,
        ReadOnlySpan<float> amount)
        where TPixelSrc : unmanaged, IPixel<TPixelSrc>
    {
        int maxLength = destination.Length;
        Guard.MustBeGreaterThanOrEqualTo(background.Length, maxLength, nameof(background.Length));
        Guard.MustBeGreaterThanOrEqualTo(source.Length, maxLength, nameof(source.Length));
        Guard.MustBeGreaterThanOrEqualTo(amount.Length, maxLength, nameof(amount.Length));

        using IMemoryOwner<Vector4> buffer = configuration.MemoryAllocator.Allocate<Vector4>(maxLength * 3);
        this.Blend(
            configuration,
            destination,
            background,
            source,
            amount,
            buffer.Memory.Span[..(maxLength * 3)]);
    }

    /// <summary>
    /// Blends a row against a constant source color.
    /// </summary>
    /// <param name="configuration"><see cref="Configuration"/> to use internally</param>
    /// <param name="destination">the destination span</param>
    /// <param name="background">the background span</param>
    /// <param name="source">the source color</param>
    /// <param name="amount">
    /// A span with values between 0 and 1 indicating the weight of the second source vector.
    /// At amount = 0, "background" is returned, at amount = 1, "source" is returned.
    /// </param>
    public void Blend(
        Configuration configuration,
        Span<TPixel> destination,
        ReadOnlySpan<TPixel> background,
        TPixel source,
        ReadOnlySpan<float> amount)
    {
        int maxLength = destination.Length;
        Guard.MustBeGreaterThanOrEqualTo(background.Length, maxLength, nameof(background.Length));
        Guard.MustBeGreaterThanOrEqualTo(amount.Length, maxLength, nameof(amount.Length));

        using IMemoryOwner<Vector4> buffer = configuration.MemoryAllocator.Allocate<Vector4>(maxLength * 2);
        this.Blend(
            configuration,
            destination,
            background,
            source,
            amount,
            buffer.Memory.Span[..(maxLength * 2)]);
    }

    /// <summary>
    /// Blends 2 rows together using caller-provided temporary vector scratch.
    /// </summary>
    /// <typeparam name="TPixelSrc">the pixel format of the source span</typeparam>
    /// <param name="configuration"><see cref="Configuration"/> to use internally</param>
    /// <param name="destination">the destination span</param>
    /// <param name="background">the background span</param>
    /// <param name="source">the source span</param>
    /// <param name="amount">
    /// A span with values between 0 and 1 indicating the weight of the second source vector.
    /// At amount = 0, "background" is returned, at amount = 1, "source" is returned.
    /// </param>
    /// <param name="workingBuffer">Reusable temporary vector scratch with capacity for at least 3 rows.</param>
    public void Blend<TPixelSrc>(
        Configuration configuration,
        Span<TPixel> destination,
        ReadOnlySpan<TPixel> background,
        ReadOnlySpan<TPixelSrc> source,
        ReadOnlySpan<float> amount,
        Span<Vector4> workingBuffer)
        where TPixelSrc : unmanaged, IPixel<TPixelSrc>
    {
        int maxLength = destination.Length;
        Guard.MustBeGreaterThanOrEqualTo(background.Length, maxLength, nameof(background.Length));
        Guard.MustBeGreaterThanOrEqualTo(source.Length, maxLength, nameof(source.Length));
        Guard.MustBeGreaterThanOrEqualTo(amount.Length, maxLength, nameof(amount.Length));
        Guard.MustBeGreaterThanOrEqualTo(workingBuffer.Length, maxLength * 3, nameof(workingBuffer.Length));

        Span<Vector4> destinationVectors = workingBuffer[..maxLength];
        Span<Vector4> backgroundVectors = workingBuffer.Slice(maxLength, maxLength);
        Span<Vector4> sourceVectors = workingBuffer.Slice(maxLength * 2, maxLength);

        this.ToBlendVector4(configuration, background[..maxLength], backgroundVectors);
        this.ToBlendVector4(configuration, source[..maxLength], sourceVectors);

        this.BlendFunction(destinationVectors, backgroundVectors, sourceVectors, amount);

        this.FromBlendVector4(configuration, destinationVectors, destination);
    }

    /// <summary>
    /// Blends a row against a constant source color using caller-provided temporary vector scratch.
    /// </summary>
    /// <param name="configuration"><see cref="Configuration"/> to use internally</param>
    /// <param name="destination">the destination span</param>
    /// <param name="background">the background span</param>
    /// <param name="source">the source color</param>
    /// <param name="amount">
    /// A span with values between 0 and 1 indicating the weight of the second source vector.
    /// At amount = 0, "background" is returned, at amount = 1, "source" is returned.
    /// </param>
    /// <param name="workingBuffer">Reusable temporary vector scratch with capacity for at least 2 rows.</param>
    public void Blend(
        Configuration configuration,
        Span<TPixel> destination,
        ReadOnlySpan<TPixel> background,
        TPixel source,
        ReadOnlySpan<float> amount,
        Span<Vector4> workingBuffer)
    {
        int maxLength = destination.Length;
        Guard.MustBeGreaterThanOrEqualTo(background.Length, maxLength, nameof(background.Length));
        Guard.MustBeGreaterThanOrEqualTo(amount.Length, maxLength, nameof(amount.Length));
        Guard.MustBeGreaterThanOrEqualTo(workingBuffer.Length, maxLength * 2, nameof(workingBuffer.Length));

        Span<Vector4> destinationVectors = workingBuffer[..maxLength];
        Span<Vector4> backgroundVectors = workingBuffer.Slice(maxLength, maxLength);

        this.ToBlendVector4(configuration, background[..maxLength], backgroundVectors);

        this.BlendFunction(destinationVectors, backgroundVectors, this.ToBlendVector4(source), amount);

        this.FromBlendVector4(configuration, destinationVectors, destination);
    }

    /// <summary>
    /// Blends 2 rows together with per-pixel coverage.
    /// </summary>
    /// <param name="configuration"><see cref="Configuration"/> to use internally</param>
    /// <param name="destination">the destination span</param>
    /// <param name="background">the background span</param>
    /// <param name="source">the source span</param>
    /// <param name="amount">
    /// A span with values between 0 and 1 indicating the weight of the second source vector.
    /// At amount = 0, "background" is returned, at amount = 1, "source" is returned.
    /// </param>
    /// <param name="coverage">A span with coverage values between 0 and 1.</param>
    public void BlendWithCoverage(
        Configuration configuration,
        Span<TPixel> destination,
        ReadOnlySpan<TPixel> background,
        ReadOnlySpan<TPixel> source,
        ReadOnlySpan<float> amount,
        ReadOnlySpan<float> coverage)
        => this.BlendWithCoverage<TPixel>(configuration, destination, background, source, amount, coverage);

    /// <summary>
    /// Blends 2 rows together with per-pixel coverage using caller-provided temporary vector scratch.
    /// </summary>
    /// <param name="configuration"><see cref="Configuration"/> to use internally</param>
    /// <param name="destination">the destination span</param>
    /// <param name="background">the background span</param>
    /// <param name="source">the source span</param>
    /// <param name="amount">
    /// A span with values between 0 and 1 indicating the weight of the second source vector.
    /// At amount = 0, "background" is returned, at amount = 1, "source" is returned.
    /// </param>
    /// <param name="coverage">A span with coverage values between 0 and 1.</param>
    /// <param name="workingBuffer">Reusable temporary vector scratch with capacity for at least 3 rows.</param>
    public void BlendWithCoverage(
        Configuration configuration,
        Span<TPixel> destination,
        ReadOnlySpan<TPixel> background,
        ReadOnlySpan<TPixel> source,
        ReadOnlySpan<float> amount,
        ReadOnlySpan<float> coverage,
        Span<Vector4> workingBuffer)
        => this.BlendWithCoverage<TPixel>(configuration, destination, background, source, amount, coverage, workingBuffer);

    /// <summary>
    /// Blends 2 rows together with per-pixel coverage.
    /// </summary>
    /// <typeparam name="TPixelSrc">the pixel format of the source span</typeparam>
    /// <param name="configuration"><see cref="Configuration"/> to use internally</param>
    /// <param name="destination">the destination span</param>
    /// <param name="background">the background span</param>
    /// <param name="source">the source span</param>
    /// <param name="amount">
    /// A span with values between 0 and 1 indicating the weight of the second source vector.
    /// At amount = 0, "background" is returned, at amount = 1, "source" is returned.
    /// </param>
    /// <param name="coverage">A span with coverage values between 0 and 1.</param>
    public void BlendWithCoverage<TPixelSrc>(
        Configuration configuration,
        Span<TPixel> destination,
        ReadOnlySpan<TPixel> background,
        ReadOnlySpan<TPixelSrc> source,
        ReadOnlySpan<float> amount,
        ReadOnlySpan<float> coverage)
        where TPixelSrc : unmanaged, IPixel<TPixelSrc>
    {
        int maxLength = destination.Length;
        Guard.MustBeGreaterThanOrEqualTo(background.Length, maxLength, nameof(background.Length));
        Guard.MustBeGreaterThanOrEqualTo(source.Length, maxLength, nameof(source.Length));
        Guard.MustBeGreaterThanOrEqualTo(amount.Length, maxLength, nameof(amount.Length));
        Guard.MustBeGreaterThanOrEqualTo(coverage.Length, maxLength, nameof(coverage.Length));

        using IMemoryOwner<Vector4> buffer = configuration.MemoryAllocator.Allocate<Vector4>(maxLength * 3);
        this.BlendWithCoverage(
            configuration,
            destination,
            background,
            source,
            amount,
            coverage,
            buffer.Memory.Span[..(maxLength * 3)]);
    }

    /// <summary>
    /// Blends a row against a constant source color with per-pixel coverage.
    /// </summary>
    /// <param name="configuration"><see cref="Configuration"/> to use internally</param>
    /// <param name="destination">the destination span</param>
    /// <param name="background">the background span</param>
    /// <param name="source">the source color</param>
    /// <param name="amount">
    /// A span with values between 0 and 1 indicating the weight of the second source vector.
    /// At amount = 0, "background" is returned, at amount = 1, "source" is returned.
    /// </param>
    /// <param name="coverage">A span with coverage values between 0 and 1.</param>
    public void BlendWithCoverage(
        Configuration configuration,
        Span<TPixel> destination,
        ReadOnlySpan<TPixel> background,
        TPixel source,
        ReadOnlySpan<float> amount,
        ReadOnlySpan<float> coverage)
    {
        int maxLength = destination.Length;
        Guard.MustBeGreaterThanOrEqualTo(background.Length, maxLength, nameof(background.Length));
        Guard.MustBeGreaterThanOrEqualTo(amount.Length, maxLength, nameof(amount.Length));
        Guard.MustBeGreaterThanOrEqualTo(coverage.Length, maxLength, nameof(coverage.Length));

        using IMemoryOwner<Vector4> buffer = configuration.MemoryAllocator.Allocate<Vector4>(maxLength * 2);
        this.BlendWithCoverage(
            configuration,
            destination,
            background,
            source,
            amount,
            coverage,
            buffer.Memory.Span[..(maxLength * 2)]);
    }

    /// <summary>
    /// Blends 2 rows together with per-pixel coverage using caller-provided temporary vector scratch.
    /// </summary>
    /// <typeparam name="TPixelSrc">the pixel format of the source span</typeparam>
    /// <param name="configuration"><see cref="Configuration"/> to use internally</param>
    /// <param name="destination">the destination span</param>
    /// <param name="background">the background span</param>
    /// <param name="source">the source span</param>
    /// <param name="amount">
    /// A span with values between 0 and 1 indicating the weight of the second source vector.
    /// At amount = 0, "background" is returned, at amount = 1, "source" is returned.
    /// </param>
    /// <param name="coverage">A span with coverage values between 0 and 1.</param>
    /// <param name="workingBuffer">Reusable temporary vector scratch with capacity for at least 3 rows.</param>
    public void BlendWithCoverage<TPixelSrc>(
        Configuration configuration,
        Span<TPixel> destination,
        ReadOnlySpan<TPixel> background,
        ReadOnlySpan<TPixelSrc> source,
        ReadOnlySpan<float> amount,
        ReadOnlySpan<float> coverage,
        Span<Vector4> workingBuffer)
        where TPixelSrc : unmanaged, IPixel<TPixelSrc>
    {
        int maxLength = destination.Length;
        Guard.MustBeGreaterThanOrEqualTo(background.Length, maxLength, nameof(background.Length));
        Guard.MustBeGreaterThanOrEqualTo(source.Length, maxLength, nameof(source.Length));
        Guard.MustBeGreaterThanOrEqualTo(amount.Length, maxLength, nameof(amount.Length));
        Guard.MustBeGreaterThanOrEqualTo(coverage.Length, maxLength, nameof(coverage.Length));
        Guard.MustBeGreaterThanOrEqualTo(workingBuffer.Length, maxLength * 3, nameof(workingBuffer.Length));

        Span<Vector4> destinationVectors = workingBuffer[..maxLength];
        Span<Vector4> backgroundVectors = workingBuffer.Slice(maxLength, maxLength);
        Span<Vector4> sourceVectors = workingBuffer.Slice(maxLength * 2, maxLength);

        this.ToBlendVector4(configuration, background[..maxLength], backgroundVectors);
        this.ToBlendVector4(configuration, source[..maxLength], sourceVectors);

        this.BlendWithCoverageFunction(destinationVectors, backgroundVectors, sourceVectors, amount, coverage);

        this.FromBlendVector4(configuration, destinationVectors, destination);
    }

    /// <summary>
    /// Blends a row against a constant source color with per-pixel coverage using caller-provided temporary vector scratch.
    /// </summary>
    /// <param name="configuration"><see cref="Configuration"/> to use internally</param>
    /// <param name="destination">the destination span</param>
    /// <param name="background">the background span</param>
    /// <param name="source">the source color</param>
    /// <param name="amount">
    /// A span with values between 0 and 1 indicating the weight of the second source vector.
    /// At amount = 0, "background" is returned, at amount = 1, "source" is returned.
    /// </param>
    /// <param name="coverage">A span with coverage values between 0 and 1.</param>
    /// <param name="workingBuffer">Reusable temporary vector scratch with capacity for at least 2 rows.</param>
    public void BlendWithCoverage(
        Configuration configuration,
        Span<TPixel> destination,
        ReadOnlySpan<TPixel> background,
        TPixel source,
        ReadOnlySpan<float> amount,
        ReadOnlySpan<float> coverage,
        Span<Vector4> workingBuffer)
    {
        int maxLength = destination.Length;
        Guard.MustBeGreaterThanOrEqualTo(background.Length, maxLength, nameof(background.Length));
        Guard.MustBeGreaterThanOrEqualTo(amount.Length, maxLength, nameof(amount.Length));
        Guard.MustBeGreaterThanOrEqualTo(coverage.Length, maxLength, nameof(coverage.Length));
        Guard.MustBeGreaterThanOrEqualTo(workingBuffer.Length, maxLength * 2, nameof(workingBuffer.Length));

        Span<Vector4> destinationVectors = workingBuffer[..maxLength];
        Span<Vector4> backgroundVectors = workingBuffer.Slice(maxLength, maxLength);

        this.ToBlendVector4(configuration, background[..maxLength], backgroundVectors);

        this.BlendWithCoverageFunction(destinationVectors, backgroundVectors, this.ToBlendVector4(source), amount, coverage);

        this.FromBlendVector4(configuration, destinationVectors, destination);
    }

    /// <summary>
    /// Converts source pixels to the scaled-vector representation consumed by this blender.
    /// </summary>
    /// <typeparam name="TPixelSource">The source pixel format.</typeparam>
    /// <param name="configuration">The configuration.</param>
    /// <param name="source">The source pixels.</param>
    /// <param name="destination">The destination vectors.</param>
    protected virtual void ToBlendVector4<TPixelSource>(
        Configuration configuration,
        ReadOnlySpan<TPixelSource> source,
        Span<Vector4> destination)
        where TPixelSource : unmanaged, IPixel<TPixelSource>
    {
        // The input format's PixelOperations owns conversion rules and bulk optimizations, including format-specific overrides.
        PixelOperations<TPixelSource>.Instance.ToUnassociatedScaledVector4(configuration, source, destination);
    }

    /// <summary>
    /// Converts a source pixel to the vector representation consumed by the blend functions.
    /// </summary>
    /// <param name="source">The source pixel.</param>
    /// <returns>The source vector.</returns>
    protected virtual Vector4 ToBlendVector4(TPixel source)
        => source.ToUnassociatedScaledVector4();

    /// <summary>
    /// Converts blend results from this blender's scaled-vector representation to destination pixels.
    /// </summary>
    /// <param name="configuration">The configuration.</param>
    /// <param name="source">The source vectors.</param>
    /// <param name="destination">The destination pixels.</param>
    protected virtual void FromBlendVector4(
        Configuration configuration,
        Span<Vector4> source,
        Span<TPixel> destination)
    {
        // Destination PixelOperations owns conversion and alpha handling. The pixel format owns rounding and storage layout.
        PixelOperations<TPixel>.Instance.FromUnassociatedScaledVector4Destructive(configuration, source, destination);
    }

    /// <summary>
    /// Blend 2 rows together.
    /// </summary>
    /// <param name="destination">destination span</param>
    /// <param name="background">the background span</param>
    /// <param name="source">the source span</param>
    /// <param name="amount">
    /// A value between 0 and 1 indicating the weight of the second source vector.
    /// At amount = 0, "background" is returned, at amount = 1, "source" is returned.
    /// </param>
    protected abstract void BlendFunction(
        Span<Vector4> destination,
        ReadOnlySpan<Vector4> background,
        ReadOnlySpan<Vector4> source,
        float amount);

    /// <summary>
    /// Blend a row against a constant source color.
    /// </summary>
    /// <param name="destination">destination span</param>
    /// <param name="background">the background span</param>
    /// <param name="source">the source color vector</param>
    /// <param name="amount">
    /// A value between 0 and 1 indicating the weight of the second source vector.
    /// At amount = 0, "background" is returned, at amount = 1, "source" is returned.
    /// </param>
    protected abstract void BlendFunction(
        Span<Vector4> destination,
        ReadOnlySpan<Vector4> background,
        Vector4 source,
        float amount);

    /// <summary>
    /// Blend 2 rows together.
    /// </summary>
    /// <param name="destination">destination span</param>
    /// <param name="background">the background span</param>
    /// <param name="source">the source span</param>
    /// <param name="amount">
    /// A span with values between 0 and 1 indicating the weight of the second source vector.
    /// At amount = 0, "background" is returned, at amount = 1, "source" is returned.
    /// </param>
    protected abstract void BlendFunction(
        Span<Vector4> destination,
        ReadOnlySpan<Vector4> background,
        ReadOnlySpan<Vector4> source,
        ReadOnlySpan<float> amount);

    /// <summary>
    /// Blend a row against a constant source color.
    /// </summary>
    /// <param name="destination">destination span</param>
    /// <param name="background">the background span</param>
    /// <param name="source">the source color vector</param>
    /// <param name="amount">
    /// A span with values between 0 and 1 indicating the weight of the second source vector.
    /// At amount = 0, "background" is returned, at amount = 1, "source" is returned.
    /// </param>
    protected abstract void BlendFunction(
        Span<Vector4> destination,
        ReadOnlySpan<Vector4> background,
        Vector4 source,
        ReadOnlySpan<float> amount);

    /// <summary>
    /// Blend 2 rows together with per-pixel coverage.
    /// </summary>
    /// <param name="destination">destination span</param>
    /// <param name="background">the background span</param>
    /// <param name="source">the source span</param>
    /// <param name="amount">
    /// A value between 0 and 1 indicating the weight of the second source vector.
    /// At amount = 0, "background" is returned, at amount = 1, "source" is returned.
    /// </param>
    /// <param name="coverage">A span with coverage values between 0 and 1.</param>
    protected virtual void BlendWithCoverageFunction(
        Span<Vector4> destination,
        ReadOnlySpan<Vector4> background,
        ReadOnlySpan<Vector4> source,
        float amount,
        ReadOnlySpan<float> coverage)
    {
        this.BlendFunction(destination, background, source, amount);

        for (int i = 0; i < destination.Length; i++)
        {
            destination[i] = PorterDuffFunctions.BlendWithCoverage(background[i], destination[i], ColorNumerics.ClampToFinite(coverage[i], 0, 1F));
        }
    }

    /// <summary>
    /// Blend a row against a constant source color with per-pixel coverage.
    /// </summary>
    /// <param name="destination">destination span</param>
    /// <param name="background">the background span</param>
    /// <param name="source">the source color vector</param>
    /// <param name="amount">
    /// A value between 0 and 1 indicating the weight of the second source vector.
    /// At amount = 0, "background" is returned, at amount = 1, "source" is returned.
    /// </param>
    /// <param name="coverage">A span with coverage values between 0 and 1.</param>
    protected virtual void BlendWithCoverageFunction(
        Span<Vector4> destination,
        ReadOnlySpan<Vector4> background,
        Vector4 source,
        float amount,
        ReadOnlySpan<float> coverage)
    {
        this.BlendFunction(destination, background, source, amount);

        for (int i = 0; i < destination.Length; i++)
        {
            destination[i] = PorterDuffFunctions.BlendWithCoverage(background[i], destination[i], ColorNumerics.ClampToFinite(coverage[i], 0, 1F));
        }
    }

    /// <summary>
    /// Blend 2 rows together with per-pixel coverage.
    /// </summary>
    /// <param name="destination">destination span</param>
    /// <param name="background">the background span</param>
    /// <param name="source">the source span</param>
    /// <param name="amount">
    /// A span with values between 0 and 1 indicating the weight of the second source vector.
    /// At amount = 0, "background" is returned, at amount = 1, "source" is returned.
    /// </param>
    /// <param name="coverage">A span with coverage values between 0 and 1.</param>
    protected virtual void BlendWithCoverageFunction(
        Span<Vector4> destination,
        ReadOnlySpan<Vector4> background,
        ReadOnlySpan<Vector4> source,
        ReadOnlySpan<float> amount,
        ReadOnlySpan<float> coverage)
    {
        this.BlendFunction(destination, background, source, amount);

        for (int i = 0; i < destination.Length; i++)
        {
            destination[i] = PorterDuffFunctions.BlendWithCoverage(background[i], destination[i], ColorNumerics.ClampToFinite(coverage[i], 0, 1F));
        }
    }

    /// <summary>
    /// Blend a row against a constant source color with per-pixel coverage.
    /// </summary>
    /// <param name="destination">destination span</param>
    /// <param name="background">the background span</param>
    /// <param name="source">the source color vector</param>
    /// <param name="amount">
    /// A span with values between 0 and 1 indicating the weight of the second source vector.
    /// At amount = 0, "background" is returned, at amount = 1, "source" is returned.
    /// </param>
    /// <param name="coverage">A span with coverage values between 0 and 1.</param>
    protected virtual void BlendWithCoverageFunction(
        Span<Vector4> destination,
        ReadOnlySpan<Vector4> background,
        Vector4 source,
        ReadOnlySpan<float> amount,
        ReadOnlySpan<float> coverage)
    {
        this.BlendFunction(destination, background, source, amount);

        for (int i = 0; i < destination.Length; i++)
        {
            destination[i] = PorterDuffFunctions.BlendWithCoverage(background[i], destination[i], ColorNumerics.ClampToFinite(coverage[i], 0, 1F));
        }
    }
}

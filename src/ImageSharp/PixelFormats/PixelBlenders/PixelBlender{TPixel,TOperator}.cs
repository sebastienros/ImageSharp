// Copyright (c) Six Labors.
// Licensed under the Six Labors Split License.

using System.Numerics;

namespace SixLabors.ImageSharp.PixelFormats.PixelBlenders;

/// <summary>
/// Blends pixels using the selected color blending and alpha composition operator.
/// </summary>
/// <typeparam name="TPixel">The destination pixel format.</typeparam>
/// <typeparam name="TOperator">The color blending and alpha composition operator.</typeparam>
internal sealed class PixelBlender<TPixel, TOperator> : PixelBlender<TPixel>
    where TPixel : unmanaged, IPixel<TPixel>
    where TOperator : struct, IPixelBlenderOperator
{
    // This type connects pixel conversion to shared vector blending. The base class manages row buffers.
    // The operator owns composition arithmetic and declares its required alpha representation.
    // PixelOperations owns format-specific conversion and bulk optimizations. Pixel formats own rounding and storage layout.

    /// <summary>
    /// Initializes a new instance of the <see cref="PixelBlender{TPixel, TOperator}"/> class.
    /// </summary>
    private PixelBlender()
    {
    }

    /// <summary>
    /// Gets the shared blender instance for this exact pixel and equation combination.
    /// </summary>
    // Keeping the singleton on the already-required closed blender type preserves per-mode lazy initialization
    // without introducing another NativeAOT holder type or eagerly allocating every blender for a pixel format.
    public static PixelBlender<TPixel> Instance { get; } = new PixelBlender<TPixel, TOperator>();

    /// <inheritdoc />
    /// <remarks>
    /// True for the straight-alpha <c>NormalSrcOver</c> equations and pixel formats of 8-bit or 16-bit integer channels
    /// with straight or no alpha. With source alpha 1 and amount 1, <c>Over</c> gives alpha 1 and the colour
    /// <c>s * (1 - dW) + s * dW</c>, and the coverage mix at coverage 1 returns that result, divided by an alpha of 1.
    /// Each step rounds within a few units of 2^-24, so every channel is within 1e-6 of the source channel, below half a
    /// level of a 16-bit channel (7.6e-6), and the stored channel is the source channel. Float formats would store the
    /// error, and associated-alpha formats use other equations, so they are not admitted.
    /// </remarks>
    private protected override bool CopiesOpaqueSourceAtFullCoverage
        => typeof(TOperator) == typeof(DefaultPixelBlenderOperators.NormalSrcOver)
        && (typeof(TPixel) == typeof(Rgba32)
            || typeof(TPixel) == typeof(Bgra32)
            || typeof(TPixel) == typeof(Argb32)
            || typeof(TPixel) == typeof(Abgr32)
            || typeof(TPixel) == typeof(Rgb24)
            || typeof(TPixel) == typeof(Bgr24)
            || typeof(TPixel) == typeof(Rgba64)
            || typeof(TPixel) == typeof(Rgb48));

    /// <inheritdoc />
    public override TPixel Blend(TPixel background, TPixel source, float amount)
    {
        // The operator type uniquely determines the alpha representation. NativeAOT therefore needs only the
        // pixel/operator closure, while the JIT can remove this static choice from the generated hot path.
        Vector4 backgroundVector = TOperator.IsAssociatedAlpha
            ? background.ToAssociatedScaledVector4()
            : background.ToUnassociatedScaledVector4();
        Vector4 sourceVector = TOperator.IsAssociatedAlpha
            ? source.ToAssociatedScaledVector4()
            : source.ToUnassociatedScaledVector4();

        Vector4 result = TOperator.Invoke(backgroundVector, sourceVector, ColorNumerics.ClampToFinite(amount, 0, 1F));

        return TOperator.IsAssociatedAlpha
            ? TPixel.FromAssociatedScaledVector4(result)
            : TPixel.FromUnassociatedScaledVector4(result);
    }

    /// <inheritdoc />
    protected sealed override void ToBlendVector4<TPixelSource>(
        Configuration configuration,
        ReadOnlySpan<TPixelSource> source,
        Span<Vector4> destination)
    {
        // The operator selects the required alpha representation. The input format's PixelOperations
        // owns conversion and its bulk optimizations, including overrides for formats different from TPixel.
        if (TOperator.IsAssociatedAlpha)
        {
            PixelOperations<TPixelSource>.Instance.ToAssociatedScaledVector4(configuration, source, destination);
        }
        else
        {
            PixelOperations<TPixelSource>.Instance.ToUnassociatedScaledVector4(configuration, source, destination);
        }
    }

    /// <inheritdoc />
    protected sealed override Vector4 ToBlendVector4(TPixel source)
        => TOperator.IsAssociatedAlpha
            ? source.ToAssociatedScaledVector4()
            : source.ToUnassociatedScaledVector4();

    /// <inheritdoc />
    protected sealed override void FromBlendVector4(
        Configuration configuration,
        Span<Vector4> source,
        Span<TPixel> destination)
    {
        // Destination PixelOperations owns conversion and alpha handling. The destination pixel format
        // owns rounding and storage layout; the composition operator does not pack pixels.
        if (TOperator.IsAssociatedAlpha)
        {
            PixelOperations<TPixel>.Instance.FromAssociatedScaledVector4Destructive(configuration, source, destination);
        }
        else
        {
            PixelOperations<TPixel>.Instance.FromUnassociatedScaledVector4Destructive(configuration, source, destination);
        }
    }

    /// <inheritdoc />
    protected sealed override void BlendFunction(Span<Vector4> destination, ReadOnlySpan<Vector4> background, ReadOnlySpan<Vector4> source, float amount)
        => PixelBlenders.Blend<TOperator>(destination, background, source, amount);

    /// <inheritdoc />
    protected sealed override void BlendFunction(Span<Vector4> destination, ReadOnlySpan<Vector4> background, Vector4 source, float amount)
        => PixelBlenders.Blend<TOperator>(destination, background, source, amount);

    /// <inheritdoc />
    protected sealed override void BlendFunction(
        Span<Vector4> destination,
        ReadOnlySpan<Vector4> background,
        ReadOnlySpan<Vector4> source,
        ReadOnlySpan<float> amount)
        => PixelBlenders.Blend<TOperator>(destination, background, source, amount);

    /// <inheritdoc />
    protected sealed override void BlendFunction(Span<Vector4> destination, ReadOnlySpan<Vector4> background, Vector4 source, ReadOnlySpan<float> amount)
        => PixelBlenders.Blend<TOperator>(destination, background, source, amount);

    /// <inheritdoc />
    protected sealed override void BlendWithCoverageFunction(
        Span<Vector4> destination,
        ReadOnlySpan<Vector4> background,
        ReadOnlySpan<Vector4> source,
        float amount,
        ReadOnlySpan<float> coverage)
        => PixelBlenders.BlendWithCoverage<TOperator>(destination, background, source, amount, coverage);

    /// <inheritdoc />
    protected sealed override void BlendWithCoverageFunction(
        Span<Vector4> destination,
        ReadOnlySpan<Vector4> background,
        Vector4 source,
        float amount,
        ReadOnlySpan<float> coverage)
        => PixelBlenders.BlendWithCoverage<TOperator>(destination, background, source, amount, coverage);

    /// <inheritdoc />
    protected sealed override void BlendWithCoverageFunction(
        Span<Vector4> destination,
        ReadOnlySpan<Vector4> background,
        ReadOnlySpan<Vector4> source,
        ReadOnlySpan<float> amount,
        ReadOnlySpan<float> coverage)
        => PixelBlenders.BlendWithCoverage<TOperator>(destination, background, source, amount, coverage);

    /// <inheritdoc />
    protected sealed override void BlendWithCoverageFunction(
        Span<Vector4> destination,
        ReadOnlySpan<Vector4> background,
        Vector4 source,
        ReadOnlySpan<float> amount,
        ReadOnlySpan<float> coverage)
        => PixelBlenders.BlendWithCoverage<TOperator>(destination, background, source, amount, coverage);
}

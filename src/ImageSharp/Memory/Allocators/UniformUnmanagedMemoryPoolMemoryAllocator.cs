// Copyright (c) Six Labors.
// Licensed under the Six Labors Split License.

using System.Runtime.CompilerServices;
using SixLabors.ImageSharp.Memory.Internals;

namespace SixLabors.ImageSharp.Memory;

internal sealed class UniformUnmanagedMemoryPoolMemoryAllocator : MemoryAllocator
{
    private const int OneMegabyte = 1 << 20;

    // 4 MB seemed to perform slightly better in benchmarks than 2MB or higher values:
    private const int DefaultContiguousPoolBlockSizeBytes = 4 * OneMegabyte;
    private const int DefaultNonPoolBlockSizeBytes = 32 * OneMegabyte;
    private readonly int sharedArrayPoolThresholdInBytes;
    private readonly int poolBufferSizeInBytes;
    private readonly int poolCapacity;
    private readonly UniformUnmanagedMemoryPool.TrimSettings trimSettings;

    private readonly UniformUnmanagedMemoryPool pool;
    private readonly UnmanagedMemoryAllocator nonPoolAllocator;
    private readonly LargeBufferCache? largeBufferCache;

    public UniformUnmanagedMemoryPoolMemoryAllocator(int? maxPoolSizeMegabytes)
        : this(
            DefaultContiguousPoolBlockSizeBytes,
            maxPoolSizeMegabytes.HasValue ? (long)maxPoolSizeMegabytes.Value * OneMegabyte : GetDefaultMaxPoolSizeBytes(),
            DefaultNonPoolBlockSizeBytes)
    {
    }

    public UniformUnmanagedMemoryPoolMemoryAllocator(
        int poolBufferSizeInBytes,
        long maxPoolSizeInBytes,
        int unmanagedBufferSizeInBytes)
        : this(
            OneMegabyte,
            poolBufferSizeInBytes,
            maxPoolSizeInBytes,
            unmanagedBufferSizeInBytes)
    {
    }

    internal UniformUnmanagedMemoryPoolMemoryAllocator(
        int sharedArrayPoolThresholdInBytes,
        int poolBufferSizeInBytes,
        long maxPoolSizeInBytes,
        int unmanagedBufferSizeInBytes)
        : this(
            sharedArrayPoolThresholdInBytes,
            poolBufferSizeInBytes,
            maxPoolSizeInBytes,
            unmanagedBufferSizeInBytes,
            UniformUnmanagedMemoryPool.TrimSettings.Default)
    {
    }

    internal UniformUnmanagedMemoryPoolMemoryAllocator(
        int sharedArrayPoolThresholdInBytes,
        int poolBufferSizeInBytes,
        long maxPoolSizeInBytes,
        int unmanagedBufferSizeInBytes,
        UniformUnmanagedMemoryPool.TrimSettings trimSettings)
    {
        this.sharedArrayPoolThresholdInBytes = sharedArrayPoolThresholdInBytes;
        this.poolBufferSizeInBytes = poolBufferSizeInBytes;
        this.poolCapacity = (int)(maxPoolSizeInBytes / poolBufferSizeInBytes);
        this.trimSettings = trimSettings;
        this.pool = new UniformUnmanagedMemoryPool(this.poolBufferSizeInBytes, this.poolCapacity, this.trimSettings);
        this.nonPoolAllocator = new UnmanagedMemoryAllocator(unmanagedBufferSizeInBytes);

        // Contiguous buffers above the pool block size come from new unmanaged blocks. A few returned blocks are kept for
        // the next such request: up to a quarter of the pool size (at most 1 GB) in at most 8 blocks. A pool size of 0
        // keeps none, and disabled trimming keeps none, because the cache trims with the pool's settings.
        long largeCacheBytes = Math.Min(maxPoolSizeInBytes / 4, 1L << 30);
        if (largeCacheBytes > poolBufferSizeInBytes && trimSettings.Enabled)
        {
            this.largeBufferCache = new LargeBufferCache(8, largeCacheBytes, trimSettings);
        }
    }

    internal UniformUnmanagedMemoryPoolMemoryAllocator(
        int sharedArrayPoolThresholdInBytes,
        int poolBufferSizeInBytes,
        long maxPoolSizeInBytes,
        int unmanagedBufferSizeInBytes,
        MemoryAllocatorOptions options)
        : this(sharedArrayPoolThresholdInBytes, poolBufferSizeInBytes, maxPoolSizeInBytes, unmanagedBufferSizeInBytes)
        => this.ApplyOptions(options);

    /// <inheritdoc />
    protected internal override int GetBufferCapacityInBytes() => this.poolBufferSizeInBytes;

    /// <inheritdoc />
    protected override AllocationTrackedMemoryManager<T> AllocateCore<T>(
        int length,
        AllocationOptions options = AllocationOptions.None)
    {
        int lengthInBytes = length * Unsafe.SizeOf<T>();
        if (lengthInBytes <= this.sharedArrayPoolThresholdInBytes)
        {
            SharedArrayPoolBuffer<T> buffer = new(length);
            if (options.Has(AllocationOptions.Clean))
            {
                buffer.GetSpan().Clear();
            }

            return buffer;
        }

        if (lengthInBytes <= this.poolBufferSizeInBytes)
        {
            UnmanagedMemoryHandle mem = this.pool.Rent();
            if (mem.IsValid)
            {
                return this.pool.CreateGuardedBuffer<T>(mem, length, options.Has(AllocationOptions.Clean));
            }
        }

        if (this.largeBufferCache is not null)
        {
            // A block from the cache, or a new block that the cache can take back when the buffer is released.
            UnmanagedMemoryHandle block = this.largeBufferCache.Rent(lengthInBytes);
            if (block.IsInvalid)
            {
                block = UnmanagedMemoryHandle.Allocate(lengthInBytes);
            }

            UnmanagedBuffer<T> buffer = new(length, new LargeBufferCache.ReturnToCacheLifetimeGuard(this.largeBufferCache, block));
            if (options.Has(AllocationOptions.Clean))
            {
                buffer.GetSpan().Clear();
            }

            return buffer;
        }

        return UnmanagedMemoryAllocator.AllocateBuffer<T>(length, options);
    }

    /// <inheritdoc />
    internal override MemoryGroup<T> AllocateGroupCore<T>(
        long totalLengthInElements,
        long totalLengthInBytes,
        int bufferAlignment,
        AllocationOptions options = AllocationOptions.None)
    {
        if (totalLengthInBytes <= this.sharedArrayPoolThresholdInBytes)
        {
            SharedArrayPoolBuffer<T> buffer = new((int)totalLengthInElements);
            return MemoryGroup<T>.CreateContiguous(buffer, options.Has(AllocationOptions.Clean));
        }

        if (totalLengthInBytes <= this.poolBufferSizeInBytes)
        {
            // Optimized path renting single array from the pool
            UnmanagedMemoryHandle mem = this.pool.Rent();
            if (mem.IsValid)
            {
                UnmanagedBuffer<T> buffer = this.pool.CreateGuardedBuffer<T>(mem, (int)totalLengthInElements, options.Has(AllocationOptions.Clean));
                return MemoryGroup<T>.CreateContiguous(buffer, options.Has(AllocationOptions.Clean));
            }
        }

        // Attempt to rent the whole group from the pool, allocate a group of unmanaged buffers if the attempt fails:
        if (MemoryGroup<T>.TryAllocate(this.pool, totalLengthInElements, bufferAlignment, options, out MemoryGroup<T>? poolGroup))
        {
            return poolGroup;
        }

        return MemoryGroup<T>.Allocate(this.nonPoolAllocator, totalLengthInElements, bufferAlignment, options);
    }

    public override void ReleaseRetainedResources()
    {
        this.pool.Release();
        this.largeBufferCache?.Release();
    }

    private static long GetDefaultMaxPoolSizeBytes()
    {
        if (Environment.Is64BitProcess)
        {
            // On 64 bit set the pool size to a portion of the total available memory.
            GCMemoryInfo info = GC.GetGCMemoryInfo();
            return info.TotalAvailableMemoryBytes / 8;
        }

        // Stick to a conservative value of 128 Megabytes on 32 bit.
        return 128 * OneMegabyte;
    }
}

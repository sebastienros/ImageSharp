// Copyright (c) Six Labors.
// Licensed under the Six Labors Split License.

using System.Diagnostics;
using System.Runtime.ConstrainedExecution;

namespace SixLabors.ImageSharp.Memory.Internals;

/// <summary>
/// Keeps a few recently returned unmanaged blocks that are larger than a pool block, so that a workload that rents and
/// returns large contiguous buffers again and again reuses mapped memory instead of allocating, faulting in, and freeing
/// a new block each time.
/// </summary>
/// <remarks>
/// A block adds its memory pressure once, when it is created (<see cref="UnmanagedMemoryHandle.Allocate(int)"/>), and
/// removes it when the cache frees it, as the blocks of <see cref="UniformUnmanagedMemoryPool"/> do. The cache frees
/// every block on a generation 2 collection when no block was rented for the trim period or when the memory load is
/// high, and when the allocator releases its retained resources.
/// The cache is a <see cref="CriticalFinalizerObject"/> for the same reason as the pool: an allocator can be dropped
/// without a call to <see cref="MemoryAllocator.ReleaseRetainedResources"/>, and the finalizer must still free the
/// cached blocks.
/// </remarks>
internal sealed class LargeBufferCache : CriticalFinalizerObject
{
    private static readonly Stopwatch Stopwatch = Stopwatch.StartNew();

    private readonly UnmanagedMemoryHandle[] blocks;
    private readonly long maxCachedBytes;
    private readonly UniformUnmanagedMemoryPool.TrimSettings trimSettings;
    private int count;
    private long cachedBytes;
    private long lastRentTimestamp;
    private int finalized;

    /// <summary>
    /// Initializes a new instance of the <see cref="LargeBufferCache"/> class.
    /// </summary>
    /// <param name="maxBlocks">The largest number of blocks to keep.</param>
    /// <param name="maxCachedBytes">The largest number of bytes to keep.</param>
    /// <param name="trimSettings">The trim settings of the pool of the same allocator.</param>
    public LargeBufferCache(int maxBlocks, long maxCachedBytes, UniformUnmanagedMemoryPool.TrimSettings trimSettings)
    {
        this.blocks = new UnmanagedMemoryHandle[maxBlocks];
        this.maxCachedBytes = maxCachedBytes;
        this.trimSettings = trimSettings;
        this.lastRentTimestamp = Stopwatch.ElapsedMilliseconds;
        Gen2GcCallback.Register(s => ((LargeBufferCache)s).Trim(), this);
    }

    /// <summary>
    /// Finalizes an instance of the <see cref="LargeBufferCache"/> class. It frees the cached blocks of an allocator that
    /// was dropped. Blocks that buffers return later are freed by <see cref="Return"/>.
    /// </summary>
    ~LargeBufferCache()
    {
        Volatile.Write(ref this.finalized, 1);
        this.Release();
    }

    /// <summary>
    /// Rents the smallest cached block that holds <paramref name="lengthInBytes"/> bytes and wastes at most half of itself.
    /// </summary>
    /// <param name="lengthInBytes">The requested length in bytes.</param>
    /// <returns>The block, or <see cref="UnmanagedMemoryHandle.NullHandle"/> when no cached block fits.</returns>
    public UnmanagedMemoryHandle Rent(int lengthInBytes)
    {
        lock (this.blocks)
        {
            int best = -1;
            for (int i = 0; i < this.count; i++)
            {
                int size = this.blocks[i].LengthInBytes;
                if (size >= lengthInBytes && size / 2 <= lengthInBytes && (best < 0 || size < this.blocks[best].LengthInBytes))
                {
                    best = i;
                }
            }

            if (best < 0)
            {
                return UnmanagedMemoryHandle.NullHandle;
            }

            // Only a hit counts as use. A miss does not keep blocks that no request fits from being trimmed.
            this.lastRentTimestamp = Stopwatch.ElapsedMilliseconds;

            UnmanagedMemoryHandle block = this.blocks[best];
            this.count--;
            this.blocks[best] = this.blocks[this.count];
            this.blocks[this.count] = default;
            this.cachedBytes -= block.LengthInBytes;
            return block;
        }
    }

    /// <summary>
    /// Keeps a returned block when the cache has room for it.
    /// </summary>
    /// <param name="block">The block. The caller frees it when this method returns <see langword="false"/>.</param>
    /// <returns><see langword="true"/> when the cache keeps the block.</returns>
    public bool Return(UnmanagedMemoryHandle block)
    {
        lock (this.blocks)
        {
            // A finalized cache keeps nothing: the finalizers of the cache and of a buffer guard can run in any order.
            if (this.finalized != 0 || this.count == this.blocks.Length || this.cachedBytes + block.LengthInBytes > this.maxCachedBytes)
            {
                return false;
            }

            this.blocks[this.count++] = block;
            this.cachedBytes += block.LengthInBytes;
            return true;
        }
    }

    /// <summary>
    /// Frees every cached block.
    /// </summary>
    public void Release()
    {
        lock (this.blocks)
        {
            for (int i = 0; i < this.count; i++)
            {
                this.blocks[i].Free();
                this.blocks[i] = default;
            }

            this.count = 0;
            this.cachedBytes = 0;
        }
    }

    /// <summary>
    /// Frees the cached blocks after a generation 2 collection when no block was rented for the trim period of the pool,
    /// or when the memory load is high.
    /// </summary>
    /// <returns><see langword="true"/> to stay registered for the next generation 2 collection.</returns>
    private bool Trim()
    {
        if (this.count == 0)
        {
            return true;
        }

        GCMemoryInfo memoryInfo = GC.GetGCMemoryInfo();
        bool highPressure = memoryInfo.MemoryLoadBytes >= memoryInfo.HighMemoryLoadThresholdBytes * this.trimSettings.HighPressureThresholdRate;
        if (highPressure || Stopwatch.ElapsedMilliseconds - Volatile.Read(ref this.lastRentTimestamp) > this.trimSettings.TrimPeriodMilliseconds)
        {
            this.Release();
        }

        return true;
    }

    /// <summary>
    /// Returns the block of a buffer to the cache when the buffer is released, or frees it when the cache is full.
    /// </summary>
    internal sealed class ReturnToCacheLifetimeGuard : UnmanagedBufferLifetimeGuard
    {
        private readonly LargeBufferCache cache;

        /// <summary>
        /// Initializes a new instance of the <see cref="ReturnToCacheLifetimeGuard"/> class.
        /// </summary>
        /// <param name="cache">The cache that takes the block back.</param>
        /// <param name="handle">The block.</param>
        public ReturnToCacheLifetimeGuard(LargeBufferCache cache, UnmanagedMemoryHandle handle)
            : base(handle) => this.cache = cache;

        /// <inheritdoc />
        protected override void Release()
        {
            if (!this.cache.Return(this.Handle))
            {
                this.Handle.Free();
            }
        }
    }
}

import numba
import os

# Get the path to the Numba cache directory
numba_cache_dir = numba.config.CACHE_DIR

# Check if the directory exists and print its location
if os.path.exists(numba_cache_dir):
    print(f"Numba cache directory is located at: {numba_cache_dir}")
else:
    print(f"Numba cache directory '{numba_cache_dir}' does not exist.")

# Clear Numba cache
numba.core.caching._cache_manager.clear_all_caches()
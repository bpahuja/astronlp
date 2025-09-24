# Correct way to read your memory-mapped file
import numpy as np
import os

def read_memmap_file(file_path, shape, dtype='float32'):
    """
    Read a memory-mapped file created with np.memmap
    
    Parameters:
    - file_path: path to the memmap file
    - shape: tuple representing the array shape
    - dtype: data type (default 'float32' as used in your code)
    """
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Get file size to verify it matches expected shape
    file_size = os.path.getsize(file_path)
    expected_size = np.prod(shape) * np.dtype(dtype).itemsize
    
    print(f"File size: {file_size:,} bytes")
    print(f"Expected size: {expected_size:,} bytes")
    
    if file_size != expected_size:
        print(f"WARNING: File size mismatch!")
        print(f"File might be incomplete or have different dimensions")
        
        # Try to infer the actual shape based on file size
        total_elements = file_size // np.dtype(dtype).itemsize
        print(f"Total elements in file: {total_elements:,}")
        
        if len(shape) == 2:
            # Assuming the second dimension is correct (target_dims)
            actual_rows = total_elements // shape[1]
            actual_shape = (actual_rows, shape[1])
            print(f"Inferred shape: {actual_shape}")
            shape = actual_shape
    
    # Create memory map to read the file
    try:
        data = np.memmap(file_path, dtype=dtype, mode='r', shape=shape)
        print(f"✓ Successfully opened memory-mapped file")
        print(f"  Shape: {data.shape}")
        print(f"  Dtype: {data.dtype}")
        return data
        
    except Exception as e:
        print(f"✗ Failed to open memory-mapped file: {e}")
        return None

def convert_memmap_to_npy(memmap_file, output_file, shape, dtype='float32'):
    """
    Convert a memory-mapped file to a standard .npy file
    """
    print(f"Converting {memmap_file} to {output_file}")
    
    # Read the memmap file
    data = read_memmap_file(memmap_file, shape, dtype)
    
    if data is not None:
        # Save as standard numpy file
        np.save(output_file, data)
        print(f"✓ Converted and saved to {output_file}")
        
        # Verify the conversion worked
        try:
            loaded_data = np.load(output_file)
            print(f"✓ Verification successful - Shape: {loaded_data.shape}")
            return True
        except Exception as e:
            print(f"✗ Verification failed: {e}")
            return False
    
    return False

# USAGE EXAMPLES:

# Example 1: Read your specific file
# You need to know the shape from your code - it should be (total_samples, target_dims)
file_path = '/vol/bitbucket/bp824/astro/data/paragraph_embeddings_specter2_v4/paragraph_embeddings_reduced.npy'

# You'll need to replace these with your actual values:
# - total_samples: the number of samples you processed
# - target_dims: the PCA target dimensions you used
total_samples = 4089648  # REPLACE WITH YOUR ACTUAL VALUE
target_dims = 100       # REPLACE WITH YOUR ACTUAL VALUE

shape = (total_samples, target_dims)
dtype = 'float32'

print("Attempting to read memory-mapped file...")
data = read_memmap_file(file_path, shape, dtype)

if data is not None:
    print(f"\nFirst few values:")
    print(data[:5, :5])  # Print first 5x5 subarray
    
    print(f"\nStatistics:")
    print(f"Min: {np.min(data)}")
    print(f"Max: {np.max(data)}")
    print(f"Mean: {np.mean(data):.6f}")
    
    # Optional: Convert to standard .npy format for easier future loading
    convert_to_standard = input("\nConvert to standard .npy format? (y/n): ").lower().strip()
    if convert_to_standard == 'y':
        output_path = file_path.replace('.npy', '_converted.npy')
        convert_memmap_to_npy(file_path, output_path, shape, dtype)

# Example 2: If you don't know the exact shape, try to infer it
def auto_detect_shape(file_path, target_dims, dtype='float32'):
    """
    Try to automatically detect the shape based on file size
    """
    file_size = os.path.getsize(file_path)
    element_size = np.dtype(dtype).itemsize
    total_elements = file_size // element_size
    
    # Assume the last dimension is target_dims
    if total_elements % target_dims == 0:
        rows = total_elements // target_dims
        return (rows, target_dims)
    else:
        print(f"WARNING: File size doesn't divide evenly by target_dims")
        return None

# Auto-detection example:
# print(f"\n" + "="*50)
# print("AUTO-DETECTION ATTEMPT:")
# target_dims = 50  # REPLACE WITH YOUR ACTUAL TARGET DIMENSIONS
# inferred_shape = auto_detect_shape(file_path, target_dims)

# if inferred_shape:
#     print(f"Inferred shape: {inferred_shape}")
#     data_auto = read_memmap_file(file_path, inferred_shape, dtype)
#     if data_auto is not None:
#         print("✓ Auto-detection successful!")
# else:
#     print("✗ Could not infer shape automatically")
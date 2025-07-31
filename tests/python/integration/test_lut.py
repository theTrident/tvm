# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import tvm
from tvm import te
import numpy as np
import time
import tvm.testing


def generate_lut_data(lut_size, dtype="float32", seed=42):
    """Generate LUT table data for testing.
    
    Args:
        lut_size (int): Size of the LUT table
        dtype (str): Data type for the LUT values
        seed (int): Random seed for reproducibility
    
    Returns:
        numpy.ndarray: Generated LUT table
    """
    np.random.seed(seed)
    if "float" in dtype:
        return np.random.uniform(-1.0, 1.0, size=lut_size).astype(dtype)
    elif "int" in dtype:
        return np.random.randint(-100, 100, size=lut_size, dtype=dtype)
    else:
        return np.random.uniform(-1.0, 1.0, size=lut_size).astype(dtype)


def generate_indices(shape, max_index, dtype="int32", seed=42):
    """Generate input indices for LUT lookup.
    
    Args:
        shape (tuple): Shape of the index tensor
        max_index (int): Maximum valid index value
        dtype (str): Data type for indices
        seed (int): Random seed for reproducibility
    
    Returns:
        numpy.ndarray: Generated index tensor
    """
    np.random.seed(seed)
    return np.random.randint(0, max_index, size=shape, dtype=dtype)


def lut_nearest_neighbor_numpy(lut_table, indices, boundary_mode="clamp"):
    """Reference numpy implementation of nearest neighbor LUT lookup.
    
    Args:
        lut_table (numpy.ndarray): LUT table data
        indices (numpy.ndarray): Input indices
        boundary_mode (str): How to handle out-of-bounds indices ("clamp" or "wrap")
    
    Returns:
        numpy.ndarray: Lookup results
    """
    lut_size = len(lut_table)
    
    if boundary_mode == "clamp":
        # Clamp indices to valid range [0, lut_size-1]
        clamped_indices = np.clip(indices, 0, lut_size - 1)
        return lut_table[clamped_indices]
    elif boundary_mode == "wrap":
        # Wrap indices using modulo operation
        wrapped_indices = indices % lut_size
        return lut_table[wrapped_indices]
    else:
        raise ValueError(f"Unsupported boundary mode: {boundary_mode}")


def lut_linear_interp_numpy(lut_table, indices, boundary_mode="clamp"):
    """Reference numpy implementation of linear interpolation LUT lookup.
    
    Args:
        lut_table (numpy.ndarray): LUT table data
        indices (numpy.ndarray): Input indices (can be fractional)
        boundary_mode (str): How to handle out-of-bounds indices
    
    Returns:
        numpy.ndarray: Interpolated lookup results
    """
    lut_size = len(lut_table)
    
    # Handle boundary conditions
    if boundary_mode == "clamp":
        clamped_indices = np.clip(indices, 0, lut_size - 1)
    elif boundary_mode == "wrap":
        clamped_indices = indices % lut_size
    else:
        raise ValueError(f"Unsupported boundary mode: {boundary_mode}")
    
    # Get integer and fractional parts
    idx_floor = np.floor(clamped_indices).astype(np.int32)
    idx_frac = clamped_indices - idx_floor
    
    # Handle edge case where index equals lut_size-1
    idx_floor = np.clip(idx_floor, 0, lut_size - 1)
    idx_ceil = np.clip(idx_floor + 1, 0, lut_size - 1)
    
    # Linear interpolation
    val_floor = lut_table[idx_floor]
    val_ceil = lut_table[idx_ceil]
    
    return val_floor * (1.0 - idx_frac) + val_ceil * idx_frac


@tvm.testing.requires_gpu
def test_lut_nearest_neighbor():
    """Test LUT nearest neighbor lookup operation."""
    
    def run_test(dtype="float32", lut_size=256, input_shape=(1024,)):
        # Create LUT table and input indices
        LUT = te.placeholder((lut_size,), name="LUT", dtype=dtype)
        INDICES = te.placeholder(input_shape, name="INDICES", dtype="int32")
        
        # Define LUT lookup computation with clamping
        def lut_lookup(*indices_coords):
            idx = INDICES(*indices_coords)
            # Clamp indices to valid range [0, lut_size-1]
            clamped_idx = te.max(te.min(idx, lut_size - 1), 0)
            return LUT[clamped_idx]
        
        OUTPUT = te.compute(input_shape, lut_lookup, name="OUTPUT")
        
        # Create schedule
        s = te.create_schedule(OUTPUT.op)
        
        # Simple scheduling - can be optimized further
        if len(input_shape) == 1:
            num_thread = 32
            bx, tx = s[OUTPUT].split(OUTPUT.op.axis[0], factor=num_thread)
            s[OUTPUT].bind(bx, te.thread_axis("blockIdx.x"))
            s[OUTPUT].bind(tx, te.thread_axis("threadIdx.x"))
        
        def check_device(device):
            dev = tvm.device(device, 0)
            if not tvm.testing.device_enabled(device):
                print(f"skip because {device} is not enabled..")
                return
            
            with tvm.target.Target(device):
                f = tvm.build(s, [LUT, INDICES, OUTPUT], name="lut_nearest")
            
            # Generate test data
            lut_data = generate_lut_data(lut_size, dtype)
            indices_data = generate_indices(input_shape, lut_size * 2)  # Some out-of-bounds
            
            # Create TVM arrays
            lut_tvm = tvm.nd.array(lut_data, dev)
            indices_tvm = tvm.nd.array(indices_data, dev)
            output_tvm = tvm.nd.array(np.zeros(input_shape, dtype=dtype), dev)
            
            # Run computation
            ftimer = f.time_evaluator(f.entry_name, dev, number=1)
            tcost = ftimer(lut_tvm, indices_tvm, output_tvm).mean
            print(f"{device}: exec={tcost:.6f} sec/op")
            
            # Validate result
            expected = lut_nearest_neighbor_numpy(lut_data, indices_data, "clamp")
            tvm.testing.assert_allclose(output_tvm.asnumpy(), expected, rtol=1e-5)
        
        # Test on different devices
        check_device("cuda")
        check_device("opencl")
        check_device("vulkan") 
        check_device("metal")
    
    # Test different configurations
    run_test("float32", 256, (1024,))
    run_test("float32", 512, (32, 32))
    run_test("int32", 128, (1024,))


@tvm.testing.requires_gpu
def test_lut_linear_interpolation():
    """Test LUT linear interpolation lookup operation."""
    
    def run_test(dtype="float32", lut_size=256, input_shape=(1024,)):
        # Create LUT table and input indices (float for interpolation)
        LUT = te.placeholder((lut_size,), name="LUT", dtype=dtype)
        INDICES = te.placeholder(input_shape, name="INDICES", dtype="float32")
        
        # Define linear interpolation LUT lookup
        def lut_interp_lookup(*indices_coords):
            idx = INDICES(*indices_coords)
            # Clamp fractional indices to valid range [0, lut_size-1]
            clamped_idx = te.max(te.min(idx, lut_size - 1), 0.0)
            
            # Get integer and fractional parts
            idx_floor = te.floor(clamped_idx).astype("int32")
            idx_frac = clamped_idx - te.floor(clamped_idx)
            
            # Ensure we don't go out of bounds
            idx_floor = te.max(te.min(idx_floor, lut_size - 1), 0)
            idx_ceil = te.max(te.min(idx_floor + 1, lut_size - 1), 0)
            
            # Linear interpolation
            val_floor = LUT[idx_floor]
            val_ceil = LUT[idx_ceil]
            
            return val_floor * (1.0 - idx_frac) + val_ceil * idx_frac
        
        OUTPUT = te.compute(input_shape, lut_interp_lookup, name="OUTPUT")
        
        # Create schedule
        s = te.create_schedule(OUTPUT.op)
        
        # Simple scheduling
        if len(input_shape) == 1:
            num_thread = 32
            bx, tx = s[OUTPUT].split(OUTPUT.op.axis[0], factor=num_thread)
            s[OUTPUT].bind(bx, te.thread_axis("blockIdx.x"))
            s[OUTPUT].bind(tx, te.thread_axis("threadIdx.x"))
        
        def check_device(device):
            dev = tvm.device(device, 0)
            if not tvm.testing.device_enabled(device):
                print(f"skip because {device} is not enabled..")
                return
            
            with tvm.target.Target(device):
                f = tvm.build(s, [LUT, INDICES, OUTPUT], name="lut_interp")
            
            # Generate test data
            lut_data = generate_lut_data(lut_size, dtype)
            # Generate fractional indices for interpolation testing
            np.random.seed(42)
            indices_data = np.random.uniform(0, lut_size - 1, size=input_shape).astype("float32")
            
            # Create TVM arrays
            lut_tvm = tvm.nd.array(lut_data, dev)
            indices_tvm = tvm.nd.array(indices_data, dev)
            output_tvm = tvm.nd.array(np.zeros(input_shape, dtype=dtype), dev)
            
            # Run computation
            ftimer = f.time_evaluator(f.entry_name, dev, number=1)
            tcost = ftimer(lut_tvm, indices_tvm, output_tvm).mean
            print(f"{device}: exec={tcost:.6f} sec/op")
            
            # Validate result
            expected = lut_linear_interp_numpy(lut_data, indices_data, "clamp")
            tvm.testing.assert_allclose(output_tvm.asnumpy(), expected, rtol=1e-4)
        
        # Test on different devices
        check_device("cuda")
        check_device("opencl")
        check_device("vulkan")
        check_device("metal")
    
    # Test different configurations
    run_test("float32", 256, (1024,))
    run_test("float32", 512, (32, 32))


@tvm.testing.requires_gpu  
def test_lut_boundary_modes():
    """Test different boundary handling modes for LUT lookup."""
    
    def run_test_boundary(boundary_mode, dtype="float32", lut_size=128):
        input_shape = (512,)
        
        # Create LUT table and input indices
        LUT = te.placeholder((lut_size,), name="LUT", dtype=dtype)
        INDICES = te.placeholder(input_shape, name="INDICES", dtype="int32")
        
        # Define boundary-aware LUT lookup
        def lut_boundary_lookup(*indices_coords):
            idx = INDICES(*indices_coords)
            
            if boundary_mode == "clamp":
                # Clamp to valid range
                safe_idx = te.max(te.min(idx, lut_size - 1), 0)
            elif boundary_mode == "wrap":
                # Wrap using modulo
                safe_idx = idx % lut_size
            else:
                # Default to clamp
                safe_idx = te.max(te.min(idx, lut_size - 1), 0)
            
            return LUT[safe_idx]
        
        OUTPUT = te.compute(input_shape, lut_boundary_lookup, name="OUTPUT")
        
        # Create schedule
        s = te.create_schedule(OUTPUT.op)
        num_thread = 32
        bx, tx = s[OUTPUT].split(OUTPUT.op.axis[0], factor=num_thread)
        s[OUTPUT].bind(bx, te.thread_axis("blockIdx.x"))
        s[OUTPUT].bind(tx, te.thread_axis("threadIdx.x"))
        
        def check_device(device):
            dev = tvm.device(device, 0)
            if not tvm.testing.device_enabled(device):
                print(f"skip because {device} is not enabled..")
                return
            
            with tvm.target.Target(device):
                f = tvm.build(s, [LUT, INDICES, OUTPUT], name=f"lut_{boundary_mode}")
            
            # Generate test data with out-of-bounds indices
            lut_data = generate_lut_data(lut_size, dtype)
            np.random.seed(42)
            indices_data = np.random.randint(-50, lut_size + 50, size=input_shape, dtype=np.int32)
            
            # Create TVM arrays
            lut_tvm = tvm.nd.array(lut_data, dev)
            indices_tvm = tvm.nd.array(indices_data, dev)
            output_tvm = tvm.nd.array(np.zeros(input_shape, dtype=dtype), dev)
            
            # Run computation
            ftimer = f.time_evaluator(f.entry_name, dev, number=1)
            tcost = ftimer(lut_tvm, indices_tvm, output_tvm).mean
            print(f"{device} ({boundary_mode}): exec={tcost:.6f} sec/op")
            
            # Validate result
            expected = lut_nearest_neighbor_numpy(lut_data, indices_data, boundary_mode)
            tvm.testing.assert_allclose(output_tvm.asnumpy(), expected, rtol=1e-5)
        
        # Test on different devices
        check_device("cuda")
        check_device("opencl")
    
    # Test different boundary modes
    run_test_boundary("clamp")
    run_test_boundary("wrap")


@tvm.testing.requires_gpu
def test_lut_batch_processing():
    """Test LUT lookup with batch processing (multi-dimensional inputs)."""
    
    def run_batch_test(dtype="float32"):
        batch_size = 16
        seq_length = 64
        lut_size = 256
        
        # Batch input shape
        input_shape = (batch_size, seq_length)
        
        # Create LUT table and batch indices
        LUT = te.placeholder((lut_size,), name="LUT", dtype=dtype)
        INDICES = te.placeholder(input_shape, name="INDICES", dtype="int32")
        
        # Define batched LUT lookup
        def batch_lut_lookup(batch_idx, seq_idx):
            idx = INDICES[batch_idx, seq_idx]
            clamped_idx = te.max(te.min(idx, lut_size - 1), 0)
            return LUT[clamped_idx]
        
        OUTPUT = te.compute(input_shape, batch_lut_lookup, name="OUTPUT")
        
        # Create schedule with 2D tiling
        s = te.create_schedule(OUTPUT.op)
        
        # 2D thread blocking
        num_thread_x = 8
        num_thread_y = 8
        
        bx, tx = s[OUTPUT].split(OUTPUT.op.axis[0], factor=num_thread_x)
        by, ty = s[OUTPUT].split(OUTPUT.op.axis[1], factor=num_thread_y)
        
        s[OUTPUT].reorder(bx, by, tx, ty)
        s[OUTPUT].bind(bx, te.thread_axis("blockIdx.x"))
        s[OUTPUT].bind(by, te.thread_axis("blockIdx.y"))
        s[OUTPUT].bind(tx, te.thread_axis("threadIdx.x"))
        s[OUTPUT].bind(ty, te.thread_axis("threadIdx.y"))
        
        def check_device(device):
            dev = tvm.device(device, 0)
            if not tvm.testing.device_enabled(device):
                print(f"skip because {device} is not enabled..")
                return
            
            with tvm.target.Target(device):
                f = tvm.build(s, [LUT, INDICES, OUTPUT], name="lut_batch")
            
            # Generate test data
            lut_data = generate_lut_data(lut_size, dtype)
            indices_data = generate_indices(input_shape, lut_size)
            
            # Create TVM arrays
            lut_tvm = tvm.nd.array(lut_data, dev)
            indices_tvm = tvm.nd.array(indices_data, dev)
            output_tvm = tvm.nd.array(np.zeros(input_shape, dtype=dtype), dev)
            
            # Run computation
            ftimer = f.time_evaluator(f.entry_name, dev, number=1)
            tcost = ftimer(lut_tvm, indices_tvm, output_tvm).mean
            print(f"{device} (batch): exec={tcost:.6f} sec/op")
            
            # Validate result
            expected = lut_nearest_neighbor_numpy(lut_data, indices_data, "clamp")
            tvm.testing.assert_allclose(output_tvm.asnumpy(), expected, rtol=1e-5)
        
        # Test on different devices
        check_device("cuda")
        check_device("opencl")
        check_device("vulkan")
        check_device("metal")
    
    run_batch_test("float32")
    run_batch_test("int32")


def test_lut_parameter_validation():
    """Test parameter validation and error handling for LUT operations."""
    
    print("Testing LUT parameter validation...")
    
    # Test basic parameter validation
    try:
        # Valid LUT size
        lut_data = generate_lut_data(256, "float32")
        assert len(lut_data) == 256
        print("✓ Valid LUT size generation")
        
        # Valid indices
        indices_data = generate_indices((100,), 256, "int32")
        assert indices_data.shape == (100,)
        assert indices_data.dtype == np.int32
        print("✓ Valid indices generation")
        
        # Test numpy reference implementations
        result_nn = lut_nearest_neighbor_numpy(lut_data, indices_data[:10], "clamp")
        assert result_nn.shape == (10,)
        print("✓ Nearest neighbor reference implementation")
        
        # Test linear interpolation reference
        float_indices = np.random.uniform(0, 255, size=10).astype("float32")
        result_interp = lut_linear_interp_numpy(lut_data, float_indices, "clamp")
        assert result_interp.shape == (10,)
        print("✓ Linear interpolation reference implementation")
        
        # Test boundary modes
        out_of_bounds_indices = np.array([-10, 300], dtype=np.int32)
        result_clamp = lut_nearest_neighbor_numpy(lut_data, out_of_bounds_indices, "clamp")
        result_wrap = lut_nearest_neighbor_numpy(lut_data, out_of_bounds_indices, "wrap")
        
        # Clamp should give first and last elements
        assert result_clamp[0] == lut_data[0]
        assert result_clamp[1] == lut_data[-1]
        print("✓ Clamp boundary mode")
        
        # Wrap should give wrapped indices
        wrapped_indices = out_of_bounds_indices % 256
        expected_wrap = lut_data[wrapped_indices]
        np.testing.assert_array_equal(result_wrap, expected_wrap)
        print("✓ Wrap boundary mode")
        
        print("All parameter validation tests passed!")
        
    except Exception as e:
        print(f"✗ Parameter validation failed: {e}")
        raise


if __name__ == "__main__":
    """Main function to run all LUT operator tests."""
    
    print("=" * 60)
    print("TVM LUT (Look-Up Table) Operator Tests")
    print("=" * 60)
    
    try:
        # Run parameter validation first
        test_lut_parameter_validation()
        print()
        
        # Run GPU-based tests
        print("Running LUT nearest neighbor tests...")
        test_lut_nearest_neighbor()
        print()
        
        print("Running LUT linear interpolation tests...")
        test_lut_linear_interpolation()
        print()
        
        print("Running LUT boundary mode tests...")
        test_lut_boundary_modes()
        print()
        
        print("Running LUT batch processing tests...")
        test_lut_batch_processing()
        print()
        
        print("=" * 60)
        print("All LUT operator tests completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
import numpy as np
import cupy as cp
from osaliency_cupy import GaborFilter, Orientation_Saliency, cpnormalize_img 

def test_gabor_filter():
    print("=== Test 1: GaborFilter single image ===")
    g = GaborFilter(frequency=0.2, theta=np.pi/4, ksize=5)
    img = cp.random.rand(64, 64).astype(cp.float16)
    resp = g(img)
    assert resp.shape == (64, 64)
    assert not cp.any(cp.isnan(resp))
    assert resp.max() > 0
    print("✅ GaborFilter works.")

def test_cpnormalize_img():
    print("=== Test 2: cpnormalize_img on Gabor response ===")
    g = GaborFilter(frequency=0.2, theta=0.0, ksize=5)
    img = cp.ones((32, 32), dtype=cp.float16)  # constant image
    resp = g(img)
    normed = cpnormalize_img(resp, M=1.0)
    assert normed.shape == resp.shape
    assert not cp.any(cp.isnan(normed))
    print("✅ cpnormalize_img works.")

def test_orientation_saliency_full_pipeline():
    print("=== Test 3: Full Orientation_Saliency pipeline ===")
    # Mock args
    class Args:
        center = [2, 3]
        surrounding = [1, 2]
        total_height = 6
        pyramid_height = 3

    args = Args()
    model = Orientation_Saliency(args)

    # Create dummy image pyramid: list of (H, W) CuPy arrays
    Is = []
    base_h, base_w = 480, 640
    for i in range(args.total_height):
        h = base_h // (2 ** (i // 2))  # simulate coarse-to-fine
        w = base_w // (2 ** (i // 2))
        Is.append(cp.random.rand(h, w, 1).astype(cp.float16))

    # Run pipeline
    model.build_pyramid(Is)
    model.Orientation_maps()
    final_map = model.synthesis_O_map()

    # Validate
    assert final_map.ndim == 2
    assert final_map.shape[0] > 1 and final_map.shape[1] > 1
    assert not cp.any(cp.isnan(final_map))
    assert final_map.max() >= 0
    print(f"✅ Full pipeline works. Output shape: {final_map.shape}")

if __name__ == "__main__":
    test_gabor_filter()
    test_cpnormalize_img()
    test_orientation_saliency_full_pipeline()
    print("\n🎉 All tests passed!")
#!/usr/bin/env python3
"""
Test script to generate a single frame with rain augmentation for visualization
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import sys

# Add LISA to path
sys.path.append('./LISA')
from pylisa.lisa import Lisa

def load_sample_point_cloud():
    """Load a sample point cloud from custom_av/points"""
    points_dir = Path("./data/custom_av/points")
    npy_files = sorted(list(points_dir.glob("*.npy")))
    
    if not npy_files:
        print("No .npy files found in ./data/custom_av/points")
        return None
        
    # Load first file
    sample_file = npy_files[0]
    print(f"Loading: {sample_file}")
    points = np.load(sample_file)
    print(f"Original points shape: {points.shape}")
    return points

def test_rain_augmentation():
    """Test rain augmentation with falling particles"""
    
    # Load sample data
    original_points = load_sample_point_cloud()
    if original_points is None:
        return
    
    # Initialize LISA with rain model
    lisa = Lisa(
        lam=905.0,
        rmax=64.0,  # Match the rmax setting
        rmin=1.5,
        bdiv=3e-3,
        dst=0.05,
        dR=0.09,
        atm_model='rain',
        mode='strongest'
    )
    lisa.atm_model = 'rain'  # Store for enhanced effects
    
    rain_rate = 20.0  # mm/hr
    
    print(f"Applying rain augmentation with rate: {rain_rate} mm/hr")
    
    # Test both standard and enhanced augmentation
    print("\n=== Standard LISA Rain ===")
    standard_result = lisa.augment(original_points, rain_rate)
    print(f"Standard result shape: {standard_result.shape}")
    
    print("\n=== Enhanced LISA with Falling Particles ===")
    enhanced_result = lisa.augment_with_falling_effects(original_points, rain_rate)
    print(f"Enhanced result shape: {enhanced_result.shape}")
    
    # Count different point types in enhanced result
    if enhanced_result.shape[1] >= 5:
        unique_labels, counts = np.unique(enhanced_result[:, 4], return_counts=True)
        print("Point type distribution:")
        for label, count in zip(unique_labels, counts):
            label_name = {0: "Original", 1: "LISA Scattered", 2: "LISA Non-scattered", 
                         3: "Rain Particles", 4: "Snow Particles"}.get(int(label), f"Unknown({int(label)})")
            print(f"  {label_name}: {count} points")
    
    return original_points, standard_result, enhanced_result

def visualize_results(original, standard, enhanced):
    """Create visualization comparing original, standard, and enhanced results"""
    
    fig = plt.figure(figsize=(18, 6))
    
    # Limit points for visualization (random sample)
    def sample_points(points, max_points=5000):
        if len(points) > max_points:
            idx = np.random.choice(len(points), max_points, replace=False)
            return points[idx]
        return points
    
    # Original points
    ax1 = fig.add_subplot(131, projection='3d')
    orig_sample = sample_points(original)
    ax1.scatter(orig_sample[:, 0], orig_sample[:, 1], orig_sample[:, 2], 
                c=orig_sample[:, 3], cmap='viridis', s=1, alpha=0.6)
    ax1.set_title(f'Original\n({len(original)} points)')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_zlim(-4, 4)
    
    # Standard LISA
    ax2 = fig.add_subplot(132, projection='3d')
    std_sample = sample_points(standard[:, :4])  # First 4 columns
    ax2.scatter(std_sample[:, 0], std_sample[:, 1], std_sample[:, 2], 
                c=std_sample[:, 3], cmap='viridis', s=1, alpha=0.6)
    ax2.set_title(f'Standard LISA Rain\n({len(standard)} points)')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    ax2.set_zlim(-4, 4)
    
    # Enhanced LISA with falling particles
    ax3 = fig.add_subplot(133, projection='3d')
    enh_sample = sample_points(enhanced)
    
    # Color by point type if labels available
    if enhanced.shape[1] >= 5:
        # Different colors for different point types
        colors = []
        for point in enh_sample:
            label = int(point[4]) if len(point) > 4 else 0
            if label == 0:   # Original
                colors.append('blue')
            elif label == 1 or label == 2:  # LISA effects
                colors.append('green') 
            elif label == 3:  # Rain particles
                colors.append('red')
            else:
                colors.append('gray')
        
        ax3.scatter(enh_sample[:, 0], enh_sample[:, 1], enh_sample[:, 2], 
                    c=colors, s=1, alpha=0.6)
    else:
        ax3.scatter(enh_sample[:, 0], enh_sample[:, 1], enh_sample[:, 2], 
                    c=enh_sample[:, 3], cmap='viridis', s=1, alpha=0.6)
    
    ax3.set_title(f'Enhanced LISA + Falling Rain\n({len(enhanced)} points)')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_zlabel('Z (m)')  
    ax3.set_zlim(-4, 4)
    
    plt.tight_layout()
    
    # Save visualization
    output_path = "./rain_augmentation_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    # Also save data for further analysis
    np.save("./original_sample.npy", original)
    np.save("./standard_rain_sample.npy", standard)
    np.save("./enhanced_rain_sample.npy", enhanced)
    print("Data files saved: original_sample.npy, standard_rain_sample.npy, enhanced_rain_sample.npy")
    
    plt.show()

def main():
    print("Testing Rain Augmentation with Rain Rate = 20.0 mm/hr")
    print("=" * 50)
    
    # Run the test
    results = test_rain_augmentation()
    if results is None:
        return
        
    original, standard, enhanced = results
    
    # Create visualization
    visualize_results(original, standard, enhanced)
    
    print("\nTest completed! Check the saved files and visualization.")

if __name__ == "__main__":
    main()
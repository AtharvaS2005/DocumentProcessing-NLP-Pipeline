"""
Noise profiling and synthesis module.
Analyzes noise patterns from training images and generates synthetic noisy images.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import json
from tqdm import tqdm
from scipy import ndimage
from skimage.restoration import estimate_sigma


class NoiseProfiler:
    """Analyzes and profiles noise characteristics from noisy images"""
    
    def __init__(self):
        self.noise_profiles = []
        self.global_stats = {}
    
    def extract_noise_from_image(self, image_path: str) -> Dict:
        """
        Extract noise characteristics from a single noisy image.
        Uses various techniques to estimate noise properties.
        """
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_float = img.astype(np.float32) / 255.0
        img_gray_float = img_gray.astype(np.float32) / 255.0
        
        # Estimate noise standard deviation using robust median estimator
        sigma = estimate_sigma(img_gray_float, average_sigmas=True)
        
        # Estimate noise using high-pass filtering
        kernel_size = 3
        img_blurred = cv2.GaussianBlur(img_gray_float, (kernel_size, kernel_size), 0)
        noise_estimate = img_gray_float - img_blurred
        
        # Calculate noise statistics
        noise_mean = np.mean(noise_estimate)
        noise_std = np.std(noise_estimate)
        noise_min = np.min(noise_estimate)
        noise_max = np.max(noise_estimate)
        
        # Frequency domain analysis
        f_transform = np.fft.fft2(img_gray_float)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        # Calculate power spectral density statistics
        psd_mean = np.mean(magnitude_spectrum)
        psd_std = np.std(magnitude_spectrum)
        
        # Texture analysis using Laplacian variance
        laplacian_var = cv2.Laplacian(img_gray, cv2.CV_64F).var()
        
        # Color channel noise analysis
        channel_stds = []
        for i in range(3):
            channel = img_float[:, :, i]
            channel_blurred = cv2.GaussianBlur(channel, (kernel_size, kernel_size), 0)
            channel_noise = channel - channel_blurred
            channel_stds.append(np.std(channel_noise))
        
        # Edge strength analysis
        edges = cv2.Canny(img_gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        profile = {
            'image_path': image_path,
            'sigma': float(sigma),
            'noise_mean': float(noise_mean),
            'noise_std': float(noise_std),
            'noise_min': float(noise_min),
            'noise_max': float(noise_max),
            'psd_mean': float(psd_mean),
            'psd_std': float(psd_std),
            'laplacian_var': float(laplacian_var),
            'channel_stds': [float(x) for x in channel_stds],
            'edge_density': float(edge_density),
            'image_shape': img.shape
        }
        
        return profile
    
    def profile_directory(self, train_dir: str, output_path: str = None) -> List[Dict]:
        """
        Profile all images in the training directory.
        """
        train_path = Path(train_dir)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [
            f for f in train_path.iterdir()
            if f.suffix.lower() in image_extensions
        ]
        
        if not image_files:
            raise ValueError(f"No images found in {train_dir}")
        
        print(f"Profiling {len(image_files)} training images...")
        
        self.noise_profiles = []
        for img_path in tqdm(image_files, desc="Profiling"):
            try:
                profile = self.extract_noise_from_image(str(img_path))
                self.noise_profiles.append(profile)
            except Exception as e:
                print(f"Error profiling {img_path}: {e}")
        
        # Calculate global statistics
        self._calculate_global_stats()
        
        # Save profiles
        if output_path:
            self.save_profiles(output_path)
        
        return self.noise_profiles
    
    def _calculate_global_stats(self):
        """Calculate global statistics across all profiles"""
        if not self.noise_profiles:
            return
        
        self.global_stats = {
            'num_images': len(self.noise_profiles),
            'avg_sigma': np.mean([p['sigma'] for p in self.noise_profiles]),
            'avg_noise_std': np.mean([p['noise_std'] for p in self.noise_profiles]),
            'avg_laplacian_var': np.mean([p['laplacian_var'] for p in self.noise_profiles]),
            'avg_edge_density': np.mean([p['edge_density'] for p in self.noise_profiles]),
            'avg_channel_stds': np.mean([p['channel_stds'] for p in self.noise_profiles], axis=0).tolist()
        }
    
    def save_profiles(self, output_path: str):
        """Save noise profiles to JSON file"""
        data = {
            'profiles': self.noise_profiles,
            'global_stats': self.global_stats
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Noise profiles saved to {output_path}")
    
    def load_profiles(self, profile_path: str):
        """Load noise profiles from JSON file"""
        with open(profile_path, 'r') as f:
            data = json.load(f)
        
        self.noise_profiles = data['profiles']
        self.global_stats = data['global_stats']


class NoiseSynthesizer:
    """Synthesizes noise based on extracted profiles"""
    
    def __init__(self, noise_profiler: NoiseProfiler):
        self.profiler = noise_profiler
    
    def synthesize_noise(
        self,
        clean_image: np.ndarray,
        target_profile: Dict = None
    ) -> np.ndarray:
        """
        Synthesize noise on a clean image based on noise profile.
        Aims for 99% accuracy in mimicking noise patterns.
        """
        if target_profile is None:
            # Use average global statistics
            if not self.profiler.global_stats:
                raise ValueError("No noise profiles available. Please profile training images first.")
            sigma = self.profiler.global_stats['avg_sigma']
            noise_std = self.profiler.global_stats['avg_noise_std']
            channel_stds = self.profiler.global_stats['avg_channel_stds']
        else:
            # Individual profile has different key names
            sigma = target_profile.get('sigma', target_profile.get('avg_sigma', 0.01))
            noise_std = target_profile.get('noise_std', target_profile.get('avg_noise_std', 0.01))
            channel_stds = target_profile.get('channel_stds', target_profile.get('avg_channel_stds', [0.01, 0.01, 0.01]))
        
        img_float = clean_image.astype(np.float32) / 255.0
        h, w, c = img_float.shape
        
        # Generate base Gaussian noise
        noisy_img = img_float.copy()
        
        # Add channel-specific noise
        for i in range(c):
            # Gaussian noise component
            gaussian_noise = np.random.normal(0, channel_stds[i], (h, w))
            
            # Add spatial correlation to noise (realistic noise has correlation)
            gaussian_noise = ndimage.gaussian_filter(gaussian_noise, sigma=0.5)
            
            noisy_img[:, :, i] += gaussian_noise

        # Prevent negative or NaN intensities before Poisson sampling
        noisy_img = np.nan_to_num(noisy_img, nan=0.0)
        noisy_img = np.clip(noisy_img, 0.0, 1.0)
        
        # Add Poisson noise component (shot noise)
        # Scale image to have reasonable Poisson statistics
        scale_factor = 50.0
        noisy_img = noisy_img * scale_factor
        noisy_img = np.random.poisson(noisy_img) / scale_factor
        
        # Add salt-and-pepper noise (sparse)
        salt_pepper_prob = 0.001
        mask = np.random.random((h, w, c))
        noisy_img[mask < salt_pepper_prob / 2] = 0
        noisy_img[mask > 1 - salt_pepper_prob / 2] = 1
        
        # Add texture-preserving noise variation
        edges = cv2.Canny((img_float[:, :, 0] * 255).astype(np.uint8), 50, 150)
        edge_mask = edges.astype(np.float32) / 255.0
        edge_mask = cv2.GaussianBlur(edge_mask, (5, 5), 2)
        edge_mask = np.stack([edge_mask] * 3, axis=-1)
        
        # Less noise on edges, more in flat regions
        edge_noise = np.random.normal(0, noise_std * 0.3, (h, w, c))
        noisy_img += edge_noise * (1 - edge_mask)
        
        # Clip to valid range
        noisy_img = np.clip(noisy_img, 0, 1)
        
        # Convert back to uint8
        noisy_img = (noisy_img * 255).astype(np.uint8)
        
        return noisy_img
    
    def generate_synthetic_dataset(
        self,
        clean_dir: str,
        output_dir: str,
        use_random_profiles: bool = True
    ) -> int:
        """
        Generate synthetic noisy images from clean images.
        """
        clean_path = Path(clean_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        clean_out = output_path / 'clean'
        noisy_out = output_path / 'noisy'
        clean_out.mkdir(exist_ok=True)
        noisy_out.mkdir(exist_ok=True)
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        clean_files = [
            f for f in clean_path.iterdir()
            if f.suffix.lower() in image_extensions
        ]
        
        if not clean_files:
            raise ValueError(f"No images found in {clean_dir}")
        
        print(f"Generating synthetic dataset from {len(clean_files)} clean images...")
        
        generated_count = 0
        manifest = []
        
        for clean_file in tqdm(clean_files, desc="Synthesizing"):
            try:
                # Read clean image
                clean_img = cv2.imread(str(clean_file))
                if clean_img is None:
                    continue
                
                # Select noise profile
                if use_random_profiles and self.profiler.noise_profiles:
                    profile = np.random.choice(self.profiler.noise_profiles)
                else:
                    profile = None
                
                # Synthesize noise
                noisy_img = self.synthesize_noise(clean_img, profile)
                
                # Save images
                clean_output_path = clean_out / clean_file.name
                noisy_output_path = noisy_out / clean_file.name
                
                cv2.imwrite(str(clean_output_path), clean_img)
                cv2.imwrite(str(noisy_output_path), noisy_img)
                
                manifest.append({
                    'clean': str(clean_output_path),
                    'noisy': str(noisy_output_path),
                    'original': str(clean_file)
                })
                
                generated_count += 1
                
            except Exception as e:
                print(f"Error processing {clean_file}: {e}")
        
        # Save manifest
        manifest_path = output_path / 'manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"Generated {generated_count} synthetic image pairs")
        print(f"Manifest saved to {manifest_path}")
        
        return generated_count


def main():
    """Main function for noise profiling and synthesis"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Noise profiling and synthesis')
    parser.add_argument('--train_dir', type=str, default='input/train',
                        help='Directory with noisy training images')
    parser.add_argument('--clean_dir', type=str, default='input/clean',
                        help='Directory with clean images')
    parser.add_argument('--output_dir', type=str, default='output/synthetic',
                        help='Output directory for synthetic dataset')
    parser.add_argument('--profile_path', type=str, default='output/noise_profile.json',
                        help='Path to save/load noise profiles')
    
    args = parser.parse_args()
    
    # Step 1: Profile noisy training images
    print("=" * 60)
    print("STEP 1: Profiling noisy training images")
    print("=" * 60)
    profiler = NoiseProfiler()
    profiler.profile_directory(args.train_dir, args.profile_path)
    
    print(f"\nGlobal noise statistics:")
    for key, value in profiler.global_stats.items():
        print(f"  {key}: {value}")
    
    # Step 2: Generate synthetic noisy images
    print("\n" + "=" * 60)
    print("STEP 2: Generating synthetic noisy images")
    print("=" * 60)
    synthesizer = NoiseSynthesizer(profiler)
    count = synthesizer.generate_synthetic_dataset(
        args.clean_dir,
        args.output_dir,
        use_random_profiles=True
    )
    
    print(f"\n✓ Successfully generated {count} synthetic image pairs")
    print(f"✓ Clean images: {args.output_dir}/clean/")
    print(f"✓ Noisy images: {args.output_dir}/noisy/")


if __name__ == "__main__":
    main()

"""
Gradio Web UI for image denoising pipeline.
Allows users to upload images, train models, and denoise images.
"""

import gradio as gr
import torch
import os
import shutil
from pathlib import Path
import json
from datetime import datetime
from PIL import Image
import numpy as np

try:
    from .models.swin_unet import SwinUnet
    from .noise import NoiseProfiler, NoiseSynthesizer
    from .train import DenoisingTrainer
    from .test import ImageDenoiser
    from .utils.data_utils import create_dataloaders
except ImportError:
    from models.swin_unet import SwinUnet
    from noise import NoiseProfiler, NoiseSynthesizer
    from train import DenoisingTrainer
    from test import ImageDenoiser
    from utils.data_utils import create_dataloaders


class DenoisingApp:
    """Main application class for the denoising pipeline"""
    
    def __init__(self):
        self.base_dir = Path('.')
        self.train_dir = self.base_dir / 'input' / 'train'
        self.clean_dir = self.base_dir / 'input' / 'clean'
        self.test_dir = self.base_dir / 'input' / 'test'
        self.synthetic_dir = self.base_dir / 'output' / 'synthetic'
        self.models_dir = self.base_dir / 'output' / 'models'
        self.results_dir = self.base_dir / 'output' / 'results'

        # Canonical model + data defaults
        self.image_size = 256
        self.window_size = 8
        self.model_config = dict(
            img_size=self.image_size,
            patch_size=4,
            in_chans=3,
            out_chans=3,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=self.window_size,
        )
        
        # Ensure directories exist
        for dir_path in [self.train_dir, self.clean_dir, self.test_dir,
                         self.synthetic_dir, self.models_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.current_model_path = None
        self.denoiser = None
    
    def upload_training_images(self, files):
        """Upload noisy training images (accepts filepath strings or file objects)."""
        if not files:
            return "❌ No files selected"
        
        count = 0
        for file in files:
            try:
                src = Path(file) if not hasattr(file, "name") else Path(file.name)
                dest = self.train_dir / src.name
                shutil.copy(src, dest)
                count += 1
            except Exception as e:
                print(f"Error uploading {getattr(file, 'name', file)}: {e}")
        
        return f"✓ Uploaded {count} training images to {self.train_dir}"
    
    def upload_clean_images(self, files):
        """Upload clean images for synthetic noise generation (filepath strings or file objects)."""
        if not files:
            return "❌ No files selected"
        
        count = 0
        for file in files:
            try:
                src = Path(file) if not hasattr(file, "name") else Path(file.name)
                dest = self.clean_dir / src.name
                shutil.copy(src, dest)
                count += 1
            except Exception as e:
                print(f"Error uploading {getattr(file, 'name', file)}: {e}")
        
        return f"✓ Uploaded {count} clean images to {self.clean_dir}"
    
    def generate_synthetic_data(self, progress=gr.Progress()):
        """Generate synthetic noisy dataset"""
        try:
            progress(0, desc="Profiling noise...")
            
            # Check if training images exist
            train_images = list(self.train_dir.glob('*.png')) + \
                          list(self.train_dir.glob('*.jpg')) + \
                          list(self.train_dir.glob('*.jpeg'))
            
            if not train_images:
                return "❌ No training images found. Please upload training images first."
            
            # Check if clean images exist
            clean_images = list(self.clean_dir.glob('*.png')) + \
                          list(self.clean_dir.glob('*.jpg')) + \
                          list(self.clean_dir.glob('*.jpeg'))
            
            if not clean_images:
                return "❌ No clean images found. Please upload clean images first."
            
            # Profile noise
            profiler = NoiseProfiler()
            profiler.profile_directory(
                str(self.train_dir),
                str(self.base_dir / 'output' / 'noise_profile.json')
            )
            
            progress(0.5, desc="Generating synthetic noisy images...")
            
            # Generate synthetic dataset
            synthesizer = NoiseSynthesizer(profiler)
            count = synthesizer.generate_synthetic_dataset(
                str(self.clean_dir),
                str(self.synthetic_dir),
                use_random_profiles=True
            )
            
            progress(1.0, desc="Complete!")
            
            return f"✓ Generated {count} synthetic image pairs\n" + \
                   f"✓ Noise profile saved\n" + \
                   f"✓ Ready for training!"
        
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def train_model(self, device, epochs, batch_size, learning_rate, progress=gr.Progress()):
        """Train the denoising model"""
        try:
            # Check if synthetic dataset exists
            clean_path = self.synthetic_dir / 'clean'
            noisy_path = self.synthetic_dir / 'noisy'
            
            if not clean_path.exists() or not noisy_path.exists():
                return "❌ Synthetic dataset not found. Please generate synthetic data first."
            
            clean_count = len(list(clean_path.glob('*')))
            noisy_count = len(list(noisy_path.glob('*')))
            
            if clean_count == 0 or noisy_count == 0:
                return "❌ Synthetic dataset is empty. Please generate synthetic data first."
            
            # Set device
            device_str = 'cuda' if device == 'GPU (CUDA)' and torch.cuda.is_available() else 'cpu'
            if device == 'GPU (CUDA)' and device_str == 'cpu':
                return "❌ CUDA not available. Please select CPU or install CUDA."
            
            progress(0.1, desc="Loading dataset...")
            
            # Create dataloaders
            train_loader, val_loader = create_dataloaders(
                data_dir=str(self.synthetic_dir),
                batch_size=batch_size,
                image_size=self.image_size,
                train_split=0.9,
                num_workers=0
            )
            
            progress(0.2, desc="Creating model...")
            
            # Create model
            model = SwinUnet(
                **self.model_config
            )
            
            progress(0.3, desc="Starting training...")
            
            # Create trainer
            trainer = DenoisingTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device_str,
                learning_rate=learning_rate,
                output_dir=str(self.models_dir),
                log_dir=str(self.base_dir / 'logs')
            )
            
            # Train
            trainer.train(num_epochs=epochs)
            
            # Update current model path
            self.current_model_path = str(self.models_dir / 'best_model.pth')
            
            progress(1.0, desc="Training complete!")
            
            return f"✓ Training completed!\n" + \
                   f"✓ Best model saved to: {self.current_model_path}\n" + \
                   f"✓ Best PSNR: {trainer.best_psnr:.2f} dB\n" + \
                   f"✓ Device: {device_str}\n" + \
                   f"✓ Ready for testing!"
        
        except Exception as e:
            return f"❌ Training error: {str(e)}"
    
    def upload_test_images(self, files):
        """Upload test images for denoising (filepath strings or file objects)."""
        if not files:
            return "❌ No files selected"
        
        # Clear previous test images
        for f in self.test_dir.glob('*'):
            if f.is_file():
                f.unlink()
        
        count = 0
        for file in files:
            try:
                src = Path(file) if not hasattr(file, "name") else Path(file.name)
                dest = self.test_dir / src.name
                shutil.copy(src, dest)
                count += 1
            except Exception as e:
                print(f"Error uploading {getattr(file, 'name', file)}: {e}")
        
        return f"✓ Uploaded {count} test images"
    
    def denoise_images(self, model_path_input, device, progress=gr.Progress()):
        """Denoise uploaded test images"""
        try:
            # Determine model path (prefer explicit, then cached, then on-disk best/last)
            candidates = []
            if model_path_input and os.path.exists(model_path_input):
                candidates.append(model_path_input)
            if self.current_model_path and os.path.exists(self.current_model_path):
                candidates.append(self.current_model_path)
            best_path = self.models_dir / "best_model.pth"
            if best_path.exists():
                candidates.append(str(best_path))
            # fallback: latest checkpoint by mtime
            checkpoints = sorted(self.models_dir.glob("checkpoint_epoch_*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
            if checkpoints:
                candidates.append(str(checkpoints[0]))

            model_path = next((c for c in candidates if os.path.exists(c)), None)
            if not model_path:
                return "❌ No trained model found. Please train a model first or provide a model path."
            
            # Check test images
            test_images = list(self.test_dir.glob('*.png')) + \
                          list(self.test_dir.glob('*.jpg')) + \
                          list(self.test_dir.glob('*.jpeg'))
            
            if not test_images:
                return "❌ No test images found. Please upload test images first."
            
            progress(0.2, desc="Loading model...")
            
            # Set device
            device_str = 'cuda' if device == 'GPU (CUDA)' and torch.cuda.is_available() else 'cpu'
            
            # Create denoiser
            denoiser = ImageDenoiser(
                model_path=model_path,
                device=device_str,
                image_size=self.image_size,
                window_size=self.window_size,
                depths=self.model_config["depths"],
                num_heads=self.model_config["num_heads"],
                embed_dim=self.model_config["embed_dim"]
            )
            
            progress(0.4, desc="Denoising images...")
            
            # Denoise directory
            denoiser.denoise_directory(
                input_dir=str(self.test_dir),
                output_dir=str(self.results_dir),
                save_comparison=True
            )
            
            progress(1.0, desc="Denoising complete!")
            
            # Get result files
            result_files = list(self.results_dir.glob('*.png')) + \
                          list(self.results_dir.glob('*.jpg')) + \
                          list(self.results_dir.glob('*.jpeg'))
            
            # Filter out comparison files for display
            denoised_files = [f for f in result_files if not f.name.startswith('comparison_')]
            
            return f"✓ Denoised {len(denoised_files)} images\n" + \
                   f"✓ Results saved to: {self.results_dir}\n" + \
                   f"✓ Device: {device_str}"
        
        except Exception as e:
            return f"❌ Denoising error: {str(e)}"
    
    def get_results(self):
        """Get denoised result images for display"""
        result_files = list(self.results_dir.glob('*.png')) + \
                      list(self.results_dir.glob('*.jpg')) + \
                      list(self.results_dir.glob('*.jpeg'))
        
        # Filter out comparison files
        denoised_files = [f for f in result_files if not f.name.startswith('comparison_')]
        
        return [str(f) for f in denoised_files[:10]]  # Return first 10


def create_ui():
    """Create Gradio interface"""
    app = DenoisingApp()
    
    brand_css = """
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600&display=swap');
    :root {
        --bg: #f5f7fb;
        --card: #ffffff;
        --ink: #0f172a;
        --muted: #6b7280;
        --accent: #0ea5e9;
        --stroke: #e5e7eb;
    }
    body, .gradio-container {
        font-family: 'Space Grotesk', 'Segoe UI', sans-serif;
        background: radial-gradient(circle at 20% 20%, #e0f2fe 0, transparent 20%),
                    radial-gradient(circle at 80% 0%, #e8f0ff 0, transparent 18%),
                    var(--bg);
        color: var(--ink);
    }
    .hero {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 16px 20px;
        background: var(--card);
        border: 1px solid var(--stroke);
        border-radius: 12px;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
        margin-bottom: 12px;
    }
    .eyebrow {
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 12px;
        color: var(--muted);
        margin: 0 0 6px 0;
    }
    .hero h1 { margin: 0; font-size: 24px; color: var(--ink); }
    .hero p { margin: 4px 0 0 0; color: var(--muted); }
    .badge {
        display: inline-block;
        padding: 6px 10px;
        background: #e0f2fe;
        color: #0b75c9;
        border-radius: 8px;
        margin-left: 8px;
        font-size: 12px;
        border: 1px solid #b9e6ff;
    }
    .card-row { gap: 12px; }
    .card {
        background: var(--card);
        border: 1px solid var(--stroke);
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.04);
    }
    .card h3 { margin-top: 0; color: var(--ink); }
    .fine { color: var(--muted); font-size: 13px; margin-bottom: 12px; }
    .gr-button.primary, .gr-button-lg.primary {
        background: var(--accent);
        border: 1px solid var(--accent);
        color: #ffffff;
    }
    """

    with gr.Blocks(title="Image Denoising Console", css=brand_css) as demo:
        gr.HTML(
            f"""
            <div class="hero">
                <div>
                    <p class="eyebrow">Swin-Unet denoising</p>
                    <h1>Enterprise console</h1>
                    <p>Upload → Synthesize → Train → Denoise on one screen.</p>
                </div>
                <div>
                    <span class="badge">Image {app.image_size}px</span>
                    <span class="badge">Window {app.window_size}</span>
                    <span class="badge">Loss: L1 + Char + SSIM + Edge</span>
                </div>
            </div>
            """
        )

        with gr.Row(elem_classes="card-row"):
            with gr.Column(scale=1, elem_classes="card"):
                gr.Markdown("### Data")
                gr.Markdown("<div class='fine'>Upload noisy + clean sets, then synthesize training pairs.</div>")
                train_upload = gr.File(label="Noisy training images", file_count="multiple", type="filepath")
                train_upload_btn = gr.Button("Upload noisy set", variant="primary")
                train_status = gr.Textbox(label="Noisy status", lines=2)

                clean_upload = gr.File(label="Clean images", file_count="multiple", type="filepath")
                clean_upload_btn = gr.Button("Upload clean set", variant="primary")
                clean_status = gr.Textbox(label="Clean status", lines=2)

                synthetic_btn = gr.Button("Generate synthetic dataset", variant="primary")
                synthetic_status = gr.Textbox(label="Synthetic status", lines=4)

            with gr.Column(scale=1, elem_classes="card"):
                gr.Markdown("### Train")
                gr.Markdown("<div class='fine'>Trains with aligned 256px / window 8 config for sharper text.</div>")
                device_choice = gr.Radio(
                    choices=["CPU", "GPU (CUDA)"],
                    value="GPU (CUDA)" if torch.cuda.is_available() else "CPU",
                    label="Device"
                )
                epochs = gr.Slider(minimum=5, maximum=100, value=50, step=5, label="Epochs")
                batch_size = gr.Slider(minimum=1, maximum=32, value=8, step=1, label="Batch size")
                learning_rate = gr.Number(value=0.0001, label="Learning rate")
                train_btn = gr.Button("Start training", variant="primary")
                train_result = gr.Textbox(label="Training log", lines=7)

            with gr.Column(scale=1, elem_classes="card"):
                gr.Markdown("### Denoise")
                gr.Markdown("<div class='fine'>Upload test set, pick model, and denoise.</div>")
                test_upload = gr.File(label="Test images", file_count="multiple", type="filepath")
                test_upload_btn = gr.Button("Upload test set", variant="primary")
                test_upload_status = gr.Textbox(label="Test status", lines=2)

                model_path_input = gr.Textbox(
                    label="Model path (optional)",
                    placeholder="output/models/best_model.pth"
                )
                device_test = gr.Radio(
                    choices=["CPU", "GPU (CUDA)"],
                    value="GPU (CUDA)" if torch.cuda.is_available() else "CPU",
                    label="Device"
                )
                denoise_btn = gr.Button("Denoise images", variant="primary")
                denoise_status = gr.Textbox(label="Denoise status", lines=4)

        with gr.Row(elem_classes="card-row"):
            with gr.Column(scale=1, elem_classes="card"):
                gr.Markdown("### Results")
                results_gallery = gr.Gallery(
                    label="Denoised previews",
                    columns=4,
                    height=340,
                    object_fit="contain",
                )
                refresh_btn = gr.Button("Refresh results", variant="primary")

        # Event handlers
        train_upload_btn.click(fn=app.upload_training_images, inputs=[train_upload], outputs=[train_status])
        clean_upload_btn.click(fn=app.upload_clean_images, inputs=[clean_upload], outputs=[clean_status])
        synthetic_btn.click(fn=app.generate_synthetic_data, inputs=[], outputs=[synthetic_status])
        train_btn.click(fn=app.train_model, inputs=[device_choice, epochs, batch_size, learning_rate], outputs=[train_result])
        test_upload_btn.click(fn=app.upload_test_images, inputs=[test_upload], outputs=[test_upload_status])
        denoise_btn.click(fn=app.denoise_images, inputs=[model_path_input, device_test], outputs=[denoise_status])
        refresh_btn.click(fn=app.get_results, inputs=[], outputs=[results_gallery])
    
    return demo


def main():
    """Launch the Gradio app"""
    import os
    port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True,
        inbrowser=True
    )


if __name__ == "__main__":
    main()

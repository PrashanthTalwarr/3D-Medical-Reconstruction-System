import os
import json
import torch
import numpy as np
import base64
import io

# Import model definition
from model import UNet3D
from diffusion import DDPM3D


def model_fn(model_dir):
    """
    Load the PyTorch model from the `model_dir` directory.
    Called once when the endpoint starts up.
    """
    print(f"Loading model from {model_dir}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = UNet3D(
        in_channels=1,
        out_channels=1,
        base_channels=32,
        time_emb_dim=128
    )
    
    # Load weights
    model_path = os.path.join(model_dir, 'model.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully on {device}")
    
    return model


def input_fn(request_body, request_content_type):
    """
    Deserialize and prepare the prediction input.
    """
    print(f"Received content type: {request_content_type}")
    
    if request_content_type == 'application/json':
        data = json.loads(request_body)
        
        # Decode base64 volume
        volume_bytes = base64.b64decode(data['volume'])
        volume = np.frombuffer(volume_bytes, dtype=np.float32)
        volume = volume.reshape(data['shape'])
        
        # Convert to tensor [B, C, D, H, W]
        volume_tensor = torch.FloatTensor(volume).unsqueeze(0).unsqueeze(0)
        
        return volume_tensor
    
    elif request_content_type == 'application/x-npy':
        # Direct numpy array
        stream = io.BytesIO(request_body)
        volume = np.load(stream)
        return torch.FloatTensor(volume).unsqueeze(0).unsqueeze(0)
    
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    """
    Apply the model to the incoming request.
    """
    print("Running inference...")
    
    device = next(model.parameters()).device
    input_data = input_data.to(device)
    
    # Initialize diffusion
    diffusion = DDPM3D(timesteps=200, device=device)
    
    with torch.no_grad():
        # Option 1: Single-step denoising (fast)
        if input_data.shape[2] <= 32:  # For small volumes, do full sampling
            reconstructed = diffusion.sample(model, input_data.shape, device)
        else:
            # Option 2: For large volumes, do partial denoising
            t = torch.full((input_data.shape[0],), diffusion.timesteps // 4, device=device).long()
            predicted_noise = model(input_data, t)
            reconstructed = diffusion.predict_start_from_noise(input_data, t, predicted_noise)
    
    return reconstructed.cpu()


def output_fn(prediction, response_content_type):
    """
    Serialize the prediction result.
    """
    print(f"Preparing response with content type: {response_content_type}")
    
    if response_content_type == 'application/json':
        # Convert to numpy and encode as base64
        volume_np = prediction.numpy()
        volume_bytes = volume_np.astype(np.float32).tobytes()
        volume_b64 = base64.b64encode(volume_bytes).decode('utf-8')
        
        response = {
            'reconstructed_volume': volume_b64,
            'shape': list(volume_np.shape),
            'dtype': str(volume_np.dtype)
        }
        
        return json.dumps(response)
    
    elif response_content_type == 'application/x-npy':
        # Return as numpy array
        stream = io.BytesIO()
        np.save(stream, prediction.numpy())
        return stream.getvalue()
    
    else:
        raise ValueError(f"Unsupported content type: {response_content_type}")

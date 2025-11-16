import boto3
import sagemaker
from sagemaker.pytorch import PyTorch, PyTorchModel
from sagemaker import get_execution_role
import time
import os
import shutil

def prepare_sagemaker_files():
    """Copy necessary files to sagemaker directory"""
    print("Preparing SageMaker deployment files...")
    
    # Ensure sagemaker directory exists
    os.makedirs('sagemaker', exist_ok=True)
    
    # Copy model files from local to sagemaker
    files_to_copy = [
        ('local/model.py', 'sagemaker/model.py'),
        ('local/diffusion.py', 'sagemaker/diffusion.py'),
        ('local/dataset.py', 'sagemaker/dataset.py'),
        ('local/cuda_ops.py', 'sagemaker/cuda_ops.py'),
    ]
    
    for src, dst in files_to_copy:
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  ✓ Copied {src} → {dst}")
        else:
            print(f"  ⚠ Warning: {src} not found")
    
    print("✓ Files prepared for SageMaker\n")


def deploy_to_sagemaker():
    """Complete SageMaker deployment workflow"""
    
    print("\n" + "="*70)
    print("AWS SAGEMAKER DEPLOYMENT - 3D MEDICAL RECONSTRUCTION")
    print("="*70 + "\n")
    
    # Prepare files
    prepare_sagemaker_files()
    
    # Initialize SageMaker session
    try:
        session = sagemaker.Session()
        role = get_execution_role()
        bucket = session.default_bucket()
        region = session.boto_region_name
        
        print(f"✓ SageMaker session initialized")
        print(f"✓ Region: {region}")
        print(f"✓ S3 Bucket: {bucket}")
        print(f"✓ IAM Role: {role[:50]}...")
        print()
        
    except Exception as e:
        print(f"\n✗ Failed to initialize SageMaker session:")
        print(f"  Error: {e}")
        print(f"\nPlease ensure:")
        print(f"  1. AWS credentials are configured (run: aws configure)")
        print(f"  2. You have SageMaker execution role permissions")
        print(f"  3. You're running in a SageMaker notebook OR have configured IAM role")
        return None
    
    # ========================================================================
    # STEP 1: Upload Training Data to S3 (Optional - we use synthetic data)
    # ========================================================================
    
    print("[Step 1/4] Preparing training data...")
    
    # For this demo, we generate synthetic data on SageMaker
    # In production, upload real medical images here
    train_input = f's3://{bucket}/med-recon/data/train'
    
    print(f"✓ Training data location: {train_input}")
    print("  (Synthetic data will be generated during training)\n")
    
    # ========================================================================
    # STEP 2: Launch SageMaker Training Job
    # ========================================================================
    
    print("[Step 2/4] Configuring SageMaker training job...")
    print()
    print("  Instance Type: ml.p3.2xlarge (Tesla V100 16GB)")
    print("  Spot Instances: Enabled (70% cost reduction)")
    print("  Estimated Cost: ~$1-2 for full training")
    print("  Estimated Time: 45-60 minutes")
    print()
    
    # Create PyTorch estimator
    estimator = PyTorch(
        entry_point='train.py',
        source_dir='sagemaker',
        role=role,
        
        # GPU instance
        instance_type='ml.p3.2xlarge',  # V100 GPU
        instance_count=1,
        
        # Framework
        framework_version='2.0',
        py_version='py310',
        
        # Hyperparameters
        hyperparameters={
            'epochs': 50,
            'batch_size': 4,
            'learning_rate': 1e-4,
            'timesteps': 200,
            'num_samples': 5000,
            'volume_size': 32,
            'base_channels': 32,
        },
        
        # Cost optimization with spot instances
        use_spot_instances=True,
        max_wait=7200,  # 2 hours max wait
        max_run=3600,   # 1 hour max training
        
        # Output locations
        output_path=f's3://{bucket}/med-recon/models/',
        checkpoint_s3_uri=f's3://{bucket}/med-recon/checkpoints/',
        
        # Monitoring
        enable_sagemaker_metrics=True,
        
        # Environment
        environment={
            'PYTHONUNBUFFERED': '1',
        },
    )
    
    print("✓ Training job configured")
    print()
    
    # Launch training
    input_decision = input("Launch training job now? This will incur AWS charges (~$1-2). (yes/no): ")
    
    if input_decision.lower() not in ['yes', 'y']:
        print("\n⚠ Training job cancelled by user")
        print("  To launch later, run this script again or use:")
        print(f"    estimator.fit('{train_input}')")
        return estimator
    
    print("\n🚀 Launching training job...")
    print("  You can monitor progress in:")
    print(f"    - AWS Console: https://console.aws.amazon.com/sagemaker/")
    print(f"    - CloudWatch Logs")
    print()
    
    try:
        estimator.fit({'training': train_input}, wait=True, logs='All')
        
        print("\n✓ Training job completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Training job failed: {e}")
        print("  Check CloudWatch logs for details")
        return None
    
    # ========================================================================
    # STEP 3: Deploy Model to Inference Endpoint
    # ========================================================================
    
    print("\n[Step 3/4] Deploying model to inference endpoint...")
    print("  Instance Type: ml.g4dn.xlarge (cost-effective GPU)")
    print("  Estimated Cost: ~$0.52/hour")
    print()
    
    deploy_decision = input("Deploy inference endpoint? This creates ongoing charges. (yes/no): ")
    
    if deploy_decision.lower() not in ['yes', 'y']:
        print("\n⚠ Deployment cancelled")
        print("  Model is saved in S3 and can be deployed later")
        print(f"  Model location: {estimator.model_data}")
        return estimator
    
    try:
        predictor = estimator.deploy(
            initial_instance_count=1,
            instance_type='ml.g4dn.xlarge',
            endpoint_name=f'med-recon-{int(time.time())}',
            
            # Serialization
            serializer=sagemaker.serializers.JSONSerializer(),
            deserializer=sagemaker.deserializers.JSONDeserializer(),
        )
        
        print(f"\n✓ Endpoint deployed successfully!")
        print(f"✓ Endpoint name: {predictor.endpoint_name}")
        
    except Exception as e:
        print(f"\n✗ Deployment failed: {e}")
        return estimator
    
    # ========================================================================
    # STEP 4: Test Inference
    # ========================================================================
    
    print("\n[Step 4/4] Testing inference endpoint...")
    
    try:
        import numpy as np
        import base64
        
        # Create test volume
        test_volume = np.random.randn(32, 32, 32).astype(np.float32)
        volume_bytes = test_volume.tobytes()
        volume_b64 = base64.b64encode(volume_bytes).decode('utf-8')
        
        # Prepare request
        request = {
            'volume': volume_b64,
            'shape': [32, 32, 32]
        }
        
        print("  Sending test request...")
        response = predictor.predict(request)
        
        print("✓ Inference successful!")
        print(f"  Response shape: {response['shape']}")
        
    except Exception as e:
        print(f"⚠ Inference test failed: {e}")
        print("  Endpoint is deployed but test failed")
    
    # ========================================================================
    # Summary
    # ========================================================================
    
    print("\n" + "="*70)
    print("DEPLOYMENT COMPLETE!")
    print("="*70)
    
    print("\n📊 Summary:")
    print(f"  ✓ Training: Complete")
    print(f"  ✓ Model: {estimator.model_data}")
    print(f"  ✓ Endpoint: {predictor.endpoint_name}")
    
    print("\n💰 Cost Breakdown (Estimated):")
    print(f"  Training (1 hour, spot): ~$0.90")
    print(f"  Inference: ~$0.52/hour (ongoing)")
    print(f"  S3 Storage: ~$0.10/month")
    
    print("\n🔗 Resources:")
    print(f"  AWS Console: https://console.aws.amazon.com/sagemaker/")
    print(f"  Endpoint: {predictor.endpoint_name}")
    print(f"  Region: {region}")
    
    print("\n⚠️  IMPORTANT - Cost Management:")
    print(f"  To stop charges, delete the endpoint:")
    print(f"    aws sagemaker delete-endpoint --endpoint-name {predictor.endpoint_name}")
    print(f"  Or use the cleanup function:")
    print(f"    cleanup_endpoint('{predictor.endpoint_name}')")
    
    print("\n📝 Next Steps:")
    print("  1. Test endpoint with real medical images")
    print("  2. Monitor CloudWatch metrics")
    print("  3. Set up auto-scaling if needed")
    print("  4. Integrate into production pipeline")
    
    print("\n" + "="*70 + "\n")
    
    return predictor


def cleanup_endpoint(endpoint_name):
    """Delete SageMaker endpoint to stop charges"""
    print(f"\n🗑️  Deleting endpoint: {endpoint_name}")
    
    try:
        client = boto3.client('sagemaker')
        
        # Delete endpoint
        client.delete_endpoint(EndpointName=endpoint_name)
        print(f"✓ Endpoint deleted: {endpoint_name}")
        
        # Delete endpoint configuration
        config_name = endpoint_name
        try:
            client.delete_endpoint_config(EndpointConfigName=config_name)
            print(f"✓ Endpoint config deleted: {config_name}")
        except:
            pass
        
        print("✓ Cleanup complete - charges stopped")
        
    except Exception as e:
        print(f"✗ Cleanup failed: {e}")
        print("  You may need to delete manually from AWS Console")


def list_endpoints():
    """List all active SageMaker endpoints"""
    try:
        client = boto3.client('sagemaker')
        response = client.list_endpoints()
        
        print("\n📋 Active SageMaker Endpoints:")
        print("-" * 70)
        
        if not response['Endpoints']:
            print("  No active endpoints found")
        else:
            for ep in response['Endpoints']:
                print(f"  • {ep['EndpointName']}")
                print(f"    Status: {ep['EndpointStatus']}")
                print(f"    Created: {ep['CreationTime']}")
                print()
        
        print("-" * 70)
        
    except Exception as e:
        print(f"✗ Failed to list endpoints: {e}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import sys
    
    print("\n" + "="*70)
    print("3D MEDICAL RECONSTRUCTION - AWS SAGEMAKER DEPLOYMENT")
    print("="*70)
    print("\nTechnologies:")
    print("  ✓ PyTorch - Deep Learning")
    print("  ✓ Diffusion Models (DDPM)")
    print("  ✓ 3D Vision - Volumetric Processing")
    print("  ✓ CUDA - GPU Acceleration")
    print("  ✓ AWS SageMaker - Cloud Training & Deployment")
    print("="*70 + "\n")
    
    # Check AWS credentials
    try:
        boto3.client('sts').get_caller_identity()
        print("✓ AWS credentials configured\n")
    except Exception as e:
        print("✗ AWS credentials not configured!")
        print("  Please run: aws configure")
        print("  And enter your AWS Access Key and Secret Key\n")
        sys.exit(1)
    
    # Check if running in SageMaker Studio or need role
    try:
        role = get_execution_role()
        print(f"✓ Using SageMaker execution role\n")
    except:
        print("⚠ Not running in SageMaker environment")
        print("  You need to configure IAM role for SageMaker")
        print("  See: https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html\n")
        
        proceed = input("Proceed anyway? (yes/no): ")
        if proceed.lower() not in ['yes', 'y']:
            sys.exit(0)
    
    # Main deployment
    if len(sys.argv) > 1 and sys.argv[1] == 'list':
        list_endpoints()
    elif len(sys.argv) > 1 and sys.argv[1] == 'cleanup':
        if len(sys.argv) > 2:
            cleanup_endpoint(sys.argv[2])
        else:
            print("Usage: python sagemaker_deploy.py cleanup ENDPOINT_NAME")
    else:
        predictor = deploy_to_sagemaker()
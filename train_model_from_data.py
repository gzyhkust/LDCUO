import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import struct
import os

# Specify to use GPU 0, 1, 2, 3
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

# Multi-GPU distributed training support
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

class SiameseNetwork(nn.Module):
    """Siamese network for vector similarity learning"""
    def __init__(self, input_dim=960, embed_dim=32):
        super(SiameseNetwork, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Linear(128, embed_dim)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Linear(128, embed_dim)
        )

        self.fc3 = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Linear(128, embed_dim)
        )

    def forward_fc1(self, x):
        """Forward propagation using fc1"""
        return self.fc1(x)
    
    def forward_fc2(self, x):
        """Forward propagation using fc2"""
        return self.fc2(x)
    
    def forward_fc3(self, x):
        """Forward propagation using fc3"""
        return self.fc3(x)

    def forward(self, input1, input2):
        """Forward propagation - combined use of fc1, fc2, fc3"""
        # Combined version: use average of fc1, fc2, fc3
        output1 = (self.fc1(input1) + self.fc2(input1) + self.fc3(input1)) / 3
        output2 = (self.fc1(input2) + self.fc2(input2) + self.fc3(input2)) / 3
        return output1, output2
    
    def copy_fc1_to_fc2(self):
        """Copy fc1 parameters to fc2"""
        with torch.no_grad():
            for p_src, p_dst in zip(self.fc1.parameters(), self.fc2.parameters()):
                p_dst.copy_(p_src)
    
    def copy_fc2_to_fc3(self):
        """Copy fc2 parameters to fc3"""
        with torch.no_grad():
            for p_src, p_dst in zip(self.fc2.parameters(), self.fc3.parameters()):
                p_dst.copy_(p_src)


class TrainingDataset:
    """Binary training dataset"""
    def __init__(self):
        self.dim = 0
        self.all_vectors = []
        self.landmarks = []
    
    @staticmethod
    def load_binary(filename):
        """
        Load training dataset from binary file
        
        Format description:
        [Header]
        - dim (uint32_t): Vector dimension
        - num_landmarks (uint32_t): Number of landmarks
        
        [Landmarks section]
        For each landmark:
          - landmark_id (uint32_t)
          - landmark_vector: dim floats
          - hop1_count (uint32_t)
          - hop1_vectors: hop1_count * dim floats
          - hop2_count (uint32_t)
          - hop2_vectors: hop2_count * dim floats
          - hop3_count (uint32_t)
          - hop3_vectors: hop3_count * dim floats
        """
        dataset = TrainingDataset()
        
        with open(filename, 'rb') as f:
            dataset._load_from_stream(f)
        
        return dataset
    
    @staticmethod
    def load_from_stdin():
        """
        Read binary data from standard input (for pipe transfer, avoiding disk writes)
        """
        import sys
        dataset = TrainingDataset()
        
        # 从stdin读取二进制数据
        stdin_buffer = sys.stdin.buffer
        dataset._load_from_stream(stdin_buffer)
        
        return dataset
    
    def _load_from_stream(self, f):
        """
        Internal method to read binary data from file stream/stdin
        """
        # Read header
        header_data = f.read(8)
        if len(header_data) < 8:
            raise ValueError(f"Failed to read header: only read {len(header_data)} bytes")
        header = struct.unpack('<II', header_data)
        self.dim = header[0]
        num_landmarks = header[1]
        
        print(f"[Read] Header info: dim={self.dim}, num_landmarks={num_landmarks}")
        
        # Read landmarks section
        self.landmarks = []
        self.all_vectors = []
        
        for landmark_idx in range(num_landmarks):
            # Read landmark_id
            landmark_id = struct.unpack('<I', f.read(4))[0]
            
            # Read landmark vector (dim floats)
            vec_bytes = f.read(self.dim * 4)
            landmark_vec = struct.unpack(f'<{self.dim}f', vec_bytes)
            landmark_vec = np.array(landmark_vec, dtype=np.float32)
            
            # Collect all vectors for later processing
            all_vec_list = [landmark_vec]
            
            # Read hop1
            hop1_count = struct.unpack('<I', f.read(4))[0]
            hop1_vecs = []
            for _ in range(hop1_count):
                vec = struct.unpack(f'<{self.dim}f', f.read(self.dim * 4))
                vec = np.array(vec, dtype=np.float32)
                hop1_vecs.append(vec)
                all_vec_list.append(vec)
            
            # Read hop2
            hop2_count = struct.unpack('<I', f.read(4))[0]
            hop2_vecs = []
            for _ in range(hop2_count):
                vec = struct.unpack(f'<{self.dim}f', f.read(self.dim * 4))
                vec = np.array(vec, dtype=np.float32)
                hop2_vecs.append(vec)
                all_vec_list.append(vec)
            
            # Read hop3
            hop3_count = struct.unpack('<I', f.read(4))[0]
            hop3_vecs = []
            for _ in range(hop3_count):
                vec = struct.unpack(f'<{self.dim}f', f.read(self.dim * 4))
                vec = np.array(vec, dtype=np.float32)
                hop3_vecs.append(vec)
                all_vec_list.append(vec)
            
            # Build landmark data
            landmark = {
                'id': landmark_id,
                'vector': landmark_vec,
                'hop1': hop1_vecs,
                'hop2': hop2_vecs,
                'hop3': hop3_vecs
            }
            self.landmarks.append(landmark)
            
            # Save all vectors
            self.all_vectors.extend(all_vec_list)
            
            if (landmark_idx + 1) % 100 == 0:
                print(f"  Read {landmark_idx + 1}/{num_landmarks} landmarks")
        
        print(f"[Read] ✓ Data loading complete")
        print(f"  - Vector dimension: {self.dim}")
        print(f"  - Landmark count: {len(self.landmarks)}")
        print(f"  - Total vectors: {len(self.all_vectors)}")


def mse_loss_batch(real_dists, embed_dists):
    """Calculate MSE loss"""
    return np.mean((real_dists - embed_dists.detach().cpu().numpy()) ** 2)


def setup_distributed():
    """
    Initialize distributed training environment
    Returns: (rank, world_size, device)
    
    Note:
    - Multi-GPU training requires starting with torchrun, e.g.:
      torchrun --nproc_per_node=4 train_model_from_data.py ...
    - Running python train_model_from_data.py directly uses single GPU
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # Started by torchrun, enter distributed mode
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        print(f"[Distributed] Process {rank}/{world_size} using GPU {local_rank}")
        
        return rank, world_size, device
    else:
        # Running script directly, use single GPU mode
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            num_gpus = torch.cuda.device_count()
            if num_gpus > 1:
                print(f"[Note] Detected {num_gpus} GPUs, but currently in single GPU mode")
                print(f"[Note] For multi-GPU training, please use the following command:")
                print(f"       torchrun --nproc_per_node={num_gpus} train_model_from_data.py <args...>")
        else:
            device = torch.device('cpu')
            print("[Warning] No GPU detected, using CPU for training")
        
        return 0, 1, device


def cleanup_distributed():
    """Cleanup distributed training environment"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    """Check if this is the main process (rank 0)"""
    return rank == 0


def train_model(data_source, output_model="siamese_model.pt", epochs=5, batch_size=128, lr=0.001, from_stdin=False, use_multi_gpu=True):
    """
    Train model from binary data (staged training, multi-GPU support)
    
    Training flow:
    1. Train fc1 with hop1 data
    2. Copy fc1 parameters to fc2, train fc2 with hop2 data
    3. Copy fc2 parameters to fc3, train fc3 with hop3 data
    
    Args:
        data_source: Binary format training data file path, or "stdin" to read from standard input
        output_model: Output model path
        epochs: Training epochs per stage
        batch_size: Batch size (default 128, batch_size per GPU in multi-GPU mode)
        lr: Learning rate
        from_stdin: Whether to read data from stdin
        use_multi_gpu: Whether to use multi-GPU training
    """
    # Initialize distributed environment
    rank, world_size, device = setup_distributed()
    is_main = is_main_process(rank)
    is_distributed = (world_size > 1)
    
    if is_main:
        print(f"\n[Multi-GPU] Detected {world_size} GPUs, {'distributed' if is_distributed else 'single GPU'} training")
    
    # Multi-GPU mode: only rank 0 loads data, then sync
    # Note: Multi-GPU mode does not support stdin because each process needs to read complete data
    if from_stdin and is_distributed:
        raise ValueError("Multi-GPU distributed training does not support reading from stdin, please use file method to pass data")
    
    if from_stdin:
        if is_main:
            print(f"[Training] Reading data from standard input...")
        dataset = TrainingDataset.load_from_stdin()
    else:
        if is_main:
            print(f"[Training] Loading data: {data_source}")
        # All processes load data from file (files can be read simultaneously by multiple processes)
        dataset = TrainingDataset.load_binary(data_source)
    
    # Sync during distributed training, ensure all processes have loaded data
    if is_distributed:
        dist.barrier()
    
    if is_main:
        print(f"[Training] ✓ Loading successful")
    
    # Create model
    model = SiameseNetwork(input_dim=dataset.dim, embed_dim=32).to(device)
    
    # If multi-GPU, wrap as DDP model
    if world_size > 1:
        model = DDP(model, device_ids=[device.index], output_device=device.index)
        if is_main:
            print(f"\n[Model] DDP model created, using {world_size} GPUs")
    else:
        if is_main:
            print(f"\n[Model] Model created, device: {device}")
    
    # Get actual model (need to access module after DDP wrapping)
    raw_model = model.module if world_size > 1 else model
    
    # ========== Stage 1: Train fc1 with hop1 ==========
    if is_main:
        print("\n" + "="*60)
        print("[Stage 1] Training fc1 with hop1 neighbors")
        print("="*60)
    
    train_fc_with_landmarks(
        model, raw_model.fc1, raw_model.forward_fc1,
        dataset.landmarks, dataset,
        hop_level=1, epochs=epochs, batch_size=batch_size, device=device, lr=lr,
        rank=rank, world_size=world_size
    )
    
    # ========== Stage 2: Copy fc1 parameters to fc2, train fc2 with hop2 ==========
    if is_main:
        print("\n" + "="*60)
        print("[Stage 2] Copying fc1 parameters to fc2, training fc2 with hop2 neighbors")
        print("="*60)
    
    # Copy fc1 parameters to fc2
    raw_model.copy_fc1_to_fc2()
    if is_main:
        print("[Parameter Transfer] ✓ fc1 parameters copied to fc2")
    
    train_fc_with_landmarks(
        model, raw_model.fc2, raw_model.forward_fc2,
        dataset.landmarks, dataset,
        hop_level=2, epochs=epochs, batch_size=batch_size, device=device, lr=lr,
        rank=rank, world_size=world_size
    )
    
    # ========== Stage 3: Copy fc2 parameters to fc3, train fc3 with hop3 ==========
    if is_main:
        print("\n" + "="*60)
        print("[Stage 3] Copying fc2 parameters to fc3, training fc3 with hop3 neighbors")
        print("="*60)
    
    # Copy fc2 parameters to fc3
    raw_model.copy_fc2_to_fc3()
    if is_main:
        print("[Parameter Transfer] ✓ fc2 parameters copied to fc3")
    
    train_fc_with_landmarks(
        model, raw_model.fc3, raw_model.forward_fc3,
        dataset.landmarks, dataset,
        hop_level=3, epochs=epochs, batch_size=batch_size, device=device, lr=lr,
        rank=rank, world_size=world_size
    )
    
    # ========== Stage 4: Contrastive learning fine-tuning ==========
    print("\n" + "="*60)
    print("[Stage 4] Contrastive learning fine-tuning using Triplet Loss")
    print("="*60)
    
    contrastive_train(
        model, dataset.landmarks, dataset,
        epochs=epochs, batch_size=batch_size, device=device, lr=lr*0.5, margin=1.0
    )
    
    # ========== Save model ==========
    # Only save model on main process
    if is_main:
        print("\n" + "="*60)
        print("[Save] Saving model")
        print("="*60)
        
        # Get original model (non-DDP wrapped)
        model_to_save = raw_model
        model_to_save.eval()
        
        try:
            # Try script method
            scripted_model = torch.jit.script(model_to_save)
            scripted_model.save(output_model)
            print(f"✓ Model saved (TorchScript): {output_model}")
        except Exception as e:
            print(f"✗ TorchScript export failed: {e}")
            print("Trying trace method...")
            try:
                dummy_input = torch.randn(1, dataset.dim).to(device)
                scripted_model = torch.jit.trace(model_to_save, (dummy_input, dummy_input))
                scripted_model.save(output_model)
                print(f"✓ Model saved (Trace): {output_model}")
            except Exception as e2:
                # If trace fails, save as state_dict
                print(f"✗ Trace export also failed: {e2}")
                print("Saving as PyTorch state_dict...")
                torch.save(model_to_save.state_dict(), output_model.replace('.pt', '_state_dict.pt'))
                print(f"✓ State dict saved: {output_model.replace('.pt', '_state_dict.pt')}")
    
    # Cleanup distributed environment
    cleanup_distributed()
    
    return raw_model


def train_fc_with_landmarks(model, fc_module, forward_fn, landmarks, dataset, 
                           hop_level, epochs, batch_size, device, lr,
                           rank=0, world_size=1):
    """
    Train a fc module with landmarks hop neighbors (memory optimized version - supports ultra-large datasets and multi-GPU)
    
    Optimizations:
    1. Use custom Dataset for lazy loading, no pre-integration of all data
    2. Memory usage only related to batch_size, not total data volume
    3. Supports arbitrary size datasets
    4. Supports multi-GPU distributed training (DDP)
    
    Args:
        model: Complete model (may be DDP wrapped)
        fc_module: The fc module to train (fc1/fc2/fc3)
        forward_fn: Corresponding forward propagation function
        landmarks: Landmark list, each landmark contains vectors and hop neighbor vectors
        dataset: TrainingDataset object
        hop_level: Hop level (1/2/3)
        epochs: Training epochs
        batch_size: Batch size (batch_size per GPU)
        device: Device
        lr: Learning rate
        rank: Current process rank (for distributed training)
        world_size: Total number of processes (GPU count)
    """
    from torch.utils.data import Dataset, DataLoader
    
    is_main = (rank == 0)
    is_distributed = (world_size > 1)
    
    # Get hop neighbors key
    hop_key = f'hop{hop_level}'
    
    if is_main:
        effective_batch_size = batch_size * world_size
        print(f"[Setup] hop level: {hop_level}, per-GPU batch size: {batch_size}, effective batch size: {effective_batch_size}, lr: {lr}, epochs: {epochs}")
    
    # ========== Custom Dataset: Lazy loading, no pre-integration of data ==========
    class LandmarkNeighborDataset(Dataset):
        """
        Lazy loading Dataset: Only read data when needed
        Memory usage: O(index array size) instead of O(all vectors)
        """
        def __init__(self, landmarks, hop_key):
            # Only store indices, not actual vector data
            self.landmarks = landmarks
            self.hop_key = hop_key
            
            # Build index mapping: (landmark_idx, neighbor_idx) -> global_idx
            self.index_map = []
            for lm_idx, landmark in enumerate(landmarks):
                neighbor_vecs = landmark[hop_key]
                for nb_idx in range(len(neighbor_vecs)):
                    self.index_map.append((lm_idx, nb_idx))
            
            if is_main:
                print(f"[Dataset] Total samples: {len(self.index_map)}")
        
        def __len__(self):
            return len(self.index_map)
        
        def __getitem__(self, idx):
            lm_idx, nb_idx = self.index_map[idx]
            landmark = self.landmarks[lm_idx]
            
            landmark_vec = landmark['vector']
            neighbor_vec = landmark[self.hop_key][nb_idx]
            
            # Calculate real L2 distance
            real_dist = np.linalg.norm(landmark_vec - neighbor_vec)
            
            return (
                torch.from_numpy(landmark_vec),
                torch.from_numpy(neighbor_vec),
                torch.tensor(real_dist, dtype=torch.float32)
            )
    
    # Create Dataset and DataLoader
    train_dataset = LandmarkNeighborDataset(landmarks, hop_key)
    
    if len(train_dataset) == 0:
        if is_main:
            print(f"[Warning] hop{hop_level} has no training data!")
        return
    
    # Estimate memory usage
    dim = landmarks[0]['vector'].shape[0] if landmarks else 0
    index_mem = len(train_dataset.index_map) * 16 / 1e6  # Approximately 16 bytes per index
    batch_mem = batch_size * dim * 4 * 2 / 1e6  # Two vectors per batch
    if is_main:
        print(f"[Memory] Index usage: {index_mem:.1f}MB, per-GPU per-batch usage: {batch_mem:.1f}MB")
    
    use_pin_memory = (device.type == 'cuda')
    
    # Use DistributedSampler for distributed training
    if is_distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        shuffle = False  # DistributedSampler already handles shuffle
    else:
        train_sampler = None
        shuffle = True
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=4,  # Multi-process loading
        pin_memory=use_pin_memory,
        prefetch_factor=2,  # Prefetch batch count
        persistent_workers=True  # Keep worker processes
    )
    
    if is_main:
        print(f"[Data] ✓ DataLoader created, batches per GPU: {len(train_loader)}")
        if is_distributed:
            print(f"[Distributed] Using DistributedSampler, data automatically sharded to {world_size} GPUs")
    
    # ========== Training loop ==========
    optimizer = optim.Adam(fc_module.parameters(), lr=lr)
    
    total_loss_all_epochs = 0
    
    for epoch in range(epochs):
        # For distributed training, set sampler epoch each epoch to ensure correct shuffle
        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_landmarks, batch_neighbors, batch_real_dists in train_loader:
            # Transfer each batch to GPU
            batch_landmarks = batch_landmarks.to(device, non_blocking=True)
            batch_neighbors = batch_neighbors.to(device, non_blocking=True)
            batch_real_dists = batch_real_dists.to(device, non_blocking=True)
            
            # Forward propagation: compute distance after embedding
            landmark_embed = forward_fn(batch_landmarks)
            neighbor_embed = forward_fn(batch_neighbors)
            
            # Calculate L2 distance in embedding space
            embed_distances = torch.norm(landmark_embed - neighbor_embed, p=2, dim=1)
            
            # MSE loss
            loss = torch.mean((embed_distances - batch_real_dists) ** 2)
            
            # Backward propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Progress display (only on main process)
            if is_main and num_batches % 1000 == 0:
                print(f"    Progress: {num_batches}/{len(train_loader)} batches, current loss: {loss.item():.6f}")
        
        # Calculate average loss
        avg_loss = total_loss / max(num_batches, 1)
        
        # Sync loss across all processes during distributed training
        if is_distributed:
            avg_loss_tensor = torch.tensor(avg_loss, device=device)
            dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.AVG)
            avg_loss = avg_loss_tensor.item()
        
        total_loss_all_epochs += avg_loss
        
        if is_main:
            print(f"  [Epoch {epoch+1}/{epochs}] Average loss: {avg_loss:.6f}, batches: {num_batches}")
    
    final_avg_loss = total_loss_all_epochs / epochs
    if is_main:
        print(f"\n[Complete] hop{hop_level} training complete")
        print(f"  Average loss: {final_avg_loss:.6f}")
    
    # Release resources
    del train_dataset, train_loader
    if device.type == 'cuda':
        torch.cuda.empty_cache()


def contrastive_train(model, landmarks, dataset, epochs=3, batch_size=128, device="cpu", lr=0.001, margin=1.0):
    """
    Contrastive learning training: Optimize model using Triplet Loss
    
    Training strategy:
    - fc1: Positive samples (landmark-hop1), Negative samples (landmark-hop3)
    - fc2: Positive samples (landmark-hop2), Negative samples (landmark-hop3)
    - fc3: Positive samples (landmark-hop3), Negative samples (landmark-hop1)
    
    Triplet Loss = max(0, d(anchor, positive) - d(anchor, negative) + margin)
    
    Args:
        model: SiameseNetwork model
        landmarks: Landmark list containing hop1/hop2/hop3 neighbor vectors
        dataset: TrainingDataset object
        epochs: Training epochs
        batch_size: Batch size
        device: Device
        lr: Learning rate
        margin: Triplet Loss margin value
    """
    
    print("\n" + "="*60)
    print("[Contrastive Learning] Contrastive learning training using Triplet Loss")
    print("="*60)
    print(f"[Setup] batch size: {batch_size}, lr: {lr}, margin: {margin}, epochs: {epochs}")
    
    # Create separate optimizers for three fc modules
    optimizer_fc1 = optim.Adam(model.fc1.parameters(), lr=lr)
    optimizer_fc2 = optim.Adam(model.fc2.parameters(), lr=lr)
    optimizer_fc3 = optim.Adam(model.fc3.parameters(), lr=lr)
    
    total_loss_all_epochs = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss_fc1 = 0
        total_loss_fc2 = 0
        total_loss_fc3 = 0
        num_batches_fc1 = 0
        num_batches_fc2 = 0
        num_batches_fc3 = 0
        
        # Iterate over each landmark
        for landmark_idx, landmark in enumerate(landmarks):
            landmark_vec = landmark['vector']  # Directly the vector
            hop1_vecs = landmark['hop1']
            hop2_vecs = landmark['hop2']
            hop3_vecs = landmark['hop3']
            
            landmark_tensor = torch.from_numpy(landmark_vec).to(device).unsqueeze(0)
            
            # ========== Train fc1: Positive samples (hop1), Negative samples (hop3) ==========
            if len(hop1_vecs) > 0 and len(hop3_vecs) > 0:
                num_pos_samples = len(hop1_vecs)
                num_neg_samples = len(hop3_vecs)
                
                for batch_start in range(0, max(num_pos_samples, num_neg_samples), batch_size):
                    batch_end = min(batch_start + batch_size, max(num_pos_samples, num_neg_samples))
                    actual_batch_size = batch_end - batch_start
                    
                    # Sample positive samples (hop1)
                    pos_indices = np.random.choice(num_pos_samples, size=actual_batch_size, replace=True)
                    pos_batch = np.array([hop1_vecs[i] for i in pos_indices], dtype=np.float32)
                    
                    # Sample negative samples (hop3)
                    neg_indices = np.random.choice(num_neg_samples, size=actual_batch_size, replace=True)
                    neg_batch = np.array([hop3_vecs[i] for i in neg_indices], dtype=np.float32)
                    
                    # Convert to tensors
                    landmark_repeat = np.repeat(landmark_vec[np.newaxis, :], actual_batch_size, axis=0)
                    landmark_tensor_batch = torch.from_numpy(landmark_repeat).to(device)
                    pos_tensor = torch.from_numpy(pos_batch).to(device)
                    neg_tensor = torch.from_numpy(neg_batch).to(device)
                    
                    # Compute embeddings
                    landmark_embed = model.forward_fc1(landmark_tensor_batch)
                    pos_embed = model.forward_fc1(pos_tensor)
                    neg_embed = model.forward_fc1(neg_tensor)
                    
                    # Calculate distances
                    pos_distances = torch.norm(landmark_embed - pos_embed, p=2, dim=1)
                    neg_distances = torch.norm(landmark_embed - neg_embed, p=2, dim=1)
                    
                    # Triplet Loss
                    triplet_loss = torch.mean(torch.clamp(pos_distances - neg_distances + margin, min=0.0))
                    
                    # Backward propagation
                    optimizer_fc1.zero_grad()
                    triplet_loss.backward()
                    optimizer_fc1.step()
                    
                    total_loss_fc1 += triplet_loss.item()
                    num_batches_fc1 += 1
            
            # ========== Train fc2: Positive samples (hop2), Negative samples (hop3) ==========
            if len(hop2_vecs) > 0 and len(hop3_vecs) > 0:
                num_pos_samples = len(hop2_vecs)
                num_neg_samples = len(hop3_vecs)
                
                for batch_start in range(0, max(num_pos_samples, num_neg_samples), batch_size):
                    batch_end = min(batch_start + batch_size, max(num_pos_samples, num_neg_samples))
                    actual_batch_size = batch_end - batch_start
                    
                    # Sample positive samples (hop2)
                    pos_indices = np.random.choice(num_pos_samples, size=actual_batch_size, replace=True)
                    pos_batch = np.array([hop2_vecs[i] for i in pos_indices], dtype=np.float32)
                    
                    # Sample negative samples (hop3)
                    neg_indices = np.random.choice(num_neg_samples, size=actual_batch_size, replace=True)
                    neg_batch = np.array([hop3_vecs[i] for i in neg_indices], dtype=np.float32)
                    
                    # Convert to tensors
                    landmark_repeat = np.repeat(landmark_vec[np.newaxis, :], actual_batch_size, axis=0)
                    landmark_tensor_batch = torch.from_numpy(landmark_repeat).to(device)
                    pos_tensor = torch.from_numpy(pos_batch).to(device)
                    neg_tensor = torch.from_numpy(neg_batch).to(device)
                    
                    # Compute embeddings
                    landmark_embed = model.forward_fc2(landmark_tensor_batch)
                    pos_embed = model.forward_fc2(pos_tensor)
                    neg_embed = model.forward_fc2(neg_tensor)
                    
                    # Calculate distances
                    pos_distances = torch.norm(landmark_embed - pos_embed, p=2, dim=1)
                    neg_distances = torch.norm(landmark_embed - neg_embed, p=2, dim=1)
                    
                    # Triplet Loss
                    triplet_loss = torch.mean(torch.clamp(pos_distances - neg_distances + margin, min=0.0))
                    
                    # Backward propagation
                    optimizer_fc2.zero_grad()
                    triplet_loss.backward()
                    optimizer_fc2.step()
                    
                    total_loss_fc2 += triplet_loss.item()
                    num_batches_fc2 += 1
            
            # ========== Train fc3: Positive samples (hop3), Negative samples (hop1) ==========
            if len(hop3_vecs) > 0 and len(hop1_vecs) > 0:
                num_pos_samples = len(hop3_vecs)
                num_neg_samples = len(hop1_vecs)
                
                for batch_start in range(0, max(num_pos_samples, num_neg_samples), batch_size):
                    batch_end = min(batch_start + batch_size, max(num_pos_samples, num_neg_samples))
                    actual_batch_size = batch_end - batch_start
                    
                    # Sample positive samples (hop3)
                    pos_indices = np.random.choice(num_pos_samples, size=actual_batch_size, replace=True)
                    pos_batch = np.array([hop3_vecs[i] for i in pos_indices], dtype=np.float32)
                    
                    # Sample negative samples (hop1)
                    neg_indices = np.random.choice(num_neg_samples, size=actual_batch_size, replace=True)
                    neg_batch = np.array([hop1_vecs[i] for i in neg_indices], dtype=np.float32)
                    
                    # Convert to tensors
                    landmark_repeat = np.repeat(landmark_vec[np.newaxis, :], actual_batch_size, axis=0)
                    landmark_tensor_batch = torch.from_numpy(landmark_repeat).to(device)
                    pos_tensor = torch.from_numpy(pos_batch).to(device)
                    neg_tensor = torch.from_numpy(neg_batch).to(device)
                    
                    # Compute embeddings
                    landmark_embed = model.forward_fc3(landmark_tensor_batch)
                    pos_embed = model.forward_fc3(pos_tensor)
                    neg_embed = model.forward_fc3(neg_tensor)
                    
                    # Calculate distances
                    pos_distances = torch.norm(landmark_embed - pos_embed, p=2, dim=1)
                    neg_distances = torch.norm(landmark_embed - neg_embed, p=2, dim=1)
                    
                    # Triplet Loss
                    triplet_loss = torch.mean(torch.clamp(pos_distances - neg_distances + margin, min=0.0))
                    
                    # Backward propagation
                    optimizer_fc3.zero_grad()
                    triplet_loss.backward()
                    optimizer_fc3.step()
                    
                    total_loss_fc3 += triplet_loss.item()
                    num_batches_fc3 += 1
        
        # Calculate average loss for this epoch
        avg_loss_fc1 = total_loss_fc1 / max(num_batches_fc1, 1) if num_batches_fc1 > 0 else 0
        avg_loss_fc2 = total_loss_fc2 / max(num_batches_fc2, 1) if num_batches_fc2 > 0 else 0
        avg_loss_fc3 = total_loss_fc3 / max(num_batches_fc3, 1) if num_batches_fc3 > 0 else 0
        epoch_avg_loss = (avg_loss_fc1 + avg_loss_fc2 + avg_loss_fc3) / 3
        total_loss_all_epochs += epoch_avg_loss
        
        print(f"  [Epoch {epoch+1}/{epochs}]")
        print(f"    fc1 loss: {avg_loss_fc1:.6f} (batches: {num_batches_fc1})")
        print(f"    fc2 loss: {avg_loss_fc2:.6f} (batches: {num_batches_fc2})")
        print(f"    fc3 loss: {avg_loss_fc3:.6f} (batches: {num_batches_fc3})")
    
    final_avg_loss = total_loss_all_epochs / epochs
    print(f"\n[Complete] Contrastive learning training complete")
    print(f"  Average loss: {final_avg_loss:.6f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train_model_from_data.py <binary_data_file|--stdin> [output_model] [epochs] [batch_size] [lr]")
        print("\nExamples:")
        print("  python train_model_from_data.py training_dataset.bin siamese_model.pt 5 128 0.001")
        print("  python train_model_from_data.py --stdin siamese_model.pt 5 128 0.001  # Read data from pipe")
        sys.exit(1)
    
    data_source = sys.argv[1]
    from_stdin = (data_source == "--stdin")
    
    output_model = sys.argv[2] if len(sys.argv) > 2 else "siamese_model.pt"
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    batch_size = int(sys.argv[4]) if len(sys.argv) > 4 else 128
    lr = float(sys.argv[5]) if len(sys.argv) > 5 else 0.001
    
    print("\n" + "="*60)
    print("HNSW SiameseNetwork Training Program")
    print("="*60)
    print(f"Data source: {'Standard input (pipe)' if from_stdin else data_source}")
    print(f"Output model: {output_model}")
    print(f"Training params: epochs={epochs}, batch_size={batch_size}, lr={lr}")
    print("="*60 + "\n")
    
    train_model(data_source, output_model, epochs=epochs, batch_size=batch_size, lr=lr, from_stdin=from_stdin)

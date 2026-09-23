import os
import pickle
import torch
import glob
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import esm
import time
from pathlib import Path
import argparse
import sys
import gc

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

class ProteinSequenceDataset(Dataset):
    
    def __init__(self, proteins):
        self.ids = []
        self.sequences = []
        
        self.max_seq_len = 1500  
        for prot in proteins:
            if len(prot['sequence']) <= self.max_seq_len:
                self.ids.append(prot['uniprot_id'])
                self.sequences.append(prot['sequence'])
            else:
                print("Skipping {} (length: {}) - exceeds max length".format(
                    prot['uniprot_id'], len(prot['sequence'])))
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.ids[idx], self.sequences[idx]

def load_filtered_proteins(data_dir=None):
    
    if data_dir is None:
        data_dir = os.environ.get('PROTEIN_DATA_DIR', '/homeb/ali/second/data')
    
    from Bio import SeqIO
    
    fasta_files = {
        0: os.path.join(data_dir, "non-NABP.fasta"),
        1: os.path.join(data_dir, "RBP.fasta"),
        2: os.path.join(data_dir, "DBP.fasta"),
    }
    
    all_proteins = []
    for label, fasta_path in fasta_files.items():
        if os.path.exists(fasta_path):
            sequences = list(SeqIO.parse(fasta_path, "fasta"))
            print("Loading {} from {}".format(len(sequences), Path(fasta_path).name))
            
            for record in sequences:
                all_proteins.append({
                    'uniprot_id': record.id,
                    'sequence': str(record.seq),
                    'label': label
                })
        else:
            print("Warning: {} not found".format(fasta_path))
    
    return all_proteins

def clear_gpu_memory():
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()

def generate_embeddings_with_recovery(
    proteins,
    output_dir='/homeb/ali/second/data/embeddings',
    batch_size=1,
    device='cuda',
    gpu_id=1, 
    save_interval=500,
    model_path=None
):
   
    if device == 'cuda' and torch.cuda.is_available():
       
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
       
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
    
    os.makedirs(output_dir, exist_ok=True)
    
    final_file = os.path.join(output_dir, 'esm_embeddings_final_gpu{}.pkl'.format(gpu_id))
    checkpoint_dir = os.path.join(output_dir, 'checkpoints_gpu{}'.format(gpu_id))
    log_file = os.path.join(output_dir, 'generation_log_gpu{}.txt'.format(gpu_id))
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    def log_message(msg):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_msg = "[{}] [GPU {}] {}".format(timestamp, gpu_id, msg)
        print(log_msg)
        with open(log_file, 'a') as f:
            f.write(log_msg + '\n')
    
    log_message("Starting embedding generation on GPU server")
    log_message("PyTorch version: {}".format(torch.__version__))
    log_message("CUDA available: {}".format(torch.cuda.is_available()))
    
    clear_gpu_memory()
    
    if torch.cuda.is_available():
        visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
        log_message("CUDA_VISIBLE_DEVICES: {}".format(visible_devices))
        log_message("Number of visible GPUs: {}".format(torch.cuda.device_count()))
        
        if torch.cuda.device_count() > 0:
            device_name = torch.cuda.get_device_name(0)
            log_message("Using GPU: {} (original ID: {})".format(device_name, gpu_id))
            log_message("CUDA total memory: {:.2f} GB".format(
                torch.cuda.get_device_properties(0).total_memory / 1e9))
    
    log_message("Loading ESM-2 model...")
    original_load = torch.load
    
    def patched_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return original_load(*args, **kwargs)
    
    torch.load = patched_load
    
    try:
        if model_path is None:
            model_path = os.environ.get('ESM2_MODEL_PATH', 
                                       '/homeb/ali/second/esm2_t36_3B_UR50D/esm2_t36_3B_UR50D.pt')
        
        if os.path.exists(model_path):
            log_message("Loading model from: {}".format(model_path))
            model, alphabet = esm.pretrained.load_model_and_alphabet_local(model_path)
        else:
            log_message("Local model not found, downloading from hub...")
            model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
            
    finally:
        torch.load = original_load
    
    if device == 'cuda' and torch.cuda.is_available() and torch.cuda.device_count() > 0:
        device = torch.device('cuda:0')  # Now this refers to our selected GPU
        log_message("Using device: {}".format(device))
    else:
        device = torch.device('cpu')
        log_message("Using CPU (slower)")
    
    if device.type == 'cuda':
        model = model.half()
        log_message("Model converted to half precision")
    
    model = model.to(device)
    model.eval()
    
    clear_gpu_memory()
    
    if device.type == 'cuda':
        log_message("Model loaded. GPU memory used: {:.2f} GB".format(
            torch.cuda.memory_allocated(device) / 1e9))
    
    batch_converter = alphabet.get_batch_converter()
    
    processed_ids = set()
    
    existing_checkpoints = glob.glob(os.path.join(checkpoint_dir, '*.pkl'))
    for cp in existing_checkpoints:
        try:
            with open(cp, 'rb') as f:
                chunk = pickle.load(f)
                processed_ids.update(chunk.keys())
            log_message("Found checkpoint: {} with {} proteins".format(os.path.basename(cp), len(chunk)))
        except Exception as e:
            log_message("Error reading checkpoint {}: {}".format(cp, e))
    
    if os.path.exists(final_file):
        try:
            with open(final_file, 'rb') as f:
                existing = pickle.load(f)
                processed_ids.update(existing.keys())
            log_message("Found final file with {} proteins".format(len(existing)))
        except Exception as e:
            log_message("Error reading final file: {}".format(e))
    
    log_message("Already processed: {} proteins".format(len(processed_ids)))
    
    unprocessed = [p for p in proteins if p['uniprot_id'] not in processed_ids]
    log_message("Remaining to process: {}".format(len(unprocessed)))
    
    if not unprocessed:
        log_message("All proteins already processed!")
        return
    
    dataset = ProteinSequenceDataset(unprocessed)
    log_message("After filtering by length: {} proteins".format(len(dataset)))
    
    num_workers = 0
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers, pin_memory=False)
    
    current_chunk = {}
    chunk_counter = len(existing_checkpoints)
    
    start_time = time.time()
    processed_count = 0
    
    with torch.no_grad():
        for idx, (batch_ids, batch_seqs) in enumerate(tqdm(dataloader, desc="Generating embeddings")):
            try:
               
                batch_data = list(zip(batch_ids, batch_seqs))
                batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
                batch_tokens = batch_tokens.to(device, non_blocking=False)
                
                results = model(batch_tokens, repr_layers=[36])
                token_representations = results["representations"][36]
                
                for i, (label, seq) in enumerate(batch_data):
                    seq_len = len(seq)
                    embedding = token_representations[i, 1:seq_len+1].cpu()
                    
                    if embedding.dtype == torch.float16:
                        embedding = embedding.float()
                    
                    current_chunk[label] = embedding
                    processed_count += 1
                
                del results, token_representations, batch_tokens
                
                if (idx + 1) % save_interval == 0:
                    chunk_file = os.path.join(checkpoint_dir, 'chunk_{:04d}.pkl'.format(chunk_counter))
                    
                    temp_file = chunk_file + '.tmp'
                    with open(temp_file, 'wb') as f:
                        pickle.dump(current_chunk, f)
                    
                    try:
                        with open(temp_file, 'rb') as f:
                            test = pickle.load(f)
                        if len(test) == len(current_chunk):
                            os.replace(temp_file, chunk_file)
                            elapsed = time.time() - start_time
                            speed = processed_count / elapsed
                            log_message("Saved chunk {}: {} proteins".format(chunk_counter, len(current_chunk)))
                            log_message("Speed: {:.2f} proteins/second".format(speed))
                            chunk_counter += 1
                            current_chunk = {}
                            clear_gpu_memory()
                    except Exception as e:
                        log_message("Error saving chunk: {}".format(e))
                
                if (idx + 1) % 50 == 0:
                    clear_gpu_memory()
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    log_message("CUDA OOM at batch {}. Attempting to recover...".format(idx))
                    clear_gpu_memory()
                    
                    if batch_size > 1:
                        new_batch_size = max(1, batch_size // 2)
                        log_message("Reducing batch size from {} to {}".format(batch_size, new_batch_size))
                        batch_size = new_batch_size
                        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                                               num_workers=num_workers, pin_memory=False)
                    continue
                else:
                    log_message("Error processing batch {}: {}".format(idx, e))
                    continue
    
    if current_chunk:
        chunk_file = os.path.join(checkpoint_dir, 'chunk_{:04d}.pkl'.format(chunk_counter))
        with open(chunk_file, 'wb') as f:
            pickle.dump(current_chunk, f)
        log_message("Saved final chunk: {} proteins".format(len(current_chunk)))
    
    log_message("\n" + "="*50)
    log_message("MERGING ALL CHUNKS")
    log_message("="*50)
    
    all_embeddings = {}
    chunk_files = sorted(glob.glob(os.path.join(checkpoint_dir, '*.pkl')))
    
    for cf in tqdm(chunk_files, desc="Merging"):
        try:
            with open(cf, 'rb') as f:
                chunk = pickle.load(f)
                all_embeddings.update(chunk)
        except Exception as e:
            log_message("Error loading {}: {}".format(os.path.basename(cf), e))
    
    log_message("Total embeddings: {}".format(len(all_embeddings)))
    
    try:
        log_message("Saving final file...")
        temp_final = final_file + '.tmp'
        with open(temp_final, 'wb') as f:
            pickle.dump(all_embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open(temp_final, 'rb') as f:
            test = pickle.load(f)
        
        if len(test) == len(all_embeddings):
            os.replace(temp_final, final_file)
            file_size = os.path.getsize(final_file) / (1024**3)
            log_message("SUCCESS! Final file saved: {}".format(final_file))
            log_message("  Size: {:.2f} GB".format(file_size))
            log_message("  Embeddings: {}".format(len(all_embeddings)))
    except Exception as e:
        log_message("Save failed: {}".format(e))
    
    total_time = time.time() - start_time
    log_message("\nTotal processing time: {:.2f} hours".format(total_time/3600))
    log_message("Average speed: {:.2f} proteins/second".format(processed_count/total_time))
    
    return all_embeddings

def main():
    parser = argparse.ArgumentParser(description='Generate ESM-2 embeddings on GPU server')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Directory containing FASTA files')
    parser.add_argument('--output_dir', type=str, default='/homeb/ali/second/data/embeddings',
                       help='Output directory for embeddings')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to local ESM-2 model file')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for GPU processing')
    parser.add_argument('--save_interval', type=int, default=500,
                       help='Save checkpoint every N proteins')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--gpu_id', type=int, default=1,  # Default to GPU 1
                       help='GPU ID to use (0-5 for your system)')
    
    args = parser.parse_args()
    
    print("\nStep 1: Loading proteins...")
    proteins = load_filtered_proteins(data_dir=args.data_dir)
    print("Loaded {} total proteins".format(len(proteins)))
    
    print("\nStep 2: Generating embeddings...")
    embeddings = generate_embeddings_with_recovery(
        proteins=proteins,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=args.device,
        gpu_id=args.gpu_id,  
        save_interval=args.save_interval,
        model_path=args.model_path
    )
    
    print("\n" + "="*80)
    print("PROCESS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
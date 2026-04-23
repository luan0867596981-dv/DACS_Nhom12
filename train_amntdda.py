import torch
import torch.nn as nn
import torch.optim as optim
import os
import csv
import json
import time
import torch_geometric.transforms as T
from sklearn.metrics import roc_auc_score
import argparse

from config import config
from preprocess import preprocess
from models.amntdda_model import AMNTDDA, _DATASET_DIMS
from models.contrastive_loss import GraphContrastiveLearning
from utils.metrics import compute_metrics, find_best_threshold

PROGRESS_FILE = 'training_progress_amntdda.json'

def generate_bpr_k_negative_edges_strict(pos_edge_index, num_nodes_drug, num_nodes_disease, K=5):
    device = pos_edge_index.device
    num_pos = pos_edge_index.size(1)
    
    dense_mask = torch.ones((num_nodes_drug, num_nodes_disease), dtype=torch.float, device=device)
    dense_mask[pos_edge_index[0], pos_edge_index[1]] = 0.0
    
    weights = dense_mask[pos_edge_index[0]]
    neg_dst_matrix = torch.multinomial(weights, num_samples=K, replacement=False)
    neg_dst = neg_dst_matrix.flatten()
    neg_src = pos_edge_index[0].repeat_interleave(K)
    
    return torch.stack([neg_src, neg_dst], dim=0)

def bpr_loss_k_fn(pos_scores, neg_scores, K):
    neg_matrix = neg_scores.view(-1, K)
    diff = pos_scores.unsqueeze(1) - neg_matrix
    return -torch.mean(torch.nn.functional.logsigmoid(diff))

def create_directories():
    os.makedirs(os.path.join(config.root_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(config.root_dir, 'checkpoints', 'AMNTDDA'), exist_ok=True)
    os.makedirs(os.path.join(config.root_dir, 'results', 'tables'), exist_ok=True)

def train_fold(dataset_name, fold, epochs, device):
    print(f"\n{'='*50}\nStarting Fold {fold}/10 for Dataset {dataset_name}\n{'='*50}")
    
    config.dataset_name = dataset_name
    config.epochs = epochs
    
    log_file_path = os.path.join(config.root_dir, 'logs', f'train_log_amntdda_{dataset_name}_fold{fold}.csv')
    with open(log_file_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'lr', 'total_loss', 'task_loss', 'contrastive_loss', 'val_auc'])

    data, drug_sim_emb, disease_sim_emb = preprocess()
    
    torch.manual_seed(fold)
    
    print("Generating Deterministic RandomLinkSplits to prevent Data Leakage...")
    transform = T.RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        is_undirected=True,
        neg_sampling_ratio=1.0, 
        add_negative_train_samples=False,
        edge_types=[('drug', 'treats', 'disease')],
        rev_edge_types=[('disease', 'treated_by', 'drug')]
    )
    train_data, val_data, test_data = transform(data)
    
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)
    drug_sim = drug_sim_emb.to(device)
    disease_sim = disease_sim_emb.to(device)
    
    num_drugs = data['drug'].x.size(0)
    num_diseases = data['disease'].x.size(0)

    # Initialize AMNTDDA 
    model = AMNTDDA(num_drugs=num_drugs, num_diseases=num_diseases).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    # Configure Learning Rate Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=config.lr_factor, patience=config.lr_patience
    )
    
    bce_loss_fn = nn.BCEWithLogitsLoss()
    gcl = GraphContrastiveLearning(temperature=config.temperature).to(device)

    dd_edge_type = ('drug', 'treats', 'disease')
    pos_edge_index = train_data[dd_edge_type].edge_label_index

    val_edge_index = val_data[dd_edge_type].edge_label_index
    val_labels = val_data[dd_edge_type].edge_label

    best_val_auc = 0.0
    best_val_aupr = 0.0
    best_epoch = 0
    best_val_threshold = 0.5
    patience_counter = 0

    fold_checkpoint_name = f'temp_amntdda_{dataset_name}_fold{fold}.pt'
    fold_checkpoint_path = os.path.join(config.root_dir, 'checkpoints', 'AMNTDDA', fold_checkpoint_name)
    
    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad()
        
        K_NEG = 5 
        neg_edge_index = generate_bpr_k_negative_edges_strict(pos_edge_index, num_drugs, num_diseases, K=K_NEG).to(device)
        train_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        
        num_pos = pos_edge_index.size(1)
        num_neg = neg_edge_index.size(1) 
        labels = torch.cat([torch.ones(num_pos), torch.zeros(num_neg)]).to(device)
        
        # AMNTDDA Forward Pass
        preds, (drug_emb, disease_emb) = model(
            drug_sim, disease_sim,
            train_edge_index[0], train_edge_index[1],
            use_transformers=False, # Match inference fix
            return_embs=True
        )
        
        # Task Loss (BCE)
        task_loss_bce = nn.BCELoss()(preds, labels)
        
        # Task Loss (BPR)
        pos_preds = preds[:num_pos]
        neg_preds = preds[num_pos:]
        pos_logits = torch.log(pos_preds.clamp(min=1e-8) / (1 - pos_preds.clamp(max=1-1e-8)))
        neg_logits = torch.log(neg_preds.clamp(min=1e-8) / (1 - neg_preds.clamp(max=1-1e-8)))
        
        loss_bpr = bpr_loss_k_fn(pos_logits, neg_logits, K_NEG)
        task_loss = task_loss_bce + (0.1 * loss_bpr)
        
        # Simple Contrastive Learning applying dropout to similarity matrices as augmentation
        v1_drug_sim = torch.nn.functional.dropout(drug_sim, p=config.feat_mask_prob)
        v1_disease_sim = torch.nn.functional.dropout(disease_sim, p=config.feat_mask_prob)
        v2_drug_sim = torch.nn.functional.dropout(drug_sim, p=config.feat_mask_prob)
        v2_disease_sim = torch.nn.functional.dropout(disease_sim, p=config.feat_mask_prob)
        
        _, (z1_drug, z1_disease) = model(v1_drug_sim, v1_disease_sim, train_edge_index[0], train_edge_index[1], use_transformers=False, return_embs=True)
        _, (z2_drug, z2_disease) = model(v2_drug_sim, v2_disease_sim, train_edge_index[0], train_edge_index[1], use_transformers=False, return_embs=True)
        
        loss_cl_drug = gcl.info_nce_loss(z1_drug, z2_drug)
        loss_cl_disease = gcl.info_nce_loss(z1_disease, z2_disease)
        contrast_loss = loss_cl_drug + loss_cl_disease
        
        total_loss = task_loss + config.contrast_weight * contrast_loss
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = model(drug_sim, disease_sim, val_edge_index[0], val_edge_index[1], use_transformers=False)
            val_preds_prob = val_preds.cpu().numpy()
            v_labels_np = val_labels.cpu().numpy()
            
            current_val_threshold = find_best_threshold(v_labels_np, val_preds_prob, metric='f1')
            val_metrics = compute_metrics(v_labels_np, val_preds_prob, drug_ids=val_edge_index[0].cpu().numpy(), threshold=current_val_threshold)
            current_val_auc = val_metrics["AUC"]
            current_val_aupr = val_metrics["AUPR"]
            
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(current_val_auc)
        
        with open(log_file_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, current_lr, total_loss.item(), task_loss.item(), contrast_loss.item(), current_val_auc])

        if epoch % 10 == 0 or epoch == config.epochs - 1:
            print(f"Fold {fold} | Ep {epoch:03d} | LR {current_lr:.5f} | Val AUC: {current_val_auc:.4f} | Val AUPR: {current_val_aupr:.4f} | Loss: {total_loss.item():.4f}")

        if current_val_auc > best_val_auc:
            best_val_auc = current_val_auc
            best_val_aupr = current_val_aupr
            best_epoch = epoch
            best_val_threshold = current_val_threshold
            patience_counter = 0
            torch.save(model.state_dict(), fold_checkpoint_path)
        else:
            patience_counter += 1
            
        if patience_counter >= config.early_stopping_patience:
            print(f"Early stopping at epoch {epoch}")
            break
            
    print("\nEvaluating Best Model on Test Set...")
    if os.path.exists(fold_checkpoint_path):
        model.load_state_dict(torch.load(fold_checkpoint_path, map_location=device, weights_only=False))
        
    model.eval()
    test_edge_index = test_data[dd_edge_type].edge_label_index
    test_labels = test_data[dd_edge_type].edge_label

    with torch.no_grad():
        test_preds = model(drug_sim, disease_sim, test_edge_index[0], test_edge_index[1], use_transformers=False)
        test_preds_prob = test_preds.cpu().numpy()
        test_labels_np = test_labels.cpu().numpy()
        test_drug_ids_np = test_edge_index[0].cpu().numpy()

        test_metrics_standard = compute_metrics(test_labels_np, test_preds_prob, drug_ids=test_drug_ids_np, threshold=best_val_threshold)

        pos_test_edges = test_edge_index[:, test_labels == 1]
        num_pos_test = pos_test_edges.size(1)
        if num_pos_test > 0:
            neg_ratio = 50
            neg_drug_coords = pos_test_edges[0].repeat_interleave(neg_ratio)
            neg_disease_coords = torch.randint(0, num_diseases, (num_pos_test * neg_ratio,), device=device)
            
            ranking_edge_index = torch.cat([pos_test_edges, torch.stack([neg_drug_coords, neg_disease_coords])], dim=1)
            ranking_labels = torch.cat([torch.ones(num_pos_test), torch.zeros(num_pos_test * neg_ratio)]).to(device)

            ranking_preds = model(drug_sim, disease_sim, ranking_edge_index[0], ranking_edge_index[1], use_transformers=False)
            ranking_preds_prob = ranking_preds.cpu().numpy()
            ranking_labels_np = ranking_labels.cpu().numpy()
            ranking_drug_ids_np = ranking_edge_index[0].cpu().numpy()

            ranking_metrics = compute_metrics(ranking_labels_np, ranking_preds_prob, drug_ids=ranking_drug_ids_np, threshold=best_val_threshold)
        else:
            ranking_metrics = {"Recall@10": 0.0, "Recall@20": 0.0, "Recall@50": 0.0}

    final_test_metrics = test_metrics_standard.copy()
    final_test_metrics["Recall@10"] = ranking_metrics["Recall@10"]
    final_test_metrics["Recall@20"] = ranking_metrics["Recall@20"]
    final_test_metrics["Recall@50"] = ranking_metrics["Recall@50"]
    
    print(f"\n--- Best Fold {fold} Metrics ({dataset_name}) ---")
    for k, v in final_test_metrics.items():
        print(f"{k}: {v:.4f}")
        
    return final_test_metrics, best_val_auc, best_epoch, best_val_threshold, fold_checkpoint_path

def main():
    parser = argparse.ArgumentParser(description='Train AMNTDDA on Multiple Datasets')
    parser.add_argument('--datasets', type=str, nargs='+', default=['C-dataset', 'B-dataset', 'F-dataset'], help='Datasets')
    parser.add_argument('--epochs', type=int, default=300, help='Max epochs')
    parser.add_argument('--folds', type=int, default=1, help='Number of folds')
    args = parser.parse_args()

    # Assign from user args to config
    config.epochs = args.epochs

    create_directories()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Hardware initialization! Device: {device}")
    
    for ds in args.datasets:
        print(f"\n{'*'*55}")
        print(f"*** BEGIN TRAINING FOR DATASET: {ds} ***")
        print(f"{'*'*55}\n")
        
        fold_metrics = []
        best_dataset_val_auc = 0.0
        
        for fold in range(1, args.folds + 1):
            metrics, fold_best_val_auc, best_epoch, best_val_threshold, fold_checkpoint_path = train_fold(ds, fold, config.epochs, device)
            
            # Store metrics with best_epoch included
            metrics_with_epoch = {"Best_Epoch": best_epoch, "Best_Threshold": best_val_threshold}
            metrics_with_epoch.update(metrics)
            fold_metrics.append(metrics_with_epoch)
            
            ds_letter = ds[0]
            dataset_target_model_path = os.path.join(config.root_dir, 'results', 'result_train', ds, 'AMNTDDA')
            os.makedirs(dataset_target_model_path, exist_ok=True)
            
            # --- OVERWRITE MODEL KHI GHI NHẬN AUC TỐT HƠN ---
            dataset_best_model_path = os.path.join(dataset_target_model_path, f'{ds_letter}-model.pt')
            
            if fold_best_val_auc > best_dataset_val_auc:
                print(f"-> Selected Fold {fold} as BEST for {ds} with Val AUC: {fold_best_val_auc:.4f}")
                best_dataset_val_auc = fold_best_val_auc
                if os.path.exists(fold_checkpoint_path):
                    import shutil
                    shutil.copy2(fold_checkpoint_path, dataset_best_model_path)
                    print(f"-> Saved newly trained {ds_letter}-model.pt (422 parameters)")
            
            if os.path.exists(fold_checkpoint_path):
                os.remove(fold_checkpoint_path)

        # Calculate Mean and Std
        import numpy as np
        metric_names = list(fold_metrics[0].keys())
        mean_metrics = {}
        std_metrics = {}
        
        for metric_name in metric_names:
            values = [fm[metric_name] for fm in fold_metrics]
            mean_metrics[metric_name] = np.mean(values)
            std_metrics[metric_name] = np.std(values)
        
        print(f"\n--- SUMMARY AVERAGE {args.folds}-FOLD FOR {ds} ---")
        for k, v in mean_metrics.items():
            print(f"{k}: {v:.4f} ± {std_metrics[k]:.4f}")
            
        # Write format: Fold/Statistics, Best_Epoch, AUC, AUPR, ...
        agg_path = os.path.join(config.root_dir, 'results', 'tables', f'10_fold_results_{ds}.csv')
        with open(agg_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            header = ['Fold/Statistics'] + metric_names
            writer.writerow(header)
            
            # Write individual folds
            for i, fm in enumerate(fold_metrics):
                row = [f'Fold {i+1}'] + [f"{fm[k]:.4f}" for k in metric_names]
                writer.writerow(row)
                
            # Write Mean
            mean_row = ['Mean'] + [f"{mean_metrics[k]:.4f}" for k in metric_names]
            writer.writerow(mean_row)
            
            # Write Std
            std_row = ['Std'] + [f"{std_metrics[k]:.4f}" for k in metric_names]
            writer.writerow(std_row)

    print("\nDONE! Pipeline AMNTDDA Training completely upgraded.")

if __name__ == '__main__':
    main()

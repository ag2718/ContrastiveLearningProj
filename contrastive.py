import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np

from tree import Tree
from data import load_data
from evals import evaluate_model, evaluate_embeddings

import argparse
from tqdm import tqdm

class_tree = None
distance_matrix = None

# use 'conda install pytorch torchvision -c pytorch' in python env to import; if using Colab probably use pip 

def precompute_lca_distances(labels, contrastive_classes, contrastive_class_to_id, class_tree):
    print (labels.shape)
    num_labels = labels.shape[0]
    lca_matrix = torch.zeros((num_labels, num_labels), dtype=torch.long)
    distance_matrix = torch.zeros((num_labels, num_labels),  dtype=torch.float)

    # Precompute LCAs and distances
    for i in range(num_labels):
        for j in range(i + 1, num_labels):

            class_i = contrastive_classes[labels[i].item()]
            class_j = contrastive_classes[labels[j].item()]
            lca = class_tree.find_lca(class_i, class_j)
            distance_i = class_tree.find_distance_to_ancestor(class_i, lca)
            distance_j = class_tree.find_distance_to_ancestor(class_j, lca)
            min_distance = min(distance_i, distance_j)
            lca_matrix[i, j] = lca
            lca_matrix[j, i] = lca
            distance_matrix[i, j] = min_distance
            distance_matrix[j, i] = min_distance

    return distance_matrix

# Basic Contrastive Learning Model
class ContrastiveModel(nn.Module):
    def __init__(self, num_classes, embedding_dim=128):
        super(ContrastiveModel, self).__init__()
        # Using ResNet backbone since we are using ImageNet and therefore compatible, feel free to change
        self.backbone = models.resnet18(pretrained=True)

        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.projection_head = nn.Sequential(
            nn.Linear(num_features, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

        # Classification head
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)  # Extract features
        embeddings = self.projection_head(features)  # Project embeddings
        logits = self.classifier(embeddings)  # Classification logits
        return embeddings, logits


# Define the NCE Loss
class NCELoss(nn.Module):
    def __init__(self, temperature=0.07, dist_func_param=1):
        super(NCELoss, self).__init__()
        self.temperature = temperature
        self.dist_func_param = dist_func_param

    def forward(self, embeddings, labels):
        global distance_matrix
        # Normalize embeddings to unit vectors
        embeddings = nn.functional.normalize(embeddings, dim=1)

        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Mask out self-similarity
        mask = torch.eye(similarity_matrix.size(0), device=similarity_matrix.device).bool()
        similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))

        distances = torch.exp(-self.dist_func_param * distance_matrix[labels][:, labels])

        # Compute log-softmax and NCE loss
        log_prob = nn.functional.log_softmax(similarity_matrix, dim=1)
        loss_matrix = log_prob * distances

        loss_matrix.fill_diagonal_(0)

        #loss = -torch.sum(log_prob * positives) / labels.sum() #loss is only how far apart the positives are
        loss = -torch.sum(loss_matrix) / loss_matrix.shape[0] # / labels.sum() #loss is only how far apart the positives are
        return loss

# Example Usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--contr_specificity', type=int, default=3)
    parser.add_argument('-l', '--class_specificity', type=int, default=2)
    parser.add_argument('-n', '--num_epochs', default=10) 
    parser.add_argument('-t', '--normalization_term', type=float, default=1.0)
  
    args = parser.parse_args()
    print(args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataloader, val_dataloader, dogs, num_total_classes = load_data()
    class_tree = Tree(dogs)

    contrastive_classes_specificity = args.contr_specificity
    contrastive_classes = class_tree.nodes_at_depth(contrastive_classes_specificity)
    contrastive_class_to_id = {_cls: i for i, _cls in enumerate(contrastive_classes)}

    clf_classes_specificity = args.class_specificity
    clf_classes = class_tree.nodes_at_depth(clf_classes_specificity)
    clf_class_to_id = {_cls: i for i, _cls in enumerate(clf_classes)}

    num_classes = len(clf_classes)

    model = ContrastiveModel(num_classes).cuda()
    nce_loss_fn = NCELoss().cuda() # Generally probably add all of this to Colab to use the GPU
    classification_loss_fn = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Precompute distance matrix
    distance_matrix = torch.tensor(precompute_lca_distances(np.arange(len(contrastive_classes)), contrastive_classes, contrastive_class_to_id, class_tree), device=device)

    for epoch in range(args.num_epochs):  # Using two epochs to test, adjust as needed
        model.train()
        running_loss = 0.0
        total_correct = 0
        total_samples = 0
        num_batches = 0
        all_predictions = []
        all_labels = []

        for i, batch in enumerate(tqdm(train_dataloader)):
            images, labels = batch
            images, labels = images.cuda(), labels.cuda()

            # Map labels to contrastive level
            contr_labels = torch.tensor([class_tree.which_ancestor(label.item(), contrastive_classes) for label in labels], device='cuda')
            contr_labels = torch.tensor([contrastive_class_to_id[contr_label.item()] for contr_label in contr_labels], device='cuda')

            # Map labels to classification level
            clf_labels = torch.tensor([class_tree.which_ancestor(label.item(), clf_classes) for label in labels], device='cuda')
            clf_labels = torch.tensor([clf_class_to_id[clf_label.item()] for clf_label in clf_labels], device='cuda')

            # Forward pass
            embeddings, logits = model(images)

            # Compute losses
            nce_loss = nce_loss_fn(embeddings, contr_labels) * len(contr_labels) / 75 * args.normalization_term
            classification_loss = classification_loss_fn(logits, clf_labels)
            total_loss = nce_loss + classification_loss

            # Backward pass and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Update running loss
            running_loss += total_loss.item()

            # Compute accuracy
            _, predictions = torch.max(logits, dim=1)  # Get class predictions
            total_correct += (predictions == clf_labels).sum().item()  # Count correct predictions
            total_samples += clf_labels.size(0)  # Update total samples
            # print(f"Batch accuracy: {total_correct / total_samples:.4f}")
            num_batches += 1

            all_predictions.append(predictions)
            all_labels.append(labels)
            

        # Compute average loss and accuracy
        epoch_loss = running_loss / num_batches
        train_accuracy = total_correct / total_samples
        
        # Validation phase
        val_loss, val_accuracy, val_precision, val_recall, val_f1, val_conf_matrix = evaluate_model(
            model, val_dataloader, classification_loss_fn, device, num_classes, class_tree, clf_classes, clf_class_to_id
        )

        # Contrastive stats (diameter and margin)
        diameter, margin = evaluate_embeddings(model, val_dataloader, device)

        # Log epoch statistics
        print(f"Epoch [{epoch + 1}/{args.num_epochs}]")
        print(f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
        print(f"Validation Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1-Score: {val_f1:.4f}")
        print(f"Diameter: {diameter:.4f}, Margin: {margin:.4f}")
        print("-" * 50)

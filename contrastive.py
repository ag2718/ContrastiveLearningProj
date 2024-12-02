import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np

from tree import Tree
from data import load_data

class_tree = None
distance_matrix = None

# use 'conda install pytorch torchvision -c pytorch' in python env to import; if using Colab probably use pip 

def precompute_lca_distances(labels, class_tree):
    num_labels = len(labels)
    lca_matrix = torch.zeros((num_labels, num_labels), dtype=torch.long)
    distance_matrix = torch.zeros((num_labels, num_labels), dtype=torch.float)

    # Precompute LCAs and distances
    for i in range(num_labels):
        for j in range(i + 1, num_labels):
            lca = class_tree.find_lca(labels[i].item(), labels[j].item())
            distance_i = class_tree.find_distance_to_ancestor(labels[i].item(), lca)
            distance_j = class_tree.find_distance_to_ancestor(labels[j].item(), lca)
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

        # Create targets: positive samples have the same label
        labels = labels.unsqueeze(0) == labels.unsqueeze(1)
        #positives = labels.float()
        distances = torch.exp(-self.dist_func_param * distance_matrix[labels][:, labels])

        # Compute log-softmax and NCE loss
        log_prob = nn.functional.log_softmax(similarity_matrix, dim=1)
        #loss = -torch.sum(log_prob * positives) / labels.sum() #loss is only how far apart the positives are
        loss = -torch.sum(log_prob * distances) / labels.sum() #loss is only how far apart the positives are
        return loss


# Example Usage
if __name__ == "__main__":
    # Sample dataset preparation, making flexible 
    # TODO: make as function of level of specificity 
    class SampleDataset(Dataset):
        def __init__(self, size=100, num_classes=10, transform=None):
            self.size = size
            self.num_classes = num_classes
            self.transform = transform
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            # Random RGB image and label
            image = torch.rand(3, 224, 224)
            label = idx % self.num_classes
            if self.transform:
                image = self.transform(image)
            return image, label

    '''
    # DataLoader setup
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = SampleDataset(size=100, num_classes=10, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Initialize model, loss, and optimizer
    num_classes = 10'''

    train_dataloader, val_dataloader, dogs, num_total_classes = load_data()
    print ("Dogs")
    print (dogs)
    print ("Num total classes")
    print (num_total_classes)
    class_tree = Tree(dogs)

    contrastive_classes_specificity = 2
    contrastive_classes = class_tree.nodes_at_depth(contrastive_classes_specificity)
    contrastive_class_to_id = {_cls: i for i, _cls in enumerate(contrastive_classes)}

    clf_classes_specificity = 1
    clf_classes = class_tree.nodes_at_depth(clf_classes_specificity)
    clf_class_to_id = {_cls: i for i, _cls in enumerate(clf_classes)}

    print ("Contrastive classes")
    print (contrastive_classes)
    print ("Clf classes")
    print (clf_classes)

    num_classes = len(clf_classes)

    model = ContrastiveModel(num_classes).cuda()
    nce_loss_fn = NCELoss().cuda() # Generally probably add all of this to Colab to use the GPU 
    classification_loss_fn = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Precompute distance matrix
    distance_matrix = torch.tensor(precompute_lca_distances(np.arange(len(contrastive_classes)), class_tree))

    print ("Distance matrix")
    print (distance_matrix[:10, :10])

    # Training loop
    # TODO: add in evals or call to eval.py file perhaps? 
    for epoch in range(2):  # Using two epochs to test but add more epochs
        model.train()
        running_loss = 0.0

        for batch in train_dataloader:
            images, labels = batch
            print ("Batch")
            print (images.shape, labels.shape)
            print (labels)
            images, labels = images.cuda(), labels.cuda()

            # for each image, find the label at the correct classification level
            labels = [class_tree.which_ancestor(label.item(), clf_classes) for label in labels]
            # Map labels to their IDs
            labels = torch.tensor([clf_class_to_id[label.item()] for label in labels], device=labels.device)
            print ("Mapped labels")
            print (labels)

            # Forward pass
            embeddings, logits = model(images)
            
            # Compute losses
            nce_loss = nce_loss_fn(embeddings, labels)
            classification_loss = classification_loss_fn(logits, labels)
            total_loss = nce_loss + classification_loss  # Combined loss

            # Backward pass and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()

        print(f"Epoch [{epoch+1}/2], Loss: {running_loss/len(train_dataloader):.4f}")

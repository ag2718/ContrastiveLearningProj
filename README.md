# SpARCL: Specificity-Aware Representations for Contrastive Learning

This project explores the role of hierarchical label structures in supervised contrastive learning by varying the specificity of positive pairs and introducing a weighted contrastive loss based on class similarity. The study evaluates the trade-offs between task-relevant specificity and generalizability in the representation space, leveraging the ImageNet dataset’s hierarchical labels, particularly a subset focused on dog classifications. A novel weighted contrastive loss scheme is employed, where the importance of pairwise dissimilarities is modulated using the least common ancestor (LCA) distance in the label hierarchy, enabling finer control of similarity measures. Experimental results using a ResNet18 backbone and a classification task demonstrate how hierarchical relationships impact representation quality and classification performance.

For more information, read our blog post!

1. Download the blog folder.
2. Open the `index.html` file in a browser of your choice.

# image-similarity-comparision
Compare different approaches for image similarity


Methods / Algorithms explored 

1. Siamese Triplet Loss
2. Siamese Contrastive Loss
3. Autoencoder based
4. Combination of Feature Matching, Histogram Matching and SSIM (Structural Similarity Index Measure)
5. CLIP embedding (only image)
6. CLIP embedding (image + text)

## Overall methodology

- Dataset used for experiments - CIFAR100
- Finetuning done for `Siamese Triplet Loss`, `Siamese Contrastive Loss`, `Autoencoder based`
- During finetuning only `90` classes were used for training, `10` were unseen during training and were used only for evaluation.
- Evaluation criteria (for all methods) -
    - We compute `top_k` most similar images based on embedding distance. If all classes in `top_K` are matching with actual label we say 1 otherwise 0
- `CLIP` is used without finetuning as its already finetuned on a large dataset and can be safely considered generic embedding.


## Results

| Method                                                                                                       | Accuracy (%) | INFERENCE FPS (NVIDIA T4) |
| ------------------------------------------------------------------------------------------------------------ | ------------ | ------------------------- |
| 1\. Siamese Triplet Loss                                                                                     | 95.16        | 400                       |
| 2\. Siamese Contrastive Loss                                                                                 | 95.58        | 400                       |
| 3\. Autoencoder based                                                                                        | 89.05        | 250                       |
| 4\. Combination of Feature Matching,<br>Histogram Matching and<br>SSIM (Structural Similarity Index Measure) | 50.95        | Very slow on CPU          |
| 5\. CLIP embedding (only image)                                                                              | 98.95        | 270                       |
| 6\. CLIP embedding (image + text)                                                                            | 99.37        | 270                       |

## Conclusion

CLIP performs best. If we combine image + text embeddings then the reults improve further.

## Further improvements

1. Using better multimodal embeddings models like `BLIP2`
2. Performing optimizations for faster inference

## DEMO

To run locally - 
`pip install -r requirements.txt`

Or directly run in `google colab`

https://colab.research.google.com/drive/1rQ911tJZf8rCEipf5hcC0RTykk5F2mM8?usp=sharing


## Citations

- [Exploring Simple Siamese Representation Learning](https://arxiv.org/pdf/2011.10566)
- [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/pdf/2103.00020)
- [OpenAI CLIP](https://openai.com/index/clip/)
- [Contrastive Loss](https://vitalab.github.io/article/2019/05/15/contrastiveLoss.html)


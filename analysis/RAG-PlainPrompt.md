## Setting: 


### Data Preparation

We sample 500 descriptions from RAG-based generations, real clinical reports, and also generate same amount of queries using GPT4o with plain prompt.
The prompt is simple as: `Given the label {}, please provide segmentation based query for it to get the desired medical image area.`


### Baseline and real data comparison

[1] RAG-based generations: These queries are produced via a Retrieval-Augmented Generation (RAG) framework in Figure2 incorporated domain-specific language extracted from clinical records and prompt as shwon in Appendix Page 12.

[2] Real Clinical Reports: These consist of authentic clinical descriptions as documented in real-world settings as mentioned in line 326 - 328.

[3] Plain Prompts: These are generated using GPT-4 with a simple instruction: “Given the label {}, please provide a segmentation-based query for it to get the desired medical image area.”
(Here, “{}” is replaced with each anatomical label.)


### Experiment details

For each set, we compute text embeddings using the SentenceTransformer model ("all-MiniLM-L6-v2"). 
The embeddings are then grouped by category, and we calculate the centroid for each group. Using Euclidean distance, we compare the distance between the centroids of the RAG-based prompts and the Clinical Reports against the distance between the Plain Prompts and the Clinical Reports.



### Experiment results


`Qualitatively`:

The RAG-based prompts distribute closely with real clinical report descriptions in t-SNE 
embedding space, whereas plain prompts form tightly clustered and isolated groups. This shows that the linguistic style and semantic structure of RAG-generated prompts more closely resemble 
realistic clinical documents, better model language use in medical settings. 


`Quantitatively`:

The centroid distance (↓) of two contrast groups, RAG vs Clincal is 0.27 while Plain vs Clincal is 0.53. Centroid distance for (RAG vs Clinical) is smaller than that for (Plain vs Clinical), indicating that RAG-based prompts are semantically closer to real clinical language.
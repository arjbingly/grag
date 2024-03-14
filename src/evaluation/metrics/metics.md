# Evaluation metrics TODO


## Traditional Measures

### Generation Metrics
1. Perplexity : Fluency
2. ROGUE (Recall oriented understudy for gisting evaluation) : Response Coherence

### Retrieval Metrics
#### Order-Unaware
1. Recall
2. Precision
3. F1 Score

#### Order-Aware
4. Mean Average Precision (MAP)
5. Mean Normalized Discounted Cumulative Gain (Mean NDCG)
6. Mean Reciprocal Rank (MRR)




## What is .. ?
**Response Coherence**  
This metric evaluates how well the model generates coherent and logical responses that align with the context of the conversation. It assesses the ability of the model to provide meaningful and contextually relevant answers.

**Fluency** 
It measures how well the model's responses are structured, grammatically correct, and linguistically coherent. It assesses the model's ability to generate smooth and natural-sounding language.


## Test Datasets
[KG-RAG-datasets](https://github.com/docugami/KG-RAG-datasets)

Documents for the above dataset are SEC 10-Q filling.
These are downloaded from the investor relations websites of the below-mentioned companies.
Time - Q3 2022 - Q3 2023 (Note that 3 10-Q fillings are made in a year.)

Company - Investor Relations websites
1. APPL - https://investor.apple.com/sec-filings/default.aspx
2. AMZN - https://ir.aboutamazon.com/sec-filings/default.aspx
3. INTC - https://www.intc.com/filings-reports/quarterly-reports
4. MSFT - https://www.microsoft.com/en-us/Investor/sec-filings.aspx
5. NVDA - https://investor.nvidia.com/financial-info/sec-filings/default.aspx

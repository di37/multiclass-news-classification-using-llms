## Text Classification with Large Language Models - Llama3, Llama3:70B, Mixtral:8x7B, and GPT-3.5-Turbo

The objective of this project is to evaluate the performance of Large Language Models (LLMs) in multi-class classification tasks using few-shot prompting. 
We have selected the [BBC News Dataset](https://www.kaggle.com/datasets/pariza/bbc-news-summary) from Kaggle for our analysis. This dataset categorizes news articles into five classes: Business, Technology, Sports, Politics, and Entertainment.

To reproduce this project, you can follow along the notebooks in `workflow` directory done step by step.
1. `01_Dataset_Preparation.ipynb`: Few shot examples selected and injected to the user prompt followed careful preparation of the test dataset such that few shot example doesn't get included in the test dataset and also managed to maintain the balance of the dataset. Also, dataset is split into 10 smaller datasets to avoid repeating experiment from scratch if anything goes wrong. 
2. `02_Text_Classification.ipynb`: Text classification using different LLMs (Llama3, Llama3:70B, Mixtral:8x7B, and GPT-3.5-Turbo) on a 500-sample test dataset. It is recommended to run `02_text_classification.py` script.
3. `03_Data_Cleaning_of_Outputs.ipynb`: As per the logs, during inferencing, we can see that most of the models except GPT-3.5-Turbo gave more than one word answer despite being mentioned in the prompt. But, that doesnt mean it gave wrong output as it still gave the classification output but with explanation that was not necessary. In this notebook, necessary clean up was done.
4. `04_Evaluation.ipynb`: Evaluation of the models on a 500-sample test dataset. All of the required metrics are computed and confusion matrix plotted. 

In the next sections, we will discuss the results of the experiments.

## Accuracy, Precision, Recall, F1 Score and Time Taken

Following are the results for the news classification task using different Language Models (LLMs) on a 500-sample test dataset.

| accuracy | precision | recall   | f1       | model                | time_taken_in_seconds |
|----------|-----------|----------|----------|---------------------|------------------------|
| 0.886    | 0.796789  | 0.894913 | 0.800293 | llama3              | 360                    |
| 0.976    | 0.981302  | 0.977032 | 0.978721 | llama3:70b          | 720                    |
| 0.914    | 0.827455  | 0.821479 | 0.843914 | mixtral             | 540                    |
| 0.962    | 0.805873  | 0.802054 | 0.803609 | gpt-3.5-turbo       | 900                    |

Based on the provided results for the news classification task using different Language Models (LLMs) on a 500-sample unseen test dataset, here are the interpretations:

1. Accuracy:

    - The llama3:70b model achieves the highest accuracy of 0.976, indicating that it correctly classifies 97.6% of the test samples.
    - The gpt-3.5-turbo model follows closely with an accuracy of 0.962, correctly classifying 96.2% of the samples.
    - The llama3 and mixtral models have lower accuracies of 0.886 and 0.914, respectively.

2. Precision:

    - The llama3:70b model has the highest precision of 0.981302, meaning that when it predicts a particular class, it is correct 98.13% of the time.
    - The llama3 model has the lowest precision of 0.796789, indicating a higher rate of false positive predictions.

3. Recall:

    - The llama3:70b model achieves the highest recall of 0.977032, suggesting that it successfully identifies 97.70% of the samples from each class.
    - The mixtral model has the lowest recall of 0.821479, indicating that it may miss some instances of each class.

4. F1 Score:

    - The llama3:70b model has the highest F1 score of 0.978721, which is the harmonic mean of precision and recall. This suggests a good balance between precision and recall.
    - The llama3 model has the lowest F1 score of 0.800293, indicating a lower overall performance compared to the other models.

5. Time Taken:

    - The llama3 model is the fastest, taking only 360 seconds to classify the 500 test samples.
    - The llama3:70b model takes the longest time of 720 seconds, which is twice the time taken by llama3. This could be due to the larger model size or more complex architecture of llama3:70b.

Overall, the llama3:70b model demonstrates the best performance in terms of accuracy, precision, recall, and F1 score. It correctly classifies the highest percentage of samples, has the lowest false positive rate, successfully identifies the most instances of each class, and maintains a good balance between precision and recall.

The gpt-3.5-turbo model also shows strong performance, with accuracy and F1 score close to llama3:70b. The mixtral and llama3 models have comparatively lower performance metrics.

However, it's important to note that the llama3:70b model takes the longest time to classify the samples.

In summary, llama3:70b exhibits the best overall performance for the news classification task under the given five categories, followed closely by gpt-3.5-turbo, while llama3 is the fastest model.

## GPU Consumption


## Confusion Matrix

Confusion Matrix as follows:

Let's analyze each model's confusion matrix, focusing on misclassifications:

**1. GPT-3.5-Turbo:**
- Strengths:

    - The classifier performs well overall, with most instances correctly classified along the diagonal of the confusion matrix.
    - The "sport" class has a particularly high number of correct classifications and only one misclassification, indicating strong performance for this class.

- Weaknesses:

    - There are some misclassifications between similar classes, such as "business" and "politics", or "tech" and other classes, suggesting some confusion between these categories.
    - The "business" class has a few instances misclassified as "others" and "politics", indicating potential areas for improvement.
    - The "tech" class has misclassifications spread across "business", "entertainment", and "politics", suggesting that the classifier might struggle to distinguish "tech" from these other classes in some cases.

**2. Mixtral:8x7B:** 
- Strengths:

    - The classifier has a strong overall performance, with most instances correctly classified along the diagonal of the confusion matrix.
    - The "sport" class performs exceptionally well, with all instances correctly classified and no misclassifications.
    - The "entertainment" and "politics" classes have a very high number of correctly classified instances compared to misclassifications.

- Weaknesses:

    - The "tech" class has a notable number of misclassifications, particularly with the "sport" class (12 instances), indicating potential confusion between these categories.
    - The "business" class has a few misclassifications spread across "politics", "sport", and "tech", suggesting some room for improvement in distinguishing between these classes.
    - Although the overall performance is good, there are still some misclassifications present for most classes, except for "sport" and "others".

**3. Llama3:70B:** 
- Strengths:

    - The classifier shows excellent overall performance, with most instances correctly classified along the diagonal of the confusion matrix and minimal misclassifications.
    - The "sport" class performs exceptionally well, with all instances correctly classified and no misclassifications.
    - The "entertainment" class has no misclassifications, indicating perfect performance for this category.
    - The "business", "politics", and "tech" classes have very few misclassifications, suggesting a strong ability to distinguish these classes from others.

- Weaknesses:

    - Although the performance is excellent overall, there are still a few misclassifications present for the "business", "politics", and "tech" classes.
    - The "tech" class has a few misclassifications spread across "business", "entertainment", and "politics", indicating some room for improvement in distinguishing this class from others.

**4. Llama3:** 
- Strengths:

    - The classifier demonstrates a good overall performance, with most instances correctly classified along the diagonal of the confusion matrix.
    - The "sport" class performs well, with 110 correctly classified instances and only 2 misclassifications.
    - The "politics" and "entertainment" classes have a high number of correctly classified instances compared to misclassifications.

- Weaknesses:

    - The "tech" class has a significant number of misclassifications, particularly with the "entertainment" class (18 instances), indicating potential confusion between these categories.
    - The "business" class has a notable number of misclassifications as "politics" (18 instances), suggesting difficulty in distinguishing between these two classes.
    - Although the overall performance is good, there are still some misclassifications present for most classes, except for "others".

## Conclusion

**1. GPT-3.5-Turbo:** This model shows a balanced performance across categories with a strong inclination towards correct predictions in less technical categories like sports and politics. The confusion with tech indicates a potential area of improvement.

**2. Mixtral:8x7B:** Shows good overall accuracy but tends to confuse tech and sports with other categories, indicating a potential overfit on specific textual patterns found in these categories.

**3. Llama3:70B:** Provides the best overall accuracy and F1 score, and the confusion matrix supports this with high correct predictions across most categories. The minimal confusion between categories indicates a strong generalization capability.

**4. Llama3:** While efficient in resource use, it shows significant confusion between business and tech, which could be problematic if precise category prediction in these areas is critical.

### Model Recommendation Based on Detailed Analysis
- If high accuracy and generalization are crucial, Llama3:70b stands out as the best choice, despite its higher resource consumption.
- For a balance between efficiency and performance, Mixtral offers a good compromise, though the confusion between categories needs to be considered.
- For resource-constrained environments, Llama3 could be considered, though it has limitations in distinguishing some categories clearly.

## Future Work
We can see how well these models have performed just by few shot prompting. In order to achieve more superior results we can do as follows:
- We can go for fine-tuning of little bit smaller models as it will just do only one task. 
- We can also go for dynamic few shot prompting. i.e. select similar examples of input and output as the news article input from the user which can be done with help of embedding models such as `e5-mistral-7b-instruct`(check out [embedding model leaderboard](https://huggingface.co/spaces/mteb/leaderboard)) and vector database such `pinecone`, `chroma`, `qdrant` etc.
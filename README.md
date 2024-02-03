# LLM-Fine-Tuning-using-Cohere-for-Medical-Data

## Introduction
In this repo, we will adopt an entirely different method for fine-tuning a large language model, leveraging a platform called Cohere. This approach allows you to craft a personalized model with just providing sample inputs and outputs, while the service handles the fine-tuning process in the background. Essentially, you supply a set of examples and, in return, obtain a fine-tuned model. For instance, in the context of a classification model, a sample entry would consist of a pair containing <text, label>.

Cohere utilizes a collection of exclusive models to execute various functions like [rerank](https://txt.cohere.com/rerank/), [embedding](https://docs.cohere.com/docs/multilingual-language-models), [chat](https://cohere.com/chat), and more, all accessible through APIs. The interfaces encompass not only generative tasks but also a diverse array of endpoints that are explicitly designed for Retrieval Augmented Generation (RAG) applications. In this context, the robust base model produces contextually informed representations. Read more about [Semantic Search](https://docs.cohere.com/docs/what-is-semantic-search)https://docs.cohere.com/docs/what-is-semantic-search for more details.

Additionally, they empower us to enhance their models by customizing them to suit our precise use case through fine-tuning. It is possible to create custom models for 3 distinct objectives: 1) Generative task where we expect the model to generate a text as the output, 2) Classifier which the model will categorizes the text in different categories, or 3) Rerank to enhance semantic search results.

Lets explores the procedure of fine-tuning a customized generative model using medical texts to extract information. The task, known as [Named Entity Recognition (NER)](https://en.wikipedia.org/wiki/Named-entity_recognition), empowers the models to identify various entities (such as names, locations, dates, etc.) within a text. Large Language Models simplify the process of instructing a model to locate desired information within content. In the following sections we will delve into the procedure of fine-tuning a model to extract diseases, chemicals, and their relationships from a paper abstract

## Cohere API
The Cohere service offers a range of robust base models tailored for various objectives. Since our focus is on generative tasks, you have the option to select either base models for faster performance or command models for enhanced capability. Both variants also include a "light" version, which is a smaller-sized model, providing you with additional choices.

To access the API, you must first [create an account](https://dashboard.cohere.com/welcome/register) on the Cohere platform. You can then proceed to the "API Keys" page, where you will find a Trial key available for free usage. Note that the trial key has rate limitations and cannot be used in a production environment. Nonetheless, there is a valuable opportunity to utilize the models and conduct your experiments prior to submitting your application for production deployment.

Now, let's install the Cohere Python package to seamlessly use their API. You should run the following command in terminal.

```
pip install cohere
```
Next, you'll need to create a Cohere object, which requires your API key and a prompt to generate a response for your request. You can utilize the following code, but please remember to replace the API placeholder with your own key.


```
import cohere  

co = cohere.Client("<API_KEY>")

prompt = """The following article contains technical terms including diseases, drugs and chemicals. Create a list only of the diseases mentioned.

Progressive neurodegeneration of the optic nerve and the loss of retinal ganglion cells is a hallmark of glaucoma, the leading cause of irreversible blindness worldwide, with primary open-angle glaucoma (POAG) being the most frequent form of glaucoma in the Western world. While some genetic mutations have been identified for some glaucomas, those associated with POAG are limited and for most POAG patients, the etiology is still unclear. Unfortunately, treatment of this neurodegenerative disease and other retinal degenerative diseases is lacking. For POAG, most of the treatments focus on reducing aqueous humor formation, enhancing uveoscleral or conventional outflow, or lowering intraocular pressure through surgical means. These efforts, in some cases, do not always lead to a prevention of vision loss and therefore other strategies are needed to reduce or reverse the progressive neurodegeneration. In this review, we will highlight some of the ocular pharmacological approaches that are being tested to reduce neurodegeneration and provide some form of neuroprotection.

List of extracted diseases:"""

response = co.generate(  
    model='command',  
    prompt = prompt,  
    max_tokens=200,  
    temperature=0.750)

base_model = response.generations[0].text

print(base_model)
```

```
- glaucoma
- primary open-angle glaucoma
```

The provided code utilizes the cohere.Client() method to input your API key. Subsequently, the prompt variable will contain the model's instructions. In this case, we want the model to read a scientific paper's abstract from the [PubMed website](https://pubmed.ncbi.nlm.nih.gov/) and extract the list of diseases it can identify. Finally, we employ the cohere object's .generate() method to specify the model type and provide the prompts, along with certain control parameters. The max_tokens parameter determines the maximum number of new tokens the model can produce, while the temperature parameter governs the level of randomness in the generated results. As you can see, the command model is robust enough to identify diseases without the need for any examples or additional information. In the upcoming sections, we will explore the fine-tuning feature to assess whether we can enhance the model performance even further.



## The Dataset
Before delving into the details of fine-tuning, let's begin by introducing the dataset we are utilizing and clarifying the objective. We will be utilizing the dataset known as [BC5CDR](https://paperswithcode.com/dataset/bc5cdr) which is short for BioCreative V Chemical Disease Relation. This dataset comprises 1,500 PubMed research papers that have been manually annotated by human experts with structured information. The data has been divided into training, validation, and testing sets, with each set containing 500 samples.

Our goal is to fine-tune the model to enable it to identify and extract the names of various diseases/chemicals and their relationship from text. This is very useful because the information about the relationships between chemicals and diseases are usually specified in the paper abstracts, but in this form it’s not actionable. That is, it’s not possible to search for “all the chemicals that influence the disease X”, because we’d have to read all the papers mentioning the “disease X” to do it. If we had an accurate way of extracting this structured information from the unstructured texts of the papers, it would be useful for doing these searches.

Now, let's perform some preprocessing on the dataset to transform it into a suitable format for the Cohere service. They support files in three formats: CSV, JSONL, or plain text files. We will use the JSONL format, which should align with the following template.

```
"prompt": "This is the first prompt", "completion": "This is the first completion"}
{"prompt": "This is the second prompt", "completion": "This is the second completion"}
```

You can download the dataset in JSON format in thus repo as bc5cdr.json

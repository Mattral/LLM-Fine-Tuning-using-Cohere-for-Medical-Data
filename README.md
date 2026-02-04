# LLM-Fine-Tuning-using-Cohere-for-Medical-Data


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

You can download the dataset in JSON format in thus repo as `bc5cdr.json`

Then, we can open file using the code below. 

```
with open('bc5cdr.json') as json_file:
    data = json.load(json_file)

print(data[0])
```

Now, we can iterate through the dataset, extract the abstracts and related entities, and include the necessary instructions for training. There are two sets of instructions: the first set aids the model in understanding the task, while the second set prompts it how to generate the response.

```
instruction = "The following article contains technical terms including diseases, drugs and chemicals. Create a list only of the diseases mentioned.\n\n"
output_instruction = "\n\nList of extracted diseases:\n"
```

The instruction variable establishes the guidelines, while the output_instruction defines the desired format for the output. Now, we loop through the dataset and format each instance.

```
the_list = []
for item in data:
  dis = []

  if item['dataset_type'] != "test": continue; # Don't use test set

	# Extract the disease names
	for ent in item['passages'][1]['entities']: # The annotations
    if ent['type'] == "Disease": # Only select disease names
      if ent['text'][0] not in dis: # Remove duplicate diseases in a text
        dis.append(ent['text'][0])

	the_list.append( 
			{'prompt':  instruction +
									item['passages'][1]['text'] +
									output_instruction,
			'completion': "- "+ "\n- ".join(dis)}
	)
```

The mention code may appear complex, but for each sample, it essentially iterates through all the annotations and selectively chooses only the disease-related ones. We employ this approach because the dataset includes extra labels for chemicals, which are not relevant to our model. Finally, it will generate a dictionary containing the prompt and completion keys. The prompt will incorporate the paper abstract and append the instructions to it, whereas the completion will contain a list of disease names, with each name on a separate line. Now, use the following code to save the dataset in JSONL format.

```
# Writing to sample.json
with open("disease_instruct_all.jsonl", "w") as outfile:
  for item in the_list:
    outfile.write(json.dumps(item) + "\n")
```

The formatted dataset will be saved in a file called disease_instruct_all.jsonl. Also, it worth noting that we are concatenating the training and validation set to make a total of 1K samples. The final dataset that is used for fine-tuning has 3K samples which is consists of 1K for diseases + 1K for Chemicals + 1K for their relationships.

## The Fine-Tuning
Now, it's time to employ the prepared dataset for the fine-tuning process. The good news is that we have completed the majority of the challenging tasks. The Cohere platform will only request a nickname to save your custom model. It's worth noting that they provide advanced options if you wish to train your model for a longer duration or adjust the learning rate. Here is a detailed guide on [training a custom model](https://docs.cohere.com/docs/finetuning), and you can also refer to the Cohere documentation for Training Custom Models, complete with helpful screenshots.

You should navigate to the models page using the sidebar and click on the “Create a custom model” button. On the next page, you will be prompted to select the model type, which, in our case, will be the Generate option. It is time to proceed to upload a dataset, either from the previous step or from your custom dataset. Afterward, click the "Review data" button to display a few samples from the dataset. This step is designed to verify that the platform can read your data as expected. If everything appears to be in order, click the "Continue" button.

The last step is to chose a nickname for your model. Also, you can change the training hyperparameters by clicking on the “HYPERPARAMETERS (OPTIONAL)” link. You have options such as train_steps to determine the duration, learning_rate to adjust the model's speed of adaptation, and batch_size, which specifies the number of samples the model processes in each iteration, among others. In our experience, the default parameters worked well, but feel free to experiment with these settings. Once you are ready, click the "Initiate training" button.

That’s it! Cohere will send you an email once the fine-tuning process is completed, providing you with the model ID for use in your APIs.

## Extract Disease Names
In the code snippet below, we employ the same prompt as seen in the first section; however, we use the model ID of the network we just fine-tuned. Let’s see if there are any improvements.
```
response = co.generate(  
    model='2075d3bc-eacf-472e-bd26-23d0284ec536-ft',  
    prompt=prompt,  
    max_tokens=200,  
    temperature=0.750)

disease_model = response.generations[0].text

print(disease_model)
```

```
- neurodegeneration
- glaucoma
- blindness
- POAG
- glaucomas
- retinal degenerative diseases
- neurodegeneration
- neurodegeneration
```

As evident from the output, the model can now identify a broad spectrum of new diseases, highlighting the effectiveness of the fine-tuning approach. The Cohere platform offers both a user-friendly interface and a potent base model to build upon.

## Extract Chemical Names
In the upcoming test, we will assess the performance of our custom models in extracting chemical names compared to the baseline model. To eliminate the need for redundant code mentions, we will only present the prompt, followed by the output of each model for easy comparison. We utilized the following prompt to extract information from a text within the test set.

```
prompt = """The following article contains technical terms including diseases, drugs and chemicals. Create a list only of the chemicals mentioned.

To test the validity of the hypothesis that hypomethylation of DNA plays an important role in the initiation of carcinogenic process, 5-azacytidine (5-AzC) (10 mg/kg), an inhibitor of DNA methylation, was given to rats during the phase of repair synthesis induced by the three carcinogens, benzo[a]-pyrene (200 mg/kg), N-methyl-N-nitrosourea (60 mg/kg) and 1,2-dimethylhydrazine (1,2-DMH) (100 mg/kg). The initiated hepatocytes in the liver were assayed as the gamma-glutamyltransferase (gamma-GT) positive foci formed following a 2-week selection regimen consisting of dietary 0.02% 2-acetylaminofluorene coupled with a necrogenic dose of CCl4. The results obtained indicate that with all three carcinogens, administration of 5-AzC during repair synthesis increased the incidence of initiated hepatocytes, for example 10-20 foci/cm2 in 5-AzC and carcinogen-treated rats compared with 3-5 foci/cm2 in rats treated with carcinogen only. Administration of [3H]-5-azadeoxycytidine during the repair synthesis induced by 1,2-DMH further showed that 0.019 mol % of cytosine residues in DNA were substituted by the analogue, indicating that incorporation of 5-AzC occurs during repair synthesis. In the absence of the carcinogen, 5-AzC given after a two thirds partial hepatectomy, when its incorporation should be maximum, failed to induce any gamma-GT positive foci. The results suggest that hypomethylation of DNA per se may not be sufficient for initiation. Perhaps two events might be necessary for initiation, the first caused by the carcinogen and a second involving hypomethylation of DNA.

List of extracted chemicals:"""
```

First, we will examine the output of the base model.

```
- 5-azacytidine (5-AzC)
- benzo[a]-pyrene
- N-methyl-N-nitrosourea
- 1,2-dimethylhydrazine
- CCl4
- 2-acetylaminofluorene
```

Followed by the predictions generated by the custom fine-tuned model.
```
- 5-azacytidine
- 5-AzC
- benzo[a]-pyrene
- N-methyl-N-nitrosourea
- 1,2-dimethylhydrazine
- 1,2-DMH
- 2-acetylaminofluorene
- CCl4
- [3H]-5-azadeoxycytidine
- cytosine
```

It is clear that the custom model is better suited for our specific task and adapts readily based on the samples it has encountered.

## Extract Relations
The final test involves employing the model to extract complex relationships between chemicals and the diseases they impact. It is an advanced task that could pose some challenges for the base model. As previously done, we begin by introducing the prompt we employed from the test set.

```
prompt = """The following article contains technical terms including diseases, drugs and chemicals. Create a list only of the influences between the chemicals and diseases mentioned.

The yield of severe cirrhosis of the liver (defined as a shrunken finely nodular liver with micronodular histology, ascites greater than 30 ml, plasma albumin less than 2.2 g/dl, splenomegaly 2-3 times normal, and testicular atrophy approximately half normal weight) after 12 doses of carbon tetrachloride given intragastrically in the phenobarbitone-primed rat was increased from 25% to 56% by giving the initial "calibrating" dose of carbon tetrachloride at the peak of the phenobarbitone-induced enlargement of the liver. At this point it was assumed that the cytochrome P450/CCl4 toxic state was both maximal and stable. The optimal rat size to begin phenobarbitone was determined as 100 g, and this size as a group had a mean maximum relative liver weight increase 47% greater than normal rats of the same body weight. The optimal time for the initial dose of carbon tetrachloride was after 14 days on phenobarbitone.

List of extracted influences:"""
```

Here is the output generated by the base model.

```
severe cirrhosis of the liver influences shrinking, finely nodular, ascites, plasma albumin, splenomegaly, testicular atrophy, carbon tetrachloride, phenobarbitone

```
And here are the generations produced by the custom model.

```
- Chemical phenobarbitone influences disease cirrhosis of the liver
- Chemical carbon tetrachloride influences disease cirrhosis of the liver
```

The base model evidently attempts to establish some connections within the text. Nevertheless, it's evident that the custom fine-tuned model excels in producing well-formatted output and distinctly connecting each chemical to the respective disease. This task poses a significant challenge for a general-purpose model; however, it showcases the effectiveness of fine-tuning by simply providing a couple of thousands samples of the task we aim to accomplish.



---

## Overview & Scope

This repository is a **case study** demonstrating how to fine-tune a Large Language Model
using **Cohere’s managed fine-tuning platform** for **biomedical information extraction** tasks.

### What this repo shows
- API-based fine-tuning without model weight access
- Named Entity Recognition (NER) for diseases and chemicals
- Relation extraction between chemicals and diseases
- How task-specific examples can significantly improve domain performance

### What this repo is NOT
- A framework for training open-weight models
- A replacement for clinical NLP systems
- A benchmarked or production-ready medical AI pipeline

This project is intended for **educational and experimental purposes**, focusing on
**workflow design and prompt–completion formatting**, rather than model internals.


---

## Disclaimer

This repository is **not intended for clinical use**.
All examples operate on publicly available research abstracts and are provided
for **educational and experimental purposes only**.

Outputs from language models should not be used for medical diagnosis,
treatment decisions, or clinical workflows without proper validation.


----


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

You can download the dataset in JSON format in thus repo as `bc5cdr.json`

Then, we can open file using the code below. 

```
with open('bc5cdr.json') as json_file:
    data = json.load(json_file)

print(data[0])
```

Now, we can iterate through the dataset, extract the abstracts and related entities, and include the necessary instructions for training. There are two sets of instructions: the first set aids the model in understanding the task, while the second set prompts it how to generate the response.

```
instruction = "The following article contains technical terms including diseases, drugs and chemicals. Create a list only of the diseases mentioned.\n\n"
output_instruction = "\n\nList of extracted diseases:\n"
```

The instruction variable establishes the guidelines, while the output_instruction defines the desired format for the output. Now, we loop through the dataset and format each instance.

```
the_list = []
for item in data:
  dis = []

  if item['dataset_type'] != "test": continue; # Don't use test set

	# Extract the disease names
	for ent in item['passages'][1]['entities']: # The annotations
    if ent['type'] == "Disease": # Only select disease names
      if ent['text'][0] not in dis: # Remove duplicate diseases in a text
        dis.append(ent['text'][0])

	the_list.append( 
			{'prompt':  instruction +
									item['passages'][1]['text'] +
									output_instruction,
			'completion': "- "+ "\n- ".join(dis)}
	)
```

The mention code may appear complex, but for each sample, it essentially iterates through all the annotations and selectively chooses only the disease-related ones. We employ this approach because the dataset includes extra labels for chemicals, which are not relevant to our model. Finally, it will generate a dictionary containing the prompt and completion keys. The prompt will incorporate the paper abstract and append the instructions to it, whereas the completion will contain a list of disease names, with each name on a separate line. Now, use the following code to save the dataset in JSONL format.

```
# Writing to sample.json
with open("disease_instruct_all.jsonl", "w") as outfile:
  for item in the_list:
    outfile.write(json.dumps(item) + "\n")
```

The formatted dataset will be saved in a file called disease_instruct_all.jsonl. Also, it worth noting that we are concatenating the training and validation set to make a total of 1K samples. The final dataset that is used for fine-tuning has 3K samples which is consists of 1K for diseases + 1K for Chemicals + 1K for their relationships.

## The Fine-Tuning
Now, it's time to employ the prepared dataset for the fine-tuning process. The good news is that we have completed the majority of the challenging tasks. The Cohere platform will only request a nickname to save your custom model. It's worth noting that they provide advanced options if you wish to train your model for a longer duration or adjust the learning rate. Here is a detailed guide on [training a custom model](https://docs.cohere.com/docs/finetuning), and you can also refer to the Cohere documentation for Training Custom Models, complete with helpful screenshots.

You should navigate to the models page using the sidebar and click on the “Create a custom model” button. On the next page, you will be prompted to select the model type, which, in our case, will be the Generate option. It is time to proceed to upload a dataset, either from the previous step or from your custom dataset. Afterward, click the "Review data" button to display a few samples from the dataset. This step is designed to verify that the platform can read your data as expected. If everything appears to be in order, click the "Continue" button.

The last step is to chose a nickname for your model. Also, you can change the training hyperparameters by clicking on the “HYPERPARAMETERS (OPTIONAL)” link. You have options such as train_steps to determine the duration, learning_rate to adjust the model's speed of adaptation, and batch_size, which specifies the number of samples the model processes in each iteration, among others. In our experience, the default parameters worked well, but feel free to experiment with these settings. Once you are ready, click the "Initiate training" button.

That’s it! Cohere will send you an email once the fine-tuning process is completed, providing you with the model ID for use in your APIs.

## Extract Disease Names
In the code snippet below, we employ the same prompt as seen in the first section; however, we use the model ID of the network we just fine-tuned. Let’s see if there are any improvements.
```
response = co.generate(  
    model='2075d3bc-eacf-472e-bd26-23d0284ec536-ft',  
    prompt=prompt,  
    max_tokens=200,  
    temperature=0.750)

disease_model = response.generations[0].text

print(disease_model)
```

```
- neurodegeneration
- glaucoma
- blindness
- POAG
- glaucomas
- retinal degenerative diseases
- neurodegeneration
- neurodegeneration
```

As evident from the output, the model can now identify a broad spectrum of new diseases, highlighting the effectiveness of the fine-tuning approach. The Cohere platform offers both a user-friendly interface and a potent base model to build upon.

## Extract Chemical Names
In the upcoming test, we will assess the performance of our custom models in extracting chemical names compared to the baseline model. To eliminate the need for redundant code mentions, we will only present the prompt, followed by the output of each model for easy comparison. We utilized the following prompt to extract information from a text within the test set.

```
prompt = """The following article contains technical terms including diseases, drugs and chemicals. Create a list only of the chemicals mentioned.

To test the validity of the hypothesis that hypomethylation of DNA plays an important role in the initiation of carcinogenic process, 5-azacytidine (5-AzC) (10 mg/kg), an inhibitor of DNA methylation, was given to rats during the phase of repair synthesis induced by the three carcinogens, benzo[a]-pyrene (200 mg/kg), N-methyl-N-nitrosourea (60 mg/kg) and 1,2-dimethylhydrazine (1,2-DMH) (100 mg/kg). The initiated hepatocytes in the liver were assayed as the gamma-glutamyltransferase (gamma-GT) positive foci formed following a 2-week selection regimen consisting of dietary 0.02% 2-acetylaminofluorene coupled with a necrogenic dose of CCl4. The results obtained indicate that with all three carcinogens, administration of 5-AzC during repair synthesis increased the incidence of initiated hepatocytes, for example 10-20 foci/cm2 in 5-AzC and carcinogen-treated rats compared with 3-5 foci/cm2 in rats treated with carcinogen only. Administration of [3H]-5-azadeoxycytidine during the repair synthesis induced by 1,2-DMH further showed that 0.019 mol % of cytosine residues in DNA were substituted by the analogue, indicating that incorporation of 5-AzC occurs during repair synthesis. In the absence of the carcinogen, 5-AzC given after a two thirds partial hepatectomy, when its incorporation should be maximum, failed to induce any gamma-GT positive foci. The results suggest that hypomethylation of DNA per se may not be sufficient for initiation. Perhaps two events might be necessary for initiation, the first caused by the carcinogen and a second involving hypomethylation of DNA.

List of extracted chemicals:"""
```

First, we will examine the output of the base model.

```
- 5-azacytidine (5-AzC)
- benzo[a]-pyrene
- N-methyl-N-nitrosourea
- 1,2-dimethylhydrazine
- CCl4
- 2-acetylaminofluorene
```

Followed by the predictions generated by the custom fine-tuned model.
```
- 5-azacytidine
- 5-AzC
- benzo[a]-pyrene
- N-methyl-N-nitrosourea
- 1,2-dimethylhydrazine
- 1,2-DMH
- 2-acetylaminofluorene
- CCl4
- [3H]-5-azadeoxycytidine
- cytosine
```

It is clear that the custom model is better suited for our specific task and adapts readily based on the samples it has encountered.

## Extract Relations
The final test involves employing the model to extract complex relationships between chemicals and the diseases they impact. It is an advanced task that could pose some challenges for the base model. As previously done, we begin by introducing the prompt we employed from the test set.

```
prompt = """The following article contains technical terms including diseases, drugs and chemicals. Create a list only of the influences between the chemicals and diseases mentioned.

The yield of severe cirrhosis of the liver (defined as a shrunken finely nodular liver with micronodular histology, ascites greater than 30 ml, plasma albumin less than 2.2 g/dl, splenomegaly 2-3 times normal, and testicular atrophy approximately half normal weight) after 12 doses of carbon tetrachloride given intragastrically in the phenobarbitone-primed rat was increased from 25% to 56% by giving the initial "calibrating" dose of carbon tetrachloride at the peak of the phenobarbitone-induced enlargement of the liver. At this point it was assumed that the cytochrome P450/CCl4 toxic state was both maximal and stable. The optimal rat size to begin phenobarbitone was determined as 100 g, and this size as a group had a mean maximum relative liver weight increase 47% greater than normal rats of the same body weight. The optimal time for the initial dose of carbon tetrachloride was after 14 days on phenobarbitone.

List of extracted influences:"""
```

Here is the output generated by the base model.

```
severe cirrhosis of the liver influences shrinking, finely nodular, ascites, plasma albumin, splenomegaly, testicular atrophy, carbon tetrachloride, phenobarbitone

```
And here are the generations produced by the custom model.

```
- Chemical phenobarbitone influences disease cirrhosis of the liver
- Chemical carbon tetrachloride influences disease cirrhosis of the liver
```

The base model evidently attempts to establish some connections within the text. Nevertheless, it's evident that the custom fine-tuned model excels in producing well-formatted output and distinctly connecting each chemical to the respective disease. This task poses a significant challenge for a general-purpose model; however, it showcases the effectiveness of fine-tuning by simply providing a couple of thousands samples of the task we aim to accomplish.


# example-wikipedia-recommendation



This example showcases  the `GraphDocument` type for searching/recommending relevant wikipedia urls. In this example, queries are wikipedia urls and the retrieved (or recommended) items are related wikipedia urls. Note that the url is used only as an ID to select a document, not as the feature vector used to do search. 

This example showcases how a `GraphDocument` is capable to store a graph like dataset. Besides, it uses a Graph Convolutional Neural Network to generate embeddings for nodes in the dataset, aggregating the text embeddings of neighbouring nodes.



#### Dataset

This example uses the [wiki-cs-dataset](https://arxiv.org/abs/2007.02901) which contains 11701 wikipedia webpages. From the data, a graph can be constructed where nodes represent different url pages and links between nodes represent the existance of a link from one wikipedia page to another. In total there are 216,123 links between the nodes.

This dataset provides the following information for each node:

- A feature vector extracted computing the average of the [GloVe][https://nlp.stanford.edu/pubs/glove.pdf] mbeddings for each word in the document.
- A class label from 0 to 9
- A list of neighbour articles (documents linked to the node). Note that this list might have a different lenght for each node in the dataset.



Labels, coded from 0 to 9, correspond to the following categories:

- 0: Computational linguistics
- 1: Databases
- 2: Operating systems
- 3: Computer architecture
- 4: Computer security
- 5: Internet protocols
- 6: Computer file systems
- 7: Distributed computing architecture
- 8: Web technology
- 9: Programming language topics



#### Install dependencies

It is encouraged to set an virtual environment with `conda create --name wikipedia_env python=3.8`
or with virtualenv to install the dependencies and manage this example.

- Install [pytorch_geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).

  ```
  pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://pytorch-geometric.com/whl/torch-1.9.0+cpu.html
  ```

- The other dependencies can be installed using a `requirements.txt` file, using either

  ```bash
  conda install --file requirements.txt
  ```

  or 

  ```bash
  pip install -r requirements.txt
  ```

  

  

#### Download the data

The dataset can be downloaded cloning the `wiki-cs-dataset` repository as follows:

```bash
git clone https://github.com/pmernyei/wiki-cs-dataset
```

Note that the  repository is expected to be inside the same folder as `example-wikipedia-recommendation`.

After cloning the repository the working directory should have the following files

```
model			
wiki-cs-dataset
README.md		
app.py			
dataset_loader.py 	
model.py	
requirements.txt  
train_and_save_model.py
```



#### Run the application

The application ca be executed running:

``````bash
python app.py -t index
``````

The application will index and store the GraphDocument.

Then, the recommendation can be done running:

``````bash
python app.py -t recommend
``````

This will show the user a set of candidate urls. The user can write one of the proposed urls (or any url from the dataset) in the terminal and get a set of recommended ulrs.

```
Candidate urls:

https://en.wikipedia.org/wiki/Scheme_(programming_language)
https://en.wikipedia.org/wiki/XLeratorDB
https://en.wikipedia.org/wiki/Universal_IR_Evaluation
https://en.wikipedia.org/wiki/ARM9
https://en.wikipedia.org/wiki/PoSeidon_(malware)

 Enter url to recommend from:

https://en.wikipedia.org/wiki/ARM9

 top 10 recommended nodes:

https://en.wikipedia.org/wiki/ARM9
https://en.wikipedia.org/wiki/Atmel_ARM-based_processors
https://en.wikipedia.org/wiki/Nomadik
https://en.wikipedia.org/wiki/ARM_Cortex-M
https://en.wikipedia.org/wiki/ARM_Cortex-R
https://en.wikipedia.org/wiki/ARM_Cortex-A
https://en.wikipedia.org/wiki/EFM32
https://en.wikipedia.org/wiki/List_of_applications_of_ARM_cores
https://en.wikipedia.org/wiki/NXP_LPC
https://en.wikipedia.org/wiki/Sitara_ARM_Processor
```



#### Dataset representation in Jina

In `app.py` the data is prepared by the function `_get_input_graph` which loads the data as a `GraphDocument` named `gd` that stores the edges between documents and the nodes. 

Edges are added with `gd.add_edges(source_docs, dest_docs)` method, which recieves a list of source and target jina `Document` objects representing the nodes that are connected.

Nodes are added with the `gd.add_single_node` method and contains information about the  GloVe vector generated from the text in the url (stored as blob). 

```python
gd.add_single_node(Document(id=url,
                   blob=x.numpy(),
                   tags={'class': int(y),
                         'title': title,
                         'url': url,
                         'label': label}))
```

The embeddings produced for each node are created by `NodeEncoder` which uses `self.model.encode(node_features, adjacency)` to generate the embedding for a given node. Note that the emedding for a node depends on all the adjacent node features.



### Optional: Training the model

The provided code does not need to train a model from scatch, but the script `train_and_save_model.py` allows you to train the provided model an store it in `saved_model.torch`. 


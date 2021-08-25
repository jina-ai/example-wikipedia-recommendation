# example-wikipedia-recommendation

Example showcasing  the `GraphDocument` type for searching/recommending relevant wikipedia urls from an input wikipedia url.

It is encouraged to set an virtual environment with `conda create --name wikipedia_env python=3.8`
or with virtualenv to install the dependencies and manage this example.



#### Installing dependencies

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

  

  

  #### Downloading the data

The dataset can be downloaded cloning the `wiki-cs-dataset` repository as follows:

```bash
git clone https://github.com/pmernyei/wiki-cs-dataset
```

Note that the  repository is expected to be inside the same folder as `example-wikipedia-recommendation`.

After cloning the repository the working directory should have the following files

```
README.md		app.py			model			requirements.txt  dataset_loader.py 	model.py	
wiki-cs-dataset
```

#### Running the application

The application ca be executted running:

``````bash
python app.py 
``````




# example-wikipedia-recommendation
Example showcasing  the `GraphDocument` type for searching/recommending relevant wikipedia urls from an input wikipedia url.



#### Installing dependencies

To install all the dependencies you can use the `requirements.txt` file. It is encouraged to set an virtual environment with `conda create --name wikipedia_env python=3.8`
or with virtualenv. Then either

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



#### Running the application

The application ca be executted running:

``````bash
python app.py 
``````




# example-wikipedia-recommendation

Example showcasing  the `GraphDocument` type for searching/recommending relevant wikipedia urls from an input wikipedia url.

It is encouraged to set an virtual environment with `conda create --name wikipedia_env python=3.8`
or with virtualenv to install the dependencies and manage this example.



#### Install dependencies

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
python app.py 
``````

The application will index the data and prompt a set of candidate urls.

The user can write one of the proposed urls (or any url from the dataset) in the terminal and get a set of recommended ulrs.

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



### Optional: Training the model

The provided code does not need to train a model from scatch, but the script `train_and_save_model.py` allows you to train the provided model an store it in `saved_model.torch`. 


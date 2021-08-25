from torch_geometric.datasets import wikics
import json

def create_url(title):
    return f'https://en.wikipedia.org/wiki/{title}'

def data_loader():
    dataset = wikics.WikiCS('./wiki-cs-dataset_autodownload')
    metadata = json.load(open('wiki-cs-dataset/dataset/metadata.json'))
    data = dataset[0]

    return data, dataset, metadata

def url_loader():
    data, dataset, metadata = data_loader()
    urls = []
    for node_metadata in metadata['nodes']:
        urls.append(create_url(node_metadata['title']))
    return urls

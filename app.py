from typing import Dict, Optional, List

from jina import Executor, requests, Document, DocumentArray, Flow
from jina.types.arrays.memmap import DocumentArrayMemmap
from jina.types.document.graph import GraphDocument

from jina_commons import get_logger
from dataset_loader import data_loader, url_loader
from model import GCN

import numpy as np
import torch
import time
import random


class SimpleIndexer(Executor):
    """
    A simple indexer that stores all the Document data together,
    in a DocumentArrayMemmap object
    To be used as a unified indexer, combining both indexing and searching
    """

    def __init__(
            self,
            index_file_name: str = 'nodes',
            default_traversal_paths: Optional[List[str]] = None,
            default_top_k: int = 5,
            distance_metric: str = 'cosine',
            **kwargs,
    ):
        """
        Initializer function for the simple indexer
        :param index_file_name: The file name for the index file
        :param default_traversal_paths: The default traversal path that is used
            if no traversal path is given in the parameters of the request.
            This defaults to ['r'].
        :param default_top_k: default value for the top_k parameter
        :param distance_metric: The distance metric to be used for finding the
            most similar embeddings. The distance metrics supported are the ones supported by `DocumentArray` match method.
        """
        super().__init__(**kwargs)
        self._docs = DocumentArrayMemmap(self.workspace + f'/{index_file_name}', key_length=128)
        self.default_traversal_paths = default_traversal_paths or ['r']
        self.default_top_k = default_top_k
        self._distance = distance_metric
        self._use_scipy = True
        if distance_metric in ['cosine', 'euclidean', 'sqeuclidean']:
            self._use_scipy = False

        self._flush = True
        self._docs_embeddings = None
        self.logger = get_logger(self)

    @requests(on='/index')
    def index(
            self,
            docs: Optional['DocumentArray'] = None,
            parameters: Optional[Dict] = {},
            **kwargs,
    ):
        """All Documents to the DocumentArray
        :param docs: the docs to add
        :param parameters: the parameters dictionary
        """
        if not docs:
            return
        traversal_paths = parameters.get(
            'traversal_paths', self.default_traversal_paths
        )
        flat_docs = docs.traverse_flat(traversal_paths)
        self._docs.extend(flat_docs)
        self._flush = True

    @requests(on='/search')
    def search(
            self,
            docs: Optional['DocumentArray'] = None,
            parameters: Optional[Dict] = {},
            **kwargs,
    ):
        """Perform a vector similarity search and retrieve the full Document match
        :param docs: the Documents to search with
        :param parameters: the parameters for the search"""
        if not docs:
            return
        if not self._docs:
            self.logger.warning(
                'no documents are indexed. searching empty docs. returning.'
            )
            return
        traversal_paths = parameters.get(
            'traversal_paths', self.default_traversal_paths
        )
        flat_docs = docs.traverse_flat(traversal_paths)
        if not flat_docs:
            return
        top_k = int(parameters.get('top_k', self.default_top_k))
        flat_docs.match(
            self._docs,
            metric=self._distance,
            use_scipy=self._use_scipy,
            limit=top_k,
        )
        self._flush = False

    @requests(on='/delete')
    def delete(self, docs: DocumentArray, parameters: Optional[Dict] = {}, **kwargs):
        """Delete entries from the index by id
        :param docs: the documents to delete
        :param parameters: parameters to the request
        """
        if docs is None:
            return
        traversal_paths = parameters.get(
            'traversal_paths', self.default_traversal_paths
        )
        delete_docs_ids = docs.traverse_flat(traversal_paths).get_attributes('id')
        for idx in delete_docs_ids:
            if idx in self._docs:
                del self._docs[idx]

    @requests(on='/update')
    def update(self, docs: DocumentArray, parameters: Optional[Dict] = {}, **kwargs):
        """Update doc with the same id
        :param docs: the documents to update
        :param parameters: parameters to the request
        """
        if docs is None:
            return
        traversal_paths = parameters.get(
            'traversal_paths', self.default_traversal_paths
        )
        flat_docs = docs.traverse_flat(traversal_paths)
        for doc in flat_docs:
            if doc.id is not None:
                self._docs[doc.id] = doc
            else:
                self._docs.append(doc)

    @requests(on='/fill_embedding')
    def fill_embedding(self, docs: DocumentArray, **kwargs):
        """retrieve embedding of Documents by id
        :param docs: DocumentArray to search with
        """
        if not docs:
            return
        for doc in docs:
            if doc.id in self._docs:
                doc.embedding = self._docs[doc.id].embedding
            else:
                self.logger.debug(f'Document {doc.id} not found in index')

    def close(self):
        self._docs.save()


class NodeEncoder(Executor):

    def __init__(self, model_path_state_dict='./model/saved_model.torch'
                 , *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = GCN(num_node_features=300, num_classes=10, hidden_channels=128)
        self.model.load_state_dict(torch.load(model_path_state_dict))
        self.model.eval()

    @staticmethod
    def _get_dataset(nodes, adjacency):
        node_features = []
        edge_list_tensor = torch.tensor(np.vstack([adjacency.row, adjacency.col]), dtype=int)
        for node in nodes:
            # do something
            node_features.append(node.blob)
        return torch.Tensor(np.stack(node_features)), edge_list_tensor

    @requests
    def encode(self, docs: Optional[DocumentArray], **kwargs):
        if not docs:
            return

        for doc in docs:
            graph = GraphDocument(doc)
            nodes = graph.nodes
            node_features, adjacency = self._get_dataset(nodes, graph.adjacency)
            results = self.model.encode(node_features, adjacency)
            for node, embedding in zip(graph.nodes, results):
                # apply embedding to each node
                node.embedding = embedding.detach().numpy()


def create_url(title):
    return f'https://en.wikipedia.org/wiki/{title}'


def _get_input_graph():
    data, dataset, metadata = data_loader()

    gd = GraphDocument()

    edges = [(int(p[0]), int(p[1])) for p in zip(data.edge_index[0].numpy(), data.edge_index[1].numpy())]
    source_docs = []
    dest_docs = []
    for i, o in edges:
        title_i, title_o = metadata['nodes'][i]['title'], metadata['nodes'][o]['title']
        label_i, label_o = metadata['nodes'][i]['label'], metadata['nodes'][o]['label']
        url_i, url_o = create_url(title_i), create_url(title_o)
        node_i = Document(id=url_i,
                          blob=data.x[i].numpy(),
                          tags={'class': int(data.y[i]),
                                'title': title_i,
                                'url': url_i,
                                'label': label_i})
        node_o = Document(id=url_o,
                          blob=data.x[o].numpy(),
                          tags={'class': int(data.y[o]),
                                'title': title_o,
                                'url': url_o,
                                'label': label_o})

        source_docs.append(node_i)
        dest_docs.append(node_o)

    gd.add_edges(source_docs, dest_docs)

    for x, y, node in zip(data.x, data.y, metadata['nodes']):
        title = node['title']
        label = node['label']
        url = create_url(title)
        gd.add_node(Document(id=url,
                             blob=x.numpy(),
                             tags={'class': int(y),
                                   'title': title,
                                   'url': url,
                                   'label': label}))
    return [gd]


def _get_input_request(input_id):
    return [Document(id=input_id)]

def index():

    t0 = time.time()
    print('Start GraphDocument construction')
    graphs = _get_input_graph()
    print(f'\n\nFinished GraphDocument construction: Took {round(time.time() - t0)} seconds.\n\n')

    f = Flow().add(uses=NodeEncoder).add(uses=SimpleIndexer,
                                         uses_with={'index_file_name': 'nodes',
                                                    'default_traversal_paths': ['c']},
                                         uses_metas={'workspace': 'tmp/'})
    with f:
        f.index(inputs=graphs)


def search():

    n_candidates = 5
    print_label = False

    def _search(f, input_id):
        resp = f.post(on='/fill_embedding', inputs=_get_input_request(input_id), return_results=True)
        new_query = resp[0].docs[0]
        if new_query.embedding is None:
            print(f' url is not in the index, nothing to recommend')
            return []
        results = f.search(inputs=[new_query], return_results=True)
        matches = results[0].docs[0].matches
        return matches

    urls = url_loader()

    f = Flow().add(uses=SimpleIndexer,
                   uses_with={'index_file_name': 'nodes',
                              'default_traversal_paths': ['r'],
                              'default_top_k': 10,
                              'distance_metric': 'cosine'}, uses_metas={'workspace': 'tmp/'})

    with f:
        # wait for input
        while True:
            candidate_urls = [random.choice(urls) for x in range(n_candidates)]

            print(f'\n\n\nCandidate urls:\n')            
            for url in candidate_urls:
                print(url)

            print(f'\n Enter url to recommend from:\n')
            input_id = input()
            matches = _search(f, input_id)
            print(f'\n top {len(matches)} recommended nodes:\n')
            for match in matches:
                if print_label:
                    print(f'{match.tags["url"]} \t label={match.tags["label"]}')
                else:
                    print(f'{match.tags["url"]}')

if __name__ == '__main__':
    index()
    search()

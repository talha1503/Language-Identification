import numpy as np
import pandas as pd
import pickle
import sys
import re

sys.setrecursionlimit(50000)


def convert_to_ngrams(text,n=3):
  '''
  Converts to text to an n-gram list.
  Input: 
    text - String input
    n - number of characters per n-gram.
  Output:
  '''
  text = text.replace(' ','.')
  text = [text[i:i+n] for i in range(len(text)-n)]
  return text

def clean_text(text):
    '''
    Removes punctuation,symbols from the text.
    Input:
        text- String input
    Output:
        text- String
    '''
    text = text.lower()
    text = re.sub('[^a-zA-Z0-9]',' ',text) 
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    return text


class Vertex:
  '''
    A class to represent vertices in the graph.
  '''
  def __init__(self,node,languages):
    self.id = node
    self.adjacent = {}
    self.languages = languages 
    # This dictionary keeps track of the frequency of the current n-gram text for all languages.
    self.languages_dict = {language:0 for language in self.languages} 
  
  def __str__(self):
    return str(self.id) + ' adjacent: ' + str([x.id for x in self.adjacent])
  
  def add_neighbour(self,vertex,weight):
    '''
    Used to add a neighbour vertex or to update the weight between vertices
    Input:
      vertex - vertex which is to be added.
      weight - A dictionary which represents the edge weight between two vertices.
    '''
    self.adjacent[vertex] = weight
  
  def get_edge_weights(self,to_vertex):
    '''
    Used to get edge weights
    '''
    return self.adjacent[to_vertex]
  
  def update_weight(self,language):
    '''
    Used to update weight for a language of the current vertex. 
    '''
    self.languages_dict[language] += 1
  

class Graph:
  '''
    A class to represent the entire corpus of training data. 
  '''
  def __init__(self,languages):
    self.vertices_dictionary = {}
    self.total_vertices = 0
    self.languages = languages
    # This dictionary is used to initialize edge weights. 
    self.languages_dict = {language:0 for language in self.languages} 

  def __iter__(self):
    return iter(self.vertices_dictionary.values())
  
  def add_vertex(self,node):
    '''
    Used for adding a vertex to the graph.
    '''
    self.total_vertices += 1
    new_vertex = Vertex(node,languages)
    self.vertices_dictionary[node] = new_vertex
    return new_vertex
  
  def get_vertex(self,node):
    '''
    Used to return a vertex from the graph. Returns false if vertex is not present.
    '''
    if self.vertices_dictionary.get(node):
      return self.vertices_dictionary[node]
    else:
      return None

  def check_edge(self,from_vertex,to_vertex):
    '''
    Used to check if edge is present in the graph between two vertices.
    '''
    if not self.vertices_dictionary.get(from_vertex): 
      return False

    if not self.vertices_dictionary.get(to_vertex): 
      return False

    if self.vertices_dictionary[from_vertex].adjacent.get(to_vertex):
      return True 
    return False

  def add_edge(self,from_vertex,to_vertex,cost = None):
    '''
    Used to add an edge between two vertices. 
    '''
    if not cost:
      cost = self.languages_dict

    if not self.vertices_dictionary.get(from_vertex):
      self.add_vertex(from_vertex)
    if not self.vertices_dictionary.get(from_vertex):
      self.add_vertex(to_vertex)
    
    self.vertices_dictionary[from_vertex].add_neighbour(self.vertices_dictionary[to_vertex],cost)

def build_graph(x,y):
  '''
  Used to build a model from the corpus.
  Input:
    x - List of n-gram data points.
    y - List of labels for the data points.
  Output:
    model - A Graph which is trained on the corpus.
  '''
  model = Graph(y.unique())
  for text,language in zip(x,y):
    # Update Vertex weights
    for i,word in enumerate(text):
      if model.get_vertex(word):
        # Update word's language dict
        model.vertices_dictionary[word].update_weight(language)
      else:
        # Add word to graph with a language dictionary
        model.add_vertex(word)
        model.vertices_dictionary[word].update_weight(language)

    # Update edge weights
    for i in range(1,len(text)):
      if model.get_vertex(text[i]):
        # Update previous node's and current node's language weight
        prev_node = text[i-1]
        if model.check_edge(prev_node,text[i]):
          edge_weights = model.vertices_dictionary[prev_node].get_edge_weights(text[i])
          edge_weights[language] += 1
          model.vertices_dictionary[prev_node].add_neighbour(text[i],edge_weights)
        else:
          # If there is no edge between previous node and current node. 
          curr_language_dict = {lang:0 for lang in model.languages}
          curr_language_dict[language] = 1
          model.add_edge(text[i-1],text[i],curr_language_dict)
  
  return model 

def evaluate_sentence(model,x,y=None):
  '''
  Function used to evaluate a text.
  '''
  # This dictionary calculates path matching scores for each language. 
  path_matching_dict = {lang:0 for lang in model.languages}
  
  # We calculate the sum of all edge weights and sum of all vertex weights which
  # will be used to normalize score for inference on test data point.
  vertex_normalizing_scores = 0
  edge_normalizing_scores = 0
  for vertex in model.vertices_dictionary.values():
    for score in vertex.languages_dict.values():
      vertex_normalizing_scores += score

    for edge_weights in vertex.adjacent.values():
      for weight in edge_weights.values():
        edge_normalizing_scores += weight
    
  for test_vertex in x:
    if model.get_vertex(test_vertex):
      model_vertex = model.get_vertex(test_vertex)
      for lang,weight in model_vertex.languages_dict.items():
        path_matching_dict[lang] += (weight / vertex_normalizing_scores)
  
  for i in range(1,len(x)):
    prev_node = x[i-1]
    if model.check_edge(prev_node,x[i]):
      edge_weights = model.vertices_dictionary[prev_node].get_edge_weights(x[i])
      for lang,weight in edge_weight.items():
        path_matching_dict[language] += (weight / edge_normalizing_scores)
  
  # We calculate the max value from our calculated scores.
  predicted_lang = max(path_matching_dict, key=path_matching_dict.get)
  
  return predicted_lang,y

def predict(text):
    n_gram_text = convert_to_ngrams(text,3)
    model = pickle.load(open('./saved_models/liga_model.pkl','rb'))
    pred,_ = evaluate_sentence(model,n_gram_text)
    return pred
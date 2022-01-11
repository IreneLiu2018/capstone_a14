import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
import pickle
from itertools import chain
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

YEAR = 2015
NUMBER_OF_YEAR = 7

def viz_prepare(dataframe_path, processed_data_path, models_path, results_path, names_path, authors_path, missing_author_years_path, number_of_authors, raw_data_path, topics_range, folder, org_data):
  data = pickle.load(open(dataframe_path, 'rb'))

  all_docs = pickle.load(open(processed_data_path, 'rb'))

  models = pickle.load(open(models_path, 'rb'))
  results = pickle.load(open(results_path, 'rb'))
  names = pickle.load(open(names_path, 'rb'))

  authors = pickle.load(open(authors_path, 'rb'))
  missing_author_years = pickle.load(open(missing_author_years_path, 'rb'))

  model_diff = topics_range[1] - topics_range[0]

  num_topics_list = range(topics_range[0], topics_range[-1]+model_diff, model_diff)
  topicnames = {
    num_topics : ["Topic" + str(i) for i in range(num_topics)] for num_topics in num_topics_list
  }

  # index names
  docnames = ["Doc" + str(i) for i in range(len(all_docs))]

  # Make the pandas dataframe
  df_document_topic = {
      num_topics : pd.DataFrame(results[f'{num_topics}'], columns=topicnames[num_topics], index=docnames) for num_topics in num_topics_list
  }

  # Get dominant topic for each document
  dominant_topic = {
      num_topics : np.argmax(df_document_topic[num_topics].values, axis=1) for num_topics in num_topics_list
  }

  for num_topics, df in df_document_topic.items():
      df['dominant_topic'] = dominant_topic[num_topics]
        



  author_list = []
  year_list = []
  for author in authors.keys():
      for i in range(NUMBER_OF_YEAR):
          if (YEAR + i) not in missing_author_years[author]:
              author_list.append(author)
              year_list.append(YEAR + i)

  for df in df_document_topic.values():
      df['author'] = author_list
      df['year'] = year_list



  averaged = {
      num_topics : df_document_topic[num_topics].groupby('author').mean().drop(['dominant_topic', 'year'], axis=1) for num_topics in df_document_topic.keys()
  }

  filtered = {
      threshold : {num_topics : averaged[num_topics].mask(averaged[num_topics] < threshold, other=0) for num_topics in averaged.keys()} for threshold in [.1]
  }


  labels = {}
  for num_topics in topics_range:
      labels[num_topics] = filtered[.1][num_topics].index.to_list()
      labels[num_topics].extend(filtered[.1][num_topics].columns.to_list())


  sources = {threshold : {} for threshold in [.1]}
  targets = {threshold : {} for threshold in [.1]}
  values = {threshold : {} for threshold in [.1]}

  for threshold in [.1]:
      for num_topics in topics_range:
          curr_sources = []
          curr_targets = []
          curr_values = []
          index_counter = 0
          for index, row in filtered[threshold][num_topics].iterrows():
              for i, value in enumerate(row):
                  if value != 0:
                      curr_sources.append(index_counter)
                      curr_targets.append(number_of_authors + i)
                      curr_values.append(value)
              index_counter += 1
          sources[threshold][num_topics] = curr_sources
          targets[threshold][num_topics] = curr_targets
          values[threshold][num_topics] = curr_values

  positions = {
      num_topics : {label : i for i, label in enumerate(labels[num_topics])} for num_topics in averaged.keys()
  }
  
  print('sources, targets, and values for sankey DONE')

  def split_into_ranks(array):
      ranks = []
      for value in array:
          for i, percentage in enumerate(np.arange(.1, 1.1, .1)):
              if value <= np.quantile(array, percentage):
                  ranks.append(i + 1)
                  break
      return ranks

  final_values = {threshold : {} for threshold in [.1]}

  for threshold in [.1]:
      for num_topics in topics_range:
          curr_values_array = np.array(values[threshold][num_topics])
          final_values[threshold][num_topics] = split_into_ranks(curr_values_array)


  counts = CountVectorizer().fit_transform(data['abstract_processed'])
  transformed_list = []
  for model in models.values():
      transformed_list.append(model.transform(counts))


  dataframes = {threshold : {} for threshold in [.1]}
  for i, matrix in enumerate(transformed_list):
      for threshold in [.1]:
          df = pd.DataFrame(matrix)
          df.mask(df < threshold, other=0, inplace=True)
          df['HDSI_author'] = data['HDSI_author']
          df['year'] = data['year']
          df['citations'] = data['times_cited'] + 1

          # noralization of citations: Scaling to a range [0, 1]
          df['citations_norm'] = df.groupby(by=['HDSI_author', 'year'])['citations'].apply(lambda x: (x-x.min())/(x.max()-x.min()))#normalize_by_group(df=df, by=['author', 'year'])['citations']
          df['abstract'] = data['abstract']
          df['title'] = data['title']
          df.fillna(1, inplace=True)
          
          #alpha weight parameter for weighting importance of citations vs topic relation
          alpha = .75
          for topic_num in range((i+2) * 5):
              df[f'{topic_num}_relevance'] = alpha * df[topic_num] + (1-alpha) * df['citations_norm']
          dataframes[threshold][(i+2) * 5] = df

  def create_top_list(data_frame, num_topics, threshold):
      top_5s = []
      the_filter = filtered[threshold][num_topics]
      for topic in range(num_topics):
          relevant = the_filter[the_filter[f'Topic{topic}'] != 0].index.to_list()
  #         print(relevant)
          to_append = data_frame[data_frame[f'{topic}_relevance'] > 0].reset_index()
        #   print(to_append.columns)
          to_append = to_append[to_append['HDSI_author'].isin(relevant)].reset_index()
          top_5s.append(to_append)
      return top_5s

  tops = {
      threshold : {num_topics : create_top_list(dataframes[threshold][num_topics], num_topics, threshold) for num_topics in num_topics_list} for threshold in [.1]
  }

  print('large dataframe including the document-topic and calculated relevant score DONE')

  # sankey diagrams for diff numbers of topics
  def display_topics_list(model, feature_names, no_top_words):
      topic_list = []
      for topic_idx, topic in enumerate(model.components_):
          topic_list.append(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
      return topic_list

  link_labels = {}
  for num_topics in topics_range:
      link_labels[num_topics] = labels[num_topics].copy()
      link_labels[num_topics][number_of_authors:] = display_topics_list(models[f'{num_topics}'], names, 10)

  lst_of_topics = topics_range.copy()

  heights = {
    lst_of_topics[0] : 1000,
    lst_of_topics[1] : 1500,
    lst_of_topics[2] : 2000,
    lst_of_topics[3] : 2500,
    lst_of_topics[4] : 3000
  }

  figs = {threshold : {} for threshold in [.1]}
  for threshold in [.1]:
      for num_topics in topics_range:
          fig = go.Figure(data=[go.Sankey(
              node = dict(
                  pad = 15,
                  thickness = 20,
                  line = dict(color = 'black', width = 0.5),
                  label = labels[num_topics],
                  color = ['#666699' for i in range(len(labels[num_topics]))],
                  customdata = link_labels[num_topics],
                  hovertemplate='%{customdata} Total Flow: %{value}<extra></extra>'
              ),
              link = dict(
                  color = ['rgba(204, 204, 204, .5)' for i in range(len(sources[threshold][num_topics]))],
                  source = sources[threshold][num_topics],
                  target = targets[threshold][num_topics],
                  value = final_values[threshold][num_topics]
              )
          )])
          fig.update_layout(title_text="Author Topic Connections", font=dict(size = 10, color = 'white'), height=heights[num_topics], paper_bgcolor="black", plot_bgcolor='black')
          figs[threshold][num_topics] = fig


  top_words = {
      lst_of_topics[0] : display_topics_list(models['{}'.format(lst_of_topics[0])], names, 10),
      lst_of_topics[1] : display_topics_list(models['{}'.format(lst_of_topics[1])], names, 10),
      lst_of_topics[2] : display_topics_list(models['{}'.format(lst_of_topics[2])], names, 10),
      lst_of_topics[3] : display_topics_list(models['{}'.format(lst_of_topics[3])], names, 10),
      lst_of_topics[4] : display_topics_list(models['{}'.format(lst_of_topics[4])], names, 10)
  }

  # 'final_hdsi_faculty_updated.csv'
  combined = pd.read_csv(raw_data_path)
  # combined[combined.title == 'Elder-Rule-Staircodes for Augmented Metric Spaces'].abstract

  locations = {}
  for i, word in enumerate(names):
      locations[word] = i

  print('sankey diagram for different numbers of topics DONE')

  pickle.dump(figs, open('{}/sankey_dash/figs.pkl'.format(folder), 'wb'))
  pickle.dump(tops, open('{}/sankey_dash/tops.pkl'.format(folder), 'wb'))
  pickle.dump(top_words, open('{}/sankey_dash/top_words.pkl'.format(folder), 'wb'))
  pickle.dump(author_list, open('{}/sankey_dash/author_list.pkl'.format(folder), 'wb'))
  pickle.dump(labels, open('{}/sankey_dash/labels.pkl'.format(folder), 'wb'))
  pickle.dump(positions, open('{}/sankey_dash/positions.pkl'.format(folder), 'wb'))
  pickle.dump(sources, open('{}/sankey_dash/sources.pkl'.format(folder), 'wb'))
  pickle.dump(targets, open('{}/sankey_dash/targets.pkl'.format(folder), 'wb'))
  
  pickle.dump(locations, open('{}/sankey_dash/locations.pkl'.format(folder), 'wb'))
  pickle.dump(models, open('{}/sankey_dash/models.pkl'.format(folder), 'wb'))
  pickle.dump(names, open('{}/sankey_dash/names.pkl'.format(folder), 'wb'))

  combined = pd.read_csv('{}/raw/{}'.format(folder, org_data))
  pickle.dump(combined, open('{}/sankey_dash/combined.pkl'.format(folder), 'wb'))

  print('variables for sankey launch SAVED')







  # return figs, tops, top_words, combined, author_list, labels, positions, sources, targets, locations, models, names

# def save_viz_variable(output_figs_path, output_tops_path, output_top_words_path, output_combined_path, output_author_list_path, output_labels_path, output_positions_path, 
#   output_sources_path, output_targets_path, output_locations_path, output_models_path, output_names_path):
  # pickle.dump(names, open(output_names_path, 'wb'))
  # tops = pickle.load(open('tops.pkl', 'rb'))
  # top_words = pickle.load(open('top_words.pkl', 'rb'))
  # combined = pickle.load(open('combined.pkl', 'rb'))
  # author_list = pickle.load(open('author_list.pkl', 'rb'))
  # labels = pickle.load(open('labels.pkl', 'rb'))
  # positions = pickle.load(open('positions.pkl', 'rb'))
  # sources = pickle.load(open('sources.pkl', 'rb'))
  # targets = pickle.load(open('targets.pkl', 'rb'))
  # locations = pickle.load(open('locations.pkl', 'rb'))
  # models = pickle.load(open('models.pkl', 'rb'))
  # names = pickle.load(open('names.pkl', 'rb'))



import http.client, urllib.request, urllib.parse, urllib.error, base64
import requests
import pandas as pd
from io import StringIO
from io import BytesIO
import json
import numpy as np



def get_data(faculty_list, key, outpath=None):

# 'data/raw/HDSI Faculty Council - Sheet1 - HDSI Faculty Council - Sheet1.csv'
    # researchers = pd.read_csv(faculty_list)

    ms_id = [2243367769, 2529224718] # justin, aaron

    # -----------------------------
    print('Data Extraction From MS API')
    data_ms = []

    headers = {
        # Request headers '93e9736ec66846b5a3252e08f3ec5f4c'
        'Ocp-Apim-Subscription-Key': key,
    }

    for id in ms_id:
        params = urllib.parse.urlencode({
            # Request parameters
            'expr': "AND(Composite(AA.AuId={}))".format(id),
            'model': 'latest',
            'count': '9999',
            'offset': '0',
            # 'orderby': '{string}',
            'attributes': 'Id,AW,DN,F.DFN,Y,J.JN,CC,AA.DAuN',
        })

        try:
            conn = http.client.HTTPSConnection('api.labs.cognitive.microsoft.com')
            conn.request("GET", "/academic/v1.0/evaluate?%s" % params, "{body}", headers)
            response = conn.getresponse()
            bytes_data = response.read()
            data_ms.append(bytes_data)
            # print(data)
            conn.close()
        except Exception as e:
            print("[Errno {0}] {1}".format(e.errno, e.strerror))

    data = []

    for i in range(len(data_ms)):
        my_json = data_ms[i].decode('utf8')
        data.append(json.loads(my_json))

    # -------------------------------

    print('Organize the data in usable format')

    wd = dict()
    for j in range(len(data)):
        # author id
        author = data[j]['expr'][22:-2]

        # for each author
        d = dict()

        for i in range(len(data[j]['entities'])):
            # paper id as keys
            id_key = str(data[j]['entities'][i]['Id'])
            if id_key not in d.keys():
                d[id_key] = {}

            # abstract inner keys
            if "abstract" not in d[id_key].keys():
                d[id_key]['abstract'] = []
            if "AW" not in data[j]['entities'][i].keys():
                d[id_key]['abstract'] = []
            else:
                d[id_key]['abstract'] += data[j]['entities'][i]['AW']
                d[id_key]['abstract'] = list(set(d[id_key]['abstract']))
            
            # title inner keys
            d[id_key]['title'] = data[j]['entities'][i]['DN']

            # field inner keys
            d[id_key]['field'] = data[j]['entities'][i]['F']

            #year inner key
            d[id_key]['year'] = data[j]['entities'][i]['Y']

            if "J" not in data[j]['entities'][i].keys():
                d[id_key]['journal'] = {}
            else:
                d[id_key]['journal'] = data[j]['entities'][i]['J']

            d[id_key]['times_cited'] = data[j]['entities'][i]['CC']

            d[id_key]['author_name'] = data[j]['entities'][i]['AA']
            

        
        wd['{}'.format(author)] = d


    # ---------------------

    print("file justin&aaron.csv outputted to data raw")

    # justin
    df1 = pd.DataFrame.from_dict(dict['2243367769'], orient='index')

    # aaron
    df2 = pd.DataFrame.from_dict(dict['2529224718'], orient='index')

    df = pd.concat([df1,df2])

    df['authors'] = df['author_name'].apply(lambda x: [x[i]['DAuN'] for i in range(len(x))])
    df['authors_count'] = df['authors'].apply(lambda x: len(x))
    df['concepts'] = df['field'].apply(lambda x: [x[i]['DFN'] for i in range(len(x))])
    df['journal.title'] = df['journal'].apply(lambda x: x['JN'] if len(x) > 0 else np.NaN)

    df = df.drop(['field','journal','author_name'],axis=1)

    # df.to_csv('data/raw/justin&aaron.csv')

    if outpath is None:
            return None
    else:
        # write data to outpath
        df.to_csv(outpath)
        print('data gathered')
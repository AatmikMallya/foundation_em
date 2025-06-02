
from neuprint import Client
import os
import pandas as pd
import numpy as np

token_id = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6ImdhcnJldHQuc2FnZXJAeWFsZS5lZHUiLCJsZXZlbCI6Im5vYXV0aCIsImltYWdlLXVybCI6Imh0dHBzOi8vbGgzLmdvb2dsZXVzZXJjb250ZW50LmNvbS9hLS9BT2gxNEdpTGNqZXlHYWNnS3NPcTgzdDNfczBoTU5sQUtlTkljRzdxMkU5Rz1zOTYtYz9zej01MD9zej01MCIsImV4cCI6MTgwMTAxNzUwNn0.dzq7Iy01JwSWbKq-Qvi8ov7Hwr0-ozpYeSnOsUD-Mx0"

on_HPC = False
if on_HPC:
    home_dir = '/home/gs697/project/clean_mito_code'
    google_application_credentials = '/home/gs697/project/application_default_credentials.json' # HPC credentials
    tensorstore_ca_bundle = '/home/gs697/project/ca-certificates.crt' # HPC credentials
else:
    home_dir = '/Users/aatmikmallya/Desktop/research/fly/synapseSegmentation'
    google_application_credentials = '~/.config/gcloud/application_default_credentials.json'
    # tensorstore_ca_bundle = '/etc/ssl/certs/ca-certificates.crt'
    tensorstore_ca_bundle = '/etc/ssl/cert.pem'



c = Client('neuprint.janelia.org', dataset = 'hemibrain:v1.2.1', token=token_id)
server = 'http://hemibrain-dvid.janelia.org'
node_class_dict = {"soma": 1,"axon": 2,"dendrite": 3,"cell body fiber": 4,"connecting cable": 5,"other": 6}
analyze_sections = np.array(['dendrite', 'connecting cable', 'axon'])

jitter_strengths = np.logspace( np.log10(0.1), np.log10(1000), 30) * 1000/8
rhoM_rhoI = (2000 / 40) * 10**4# https://link.springer.com/article/10.1007/BF00161091 - HS cell
#rhoM_rhoI = (28000 / 150) * 10**4 # https://www.nature.com/articles/s41586-022-04428-3 - T4

neuron_quality = pd.read_csv(home_dir + '/saved_data/neuron_quality.csv')
neuron_quality_np = neuron_quality.to_numpy()
unique_neuron_types, counts = np.unique( neuron_quality_np[:,1], return_counts = True )
unique_neuron_types = unique_neuron_types[ counts >= 10 ]
counts = counts[ counts >= 10 ] # only use LC neurons with at least 10 neurons
analyze_neurons = np.append( unique_neuron_types[-3:], unique_neuron_types[:-3] )
analyze_neurons = list( analyze_neurons[ analyze_neurons != 'LC14' ] )

vis_neurons = ['LC4', 'LC9', 'LC12']

LC_colors = {'LC4': [0.904705882352941, 0.191764705882353, 0.198823529411765],
             'LC6': [0.294117647058824, 0.544705882352941, 0.749411764705882],
             'LC9': [0,0,0],
             'LC10':[255/255, 215/255, 0],
             'LC11':[0.865000000000000, 0.811000000000000, 0.433000000000000],
             'LC12':[0.685882352941177, 0.403529411764706, 0.241176470588235],
             'LC13':[0.971764705882353, 0.555294117647059, 0.774117647058824],
             'LC15':[0.904705882352941, 0.191764705882353, 0.198823529411765],
             'LC16':[0.294117647058824, 0.544705882352941, 0.749411764705882],
             'LC17':[0,0,0],
             'LC18':[255/255, 215/255, 0],
             'LC20':[0.865000000000000, 0.811000000000000, 0.433000000000000],
             'LC21':[0.685882352941177, 0.403529411764706, 0.241176470588235],
             'LC22':[0.971764705882353, 0.555294117647059, 0.774117647058824],
             'LC24':[0.904705882352941, 0.191764705882353, 0.198823529411765],
             'LC25':[0.294117647058824, 0.544705882352941, 0.749411764705882],
             'LC26':[0,0,0],
             'LC27':[255/255, 215/255, 0],
             'LC29':[0.865000000000000, 0.811000000000000, 0.433000000000000],
             'LC31':[0.685882352941177, 0.403529411764706, 0.241176470588235],
             'LC36':[0.971764705882353, 0.555294117647059, 0.774117647058824]}


# green - [0.371764705882353, 0.717647058823529, 0.361176470588235]
# purple - 
# orange - [1, 0.548235294117647, 0.100000000000000]# LC9
LC_markers ={'LC4' :'o','LC6' :'o','LC9' :'o','LC10':'o','LC11':'o','LC12':'o','LC13':'o',
             'LC15':'^','LC16':'^','LC17':'^','LC18':'^','LC20':'^','LC21':'^','LC22':'^',
             'LC24':'P','LC25':'P','LC26':'P','LC27':'P','LC29':'P','LC31':'P','LC36':'P'}
LC_linestyles = {'LC4' : 'solid','LC6' : 'solid','LC9' : 'solid','LC10':'solid' ,'LC11':'solid' ,'LC12':'solid' ,'LC13':'solid' ,
                 'LC15':'dashed','LC16':'dashed','LC17':'dashed','LC18':'dashed','LC20':'dashed','LC21':'dashed','LC22':'dashed',
                 'LC24':'dotted','LC25':'dotted','LC26':'dotted','LC27':'dotted','LC29':'dotted','LC31':'dotted','LC36':'dotted'}

# purple
# gold

section_colors = {'axon': [0.371764705882353, 0.717647058823529, 0.361176470588235],
                  'dendrite': [152/255, 78/255, 163/255],
                  'connecting cable': [1, 0.548235294117647, 0.100000000000000],
                  'cell body fiber': [0, 0.30196078431372547, 0.25098039215686274],
                  'soma': [0, 0.6941176470588235, 0.6901960784313725],
                  'other': [0.7686274509803922, 0.4980392156862745, 0.19607843137254902]}
synapse_colors = {'pre': [0.904705882352941, 0.191764705882353, 0.198823529411765],
                  'post': [55/255, 126/255, 184/255]}

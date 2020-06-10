from magpie import MagpieServer
import pandas as pd
m = MagpieServer()
data = pd.read_csv('data/Supercon_data.csv')
magpie_features = m.generate_attributes('gfa',list(data['name']))

magpie_features['name'] = list(data['name'])
magpie_features.to_csv('data/Magpie_data.csv')

print (magpie_features)
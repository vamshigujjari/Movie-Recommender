import numpy as np
import scipy
from lightfm.datasets import fetch_movielens
from lightfm import LightFM
import csv

data = fetch_movielens(min_rating=4.0)

print (repr(data['train']))
print (repr(data['test']))


model = LightFM(loss='warp')


model.fit(data['train'], epochs=450, num_threads=2)



def sample_recommender(model,data, user_ids):
	n_users, n_items = data['train'].shape


	for user_id in user_ids:
		known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

		scores = model.predict(user_id, np.arange(n_items))

		top_scores = data['item_labels'][np.argsort(-scores)]
		
		
		with open('recommendations.csv', 'w') as  f:
			writer = csv.DictWriter(f, fieldnames=['Known Positives'])
			writer.writeheader()
			for i in known_positives[:5]:
				writer.writerow({'Known Positives':i.encode('UTF-8')})
			
			for i in range(1):
				writer.writerow({'Known Positives':'*****'})
			
			writer = csv.DictWriter(f, fieldnames=['Recommended Movies'])
			writer.writeheader()
			for i in top_scores[:5]:
				writer.writerow({'Recommended Movies':i.encode('UTF-8')})

id = raw_input("Enter the UserID:>")
sample_recommender(model,data, id)
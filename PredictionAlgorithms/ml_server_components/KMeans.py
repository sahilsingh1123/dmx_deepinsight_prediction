import numpy as np
import time
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def perform_k_means(data, no_of_clusters):
    start_time = time.time()
    x = np.array(data)
    no_of_rows = len(x)
    json_response = {}
    if no_of_rows > 0:
        print('user no of clusters = ' + str(no_of_clusters))
        print('no of rows = ' + ' = ' + str(no_of_rows))
        if no_of_clusters is None or no_of_clusters < 1:
            if no_of_rows < 10:
                no_of_clusters = no_of_rows
            else:
                pca = PCA()
                pca.fit(x)
                ratio = 0
                for explained_ratio in pca.explained_variance_ratio_:
                    ratio += explained_ratio
                    no_of_clusters += 1
                    if ratio > 0.9:
                        break
        if no_of_rows < no_of_clusters:
            no_of_clusters = no_of_rows
        if no_of_clusters < 1:
            no_of_clusters = no_of_rows if no_of_rows < 3 else 3
        elif no_of_clusters > 10:
            no_of_clusters = 10
        print('no of clusters = ' + str(no_of_clusters))
        x = np.nan_to_num(x)
        print("Nan to number conversion successfull")
        k_means = KMeans(n_clusters=no_of_clusters, random_state=5).fit(x)
        print('labels size = ' + str(len(k_means.labels_)))
        cluster_list = []
        for label in k_means.labels_:
            cluster_list.append("Cluster" + str(label + 1))
        # print(cluster_list)
        json_response = {'run_status': 'success', 'clusters': cluster_list, 'execution_time': time.time() - start_time}
    else:
        json_response = {'run_status': 'success', 'clusters': ["Cluster1"], 'execution_time': time.time() - start_time}
    # print(json.dumps(json_response))
    #return str(json.dumps(json_response)).encode('utf-8')
    return json_response
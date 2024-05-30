
from sklearn.metrics.pairwise import cosine_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
import json
from tqdm import tqdm

def parse_nl_similarity(filename):
    all_nl = []
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)  # list[dict{}]
    f.close()
    for dict in tqdm(data, desc="load all nl into a list"):
        all_nl.append(dict["nl"])
    # 转换为TF-IDF特征向量
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(all_nl)

    # 使用DBSCAN算法进行聚类
    dbscan = DBSCAN(eps=0.3, min_samples=2, metric="cosine")
    dbscan.fit(tfidf)
    # distances = cosine_distances(tfidf).tolist()
    # print(type(distances))
    similar_groups=[]
    # 输出聚类结果到列表[[]],里面的元素是索引号
    for i in tqdm(set(dbscan.labels_),desc="clustering all similar nl"):
        # print(f'Cluster {i}:')
        similar_group=[]
        for j in range(len(all_nl)):
            if dbscan.labels_[j] == i:
                similar_group.append(j)
        if i!=-1:
            similar_groups.append(similar_group)
    return similar_groups
# print(parse_nl_similarity("cosqa-train.json"))
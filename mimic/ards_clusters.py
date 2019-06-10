import psycopg2
import logging, os, sys

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
import gensim
from sklearn.cluster import KMeans
import numpy as np
from collections import Counter

NUM_CLUSTERS = os.environ.get("NUM_CLUSTERS")
EMBEDDING_SIZE = os.environ.get("EMBEDDING_SIZE")
DB_NAME = os.environ.get("DB_NAME")
DB_USER = os.environ.get("POSTGRES_USER")
DB_PASSWORD = os.environ.get("POSTGRES_PASSWORD")
ICD_EMBEDDINGS_FILE = os.environ.get("ICD_EMBEDDINGS_FILE")
CLUSTERS_OUTPUT_FILE = os.environ.get("CLUSTERS_OUTPUT_FILE")

if None in (DB_NAME, DB_USER, DB_PASSWORD, ICD_EMBEDDINGS_FILE, CLUSTERS_OUTPUT_FILE, NUM_CLUSTERS, EMBEDDING_SIZE):
    print(
        "Please set the following env vars: DB_NAME,DB_USER,DB_PASSWORD,ICD_EMBEDDINGS_FILE,CLUSTERS_OUTPUT_FILE,NUM_CLUSTERS,EMBEDDING_SIZE")
    sys.exit(1)


def get_ards_patients_all_distinct_icds():
    query = """
select distinct icd9_code from mimiciii.diagnoses_icd where  
(hadm_id in (select distinct(hadm_id) from mimiciii.diagnoses_icd where icd9_code in ('51881','51882','51884','51851','51852','51853','5184','5187','78552','99592'))
and  hadm_id in (select distinct(hadm_id) from mimiciii.procedures_icd where icd9_code in ('9670','9671','9672'))
and  hadm_id not in (select distinct(hadm_id) from mimiciii.diagnoses_icd where icd9_code in ('49391','49392','49322','4280')))
"""
    conn = psycopg2.connect("dbname=%s user=%s password=%s" % (DB_NAME, DB_USER, DB_PASSWORD))
    cur = conn.cursor()
    cur.execute(query)
    allicds = []
    for record in cur:
        icd = record[0]
        allicds.append(icd)
    conn.close()
    CODE_VALUES = list(set(allicds))
    return CODE_VALUES


def create_dataset():
    examples = []
    db_index_lookup = []
    conn = psycopg2.connect("dbname=%s user=%s password=%s" % (DB_NAME, DB_USER, DB_PASSWORD))
    try:
        cur = conn.cursor()
        cur2 = conn.cursor()
        cur.execute("""
    select distinct hadm_id from (
    select patients.subject_id, hadm_id, expire_flag from 
     mimiciii.admissions
    join mimiciii.patients  on mimiciii.patients.subject_id =  mimiciii.admissions.subject_id
    where  DATE_PART('year', admittime)  -  DATE_PART('year', dob)  > 17
    and  hadm_id in (select distinct(hadm_id) from mimiciii.diagnoses_icd where icd9_code in ('51881','51882','51884','51851','51852','51853','5184','5187','78552','99592'))
    and  hadm_id in (select distinct(hadm_id) from mimiciii.procedures_icd where icd9_code in ('9670','9671','9672'))
    and  hadm_id not in (select distinct(hadm_id) from mimiciii.diagnoses_icd where icd9_code in ('49391','49392','49322','4280'))
    ) a 
    """)
        for record in cur:
            example = [0] * len(CODE_VALUES)
            enc_id = record[0]
            query = """
                select icd9_code from mimiciii.diagnoses_icd where hadm_id=%s
                """ % enc_id
            icds_for_encounter = []
            cur2.execute(query)
            for r in cur2:
                icd = r[0]
                if icd:
                    icds_for_encounter.append(icd)
            for i in icds_for_encounter:
                if i in CODE_VALUES:
                    ind = CODE_VALUES.index(i)
                    example[ind] = 1
            toaverage = []
            for i, code in enumerate(CODE_VALUES):
                if example[i] == 1 and code in model:
                    toaverage.append(model[code])
            value_to_append = np.average(np.array(toaverage), axis=0).tolist()
            if not isinstance(value_to_append, float):
                examples.append(value_to_append)
                db_index_lookup.append(enc_id)
    finally:
        conn.close()
    return examples, db_index_lookup


def icdcode_to_desc():
    conn = psycopg2.connect("dbname=%s user=%s password=%s" % (DB_NAME, DB_USER, DB_PASSWORD))
    code_to_description = {}
    query = """
    select distinct icd9_code, long_title from mimiciii.d_icd_diagnoses
    """
    cur = conn.cursor()
    cur.execute(query)
    for record in cur:
        code, desc = record
        if desc:
            code_to_description[code] = desc
    conn.close()
    return code_to_description


def get_all_mortalities():
    conn = psycopg2.connect("dbname=%s user=%s password=%s" % (DB_NAME, DB_USER, DB_PASSWORD))
    query = """
    select a.hadm_id, expire_flag from mimiciii.admissions a, mimiciii.patients p where p.subject_id=a.subject_id and expire_flag=1
    """
    cur = conn.cursor()
    cur.execute(query)
    mortalities = set()
    for record in cur:
        hadm_id, expire_flag = record
        if expire_flag == 1:
            mortalities.add(hadm_id)
    conn.close()
    return mortalities


def reset_clusters():
    conn = psycopg2.connect("dbname=%s user=%s password=%s" % (DB_NAME, DB_USER, DB_PASSWORD))
    cur = conn.cursor()
    q = 'update mimiciii.diagnoses_icd set cluster=-1'
    cur.execute(q)
    conn.commit()
    conn.close()


def set_clusters(db_index_lookup, kmns):
    conn = psycopg2.connect("dbname=%s user=%s password=%s" % (DB_NAME, DB_USER, DB_PASSWORD))
    cur = conn.cursor()
    for i, l in enumerate(kmns.labels_):
        q = 'update mimiciii.diagnoses_icd set cluster=%s where hadm_id=%s' % (l, db_index_lookup[i])
        cur.execute(q)
        conn.commit()
    conn.close()


def compute_cluster_mortality(cluster):
    hadm_ids = set()
    query = """
select distinct hadm_id from  mimiciii.diagnoses_icd where cluster=%s
""" % cluster
    conn = psycopg2.connect("dbname=mimic user=postgres password=emala75")
    cur = conn.cursor()
    cur.execute(query)
    for record in cur:
        hadm_ids.add(record[0])
    conn.close()
    return float(len(hadm_ids.intersection(mortalities))) / float(len(hadm_ids))


def lookup(code):
    return code_to_description.get(code, None)


def print_cluster_centroids(cluster_centers, cluster_index, model):
    output = ""
    cluster_center = cluster_centers[cluster_index]
    icd_embedding_center = cluster_center[0:EMBEDDING_SIZE]
    result = model.similar_by_vector(icd_embedding_center, topn=20)
    output += "\nMoratlity=%s\n" % compute_cluster_mortality(cluster_index)
    output += "\nICD centroids\n"
    for code, similarity in result:
        output += (u'\n%s, %s, %s\n' % (code, lookup(code), similarity))
    return output


def output_results(kmns, model,cluster_centers):
    with open(CLUSTERS_OUTPUT_FILE, "w") as outsummary:
        for cluster in range(NUM_CLUSTERS):
            c = Counter(list(kmns.labels_))
            outsummary.write(u"\n\nCluster %s, total encounters %s\n" % (cluster, c[cluster]))
            result = print_cluster_centroids(cluster_centers, cluster, model)
            outsummary.write(result)


if __name__ == '__main__':
    CODE_VALUES = get_ards_patients_all_distinct_icds()
    model = gensim.models.KeyedVectors.load(ICD_EMBEDDINGS_FILE, mmap='r')
    examples, db_index_lookup = create_dataset()
    code_to_description = icdcode_to_desc()
    mortalities = get_all_mortalities()
    X = np.array(examples)
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=0, max_iter=1000).fit(X)
    reset_clusters()
    set_clusters(db_index_lookup, kmeans)
    cluster_centers = kmeans.cluster_centers_
    output_results(kmeans, model,cluster_centers)

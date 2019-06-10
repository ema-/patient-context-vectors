import psycopg2
import random
import os, logging, sys

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from gensim.models import Word2Vec

DB_NAME = os.environ.get("DB_NAME")
DB_USER = os.environ.get("POSTGRES_USER")
DB_PASSWORD = os.environ.get("POSTGRES_PASSWORD")
ICD_EMBEDDINGS_FILE = os.environ.get("ICD_EMBEDDINGS_FILE")

if None in (DB_NAME,DB_USER,DB_PASSWORD,ICD_EMBEDDINGS_FILE):
    print("Please set the following env vars: DB_NAME,DB_USER,DB_PASSWORD,ICD_EMBEDDINGS_FILE")
    sys.exit(1)

distinct_hadmid_query= """
select distinct hadm_id from mimiciii.diagnoses_icd
"""

training=[]
conn = psycopg2.connect("dbname=%s user=%s password=%s" % (DB_NAME, DB_USER, DB_PASSWORD))
try:
    cur = conn.cursor()
    cur2 = conn.cursor()
    cur.execute(distinct_hadmid_query)
    for record in cur:
        encounter_id=record[0]
        query = """
        select
        icd9_code
        from mimiciii.diagnoses_icd where hadm_id='%s'
        """%encounter_id
        example=[]
        cur2.execute(query)
        for r in cur2:
            icd=r[0]
            example.append(icd)
        icdcodes_length = len(example)
        for i in range(int(icdcodes_length/2)):
            shuffled_example=random.sample(example, icdcodes_length)
            training.append(shuffled_example)
finally:
    conn.close()


model = Word2Vec(training, size=50, window=4, min_count=1, negative=25)
word_vectors = model.wv
word_vectors.save(ICD_EMBEDDINGS_FILE)


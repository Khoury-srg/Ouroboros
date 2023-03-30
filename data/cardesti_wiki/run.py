import mysql.connector
import random

DATASET_SIZE = 100000

def query_db(db, sql):
    db.execute(sql)
    ret = db.fetchall()
    return ret

def query_min_max(db, key):
    temp = "SELECT min(%s),max(%s) FROM page"
    ret = query_db(db, temp % (key, key))
    return ret[0][0], ret[0][1]

def gen_one(db, keys, min_max):
    input_vec = []
    sql = "SELECT count(*) FROM page WHERE "
    for key in keys:
        val1 = random.randint(min_max[key][0], min_max[key][1])
        val2 = random.randint(min_max[key][0], min_max[key][1])
        low = min(val1, val2)
        high = max(val1, val2)

        sql += "(%s >= %d AND %s <= %d) AND " % (key, low, key, high)
        input_vec.append(low)
        input_vec.append(high)

    sql += "TRUE"
    print(sql)
    ret = query_db(db, sql)
    input_vec.append(ret[0][0])
    return input_vec, sql

def dump2file(path, keys, vecs):
    with open(path, "w") as f:
        header = ""
        for key in keys:
            header += "%s_low,%s_high," % (key, key)
        f.write(header+"num_rows,\n")

        for vec in vecs:
            line = ""
            for val in vec:
                line += str(val) + ","
            f.write(line+"\n")

def main():
    mydb = mysql.connector.connect(
          host="localhost",
          user="admin",
          password="wiki123",
          database="wiki"

    )
    db = mydb.cursor()

    keys = [
        "page_latest",
        "page_len",
    ]
    min_max = {}
    for key in keys:
        min_v, max_v = query_min_max(db, key)
        min_max[key] = [min_v,max_v]
        print(key, min_v, max_v)

    random.seed(12345)

    output = []
    sqls = []
    for i in range(DATASET_SIZE):
        vec, sql = gen_one(db, keys, min_max)
        output.append(vec)
        sqls.append(sql)

    dump2file("cardesti_wiki.csv", keys, output)
    dump2file("cardesti_wiki.sql", keys, sqls)


if __name__ == "__main__":
    main()




"""
  [YES]`page_latest` int(8) unsigned NOT NULL DEFAULT '0',
  [YES]`page_len` int(8) unsigned NOT NULL DEFAULT '0',
  [OK]`page_is_redirect` tinyint(1) unsigned NOT NULL DEFAULT '0',
  [OK]`page_is_new` tinyint(1) unsigned NOT NULL DEFAULT '0',
  [NO] `page_id` int(8) unsigned NOT NULL AUTO_INCREMENT,
  [NO]`page_namespace` int(11) NOT NULL DEFAULT '0',
  [NO]`page_title` varbinary(255) NOT NULL DEFAULT '',
  [NO]`page_restrictions` varbinary(255) NOT NULL,
  [?]`page_random` double unsigned NOT NULL DEFAULT '0',
  [NO]`page_touched` varbinary(14) NOT NULL DEFAULT '',
  [NO]`page_links_updated` varbinary(14) DEFAULT NULL,
  [OK]`page_no_title_convert` tinyint(1) NOT NULL DEFAULT '0',
  [NO]`page_content_model` varbinary(32) DEFAULT NULL,
  [NO]`page_lang` varbinary(35) DEFAULT NULL,
"""

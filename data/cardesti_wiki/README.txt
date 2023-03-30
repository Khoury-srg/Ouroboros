wiki dataset
---

* this is a dataset for cardinality estimation of wikipedia.

* Cardinality estimation is to predict the number of return database rows for a given SQL statement.

* In this wkik dataset, the SQL is like:
    "SELECT count(*) FROM page WHERE (page_latest >= 3531223 AND page_latest <=
      6182572) AND (page_len >= 21326 AND page_len <= 626293) AND TRUE"
  (see the SQLs of the eintire dataset in "cardesti_wiki.sql")

* The NN's input is an 4-element vector, for example:
  [3531223,6182572,21326,626293].
  (This vector represents the above SQL.)

  ** four elements are two pairs, meaning the lower and the upper value of
  two columns in the table (namely, page_latest and page_len)

  ** here are the min and max of these four columns in the wiki database:
    page_latest 36363 6414244
    page_len 0 1567349

  ** NN's output is an integer (i.e., the number of return rows for this SQL)

  ** the wiki dataset is the file "cardesti_wiki.csv".
     each entry is like:
    "3531223,6182572,21326,626293,91".
    The first 4 elements are the input, the last element is the output.


specification
---

* we will verify relative spec for this case.

* I don't know how hard verifying relative spec is. I list some possible spec
  below. Feel free to simplify the specs.

* (1) fix 3 out of 4 input elements, for example:
      [3531223,X,21326,626293].
      Given two inputs 
        X0 = [3531223,x0,21326,626293] and
        X1 = [3531223,x1,21326,626293],
      the spec is:
        if x0 < x1, then NN(X0) < NN(X1)


* (2) give 3 input elements some range, for example:
      [3531223+-1000,X,21326+-1000,626293+-1000].  (where "A+-B" means a range [A-B, A+b])
      Given two inputs 
        X0 = [3531223+-1000,x0,21326+-1000,626293+-1000] and
        X1 = [3531223+-1000,x1,21326+-1000,626293+-1000]
      the spec is:
        if x0 < x1, then NN(X0) < NN(X1)

* (3) do not give range to the other 3 elements
      [ANY,X,ANY,ANY]
      Given two inputs 
        X0 = [ANY,x0,ANY,ANY] and
        X1 = [ANY,x1,ANY,ANY]
      the spec is:
        if x0 < x1, then NN(X0) < NN(X1)



other details
---

* database table:

CREATE TABLE `page` (
  `page_id` int(8) unsigned NOT NULL AUTO_INCREMENT,
  `page_namespace` int(11) NOT NULL DEFAULT '0',
  `page_title` varbinary(255) NOT NULL DEFAULT '',
  `page_restrictions` varbinary(255) NOT NULL,
  `page_is_redirect` tinyint(1) unsigned NOT NULL DEFAULT '0',
  `page_is_new` tinyint(1) unsigned NOT NULL DEFAULT '0',
  `page_random` double unsigned NOT NULL DEFAULT '0',
  `page_touched` varbinary(14) NOT NULL DEFAULT '',
  `page_links_updated` varbinary(14) DEFAULT NULL,
  `page_latest` int(8) unsigned NOT NULL DEFAULT '0',
  `page_len` int(8) unsigned NOT NULL DEFAULT '0',
  `page_no_title_convert` tinyint(1) NOT NULL DEFAULT '0',
  `page_content_model` varbinary(32) DEFAULT NULL,
  `page_lang` varbinary(35) DEFAULT NULL,
  PRIMARY KEY (`page_id`),
  UNIQUE KEY `name_title` (`page_namespace`,`page_title`),
  KEY `page_random` (`page_random`),
  KEY `page_len` (`page_len`),
  KEY `page_redirect_namespace_len` (`page_is_redirect`,`page_namespace`,`page_len`)
) ENGINE=InnoDB AUTO_INCREMENT=678737 DEFAULT CHARSET=binary;


* number of rows: 491374

* column, min, max
  page_latest 36363 6414244
  page_len 0 1567349
  page_is_redirect 0 1
  page_is_new 0 1

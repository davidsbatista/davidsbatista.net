---
layout: post
title: Pandas DataFrame cheat-sheet
date: 2022-12-07 00:00:00
tags: pandas cheat-sheet reference-post
categories: [blog]
comments: true
disqus_identifier: 20221207
preview_pic: /assets/images/2022-12-07-Zoo-Berlin-zoo-berlin-panda-plant-eye-1643703-pxhere.com.jpg
---

I will just use a blog post to keep track of typical operations I need to do over pandas DataFrame. I came to realise and I need to do them whenever I need to explore DataFrames, but I keep forgetting them.


## __Columns__

#### Remove columns from a DataFrame

    df_rels_in_scope.drop(['ent1','ent2','label'], axis=1)

#### Rename columns names with a dictionary

	df = df.rename(columns={'rel_type':'arg_ent_type'})

#### Add a new column by applying a function to other columns

	def replace_arg(arg_type_str):
	if arg_type_str is None:
	    return None
	else:
	    return re.sub(r'[0-9]+','',arg_type_str)

	norm_arg = df_bookings_only.apply(lambda row: replace_arg(row.arg_type), axis=1)
	df_bookings_only.insert(len(df_bookings_only.columns), "arg_type_norm", norm_arg)

#### Sort after a groupby count

	df_entities[['entity_text','entity_type']].groupby('entity_type').count().sort_values(by='entity_text', ascending=False)


#### Select rows whose column value equals some value

	df.loc[df['column_name'] == some_value]


## __Display__

#### Number of rows to display

	pd.set_option('display.max_rows', 1500)

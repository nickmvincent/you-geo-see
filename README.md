In addition to providing automated processing of key components of the SERP (e.g. the links, their position, and their destination) our software is also designed to save the raw HTML of the SERPs. This allowed for more detailed human verficiation, i.e. we make sure that the representation of each SERP in the database matches how a human interprets the SERP. This is an important feature, as there is no guarantee that SERPs will keep the same content format or styles, which the software uses to distinguish different types of results. Therefore, it is important that human validation is performed when exploring new queries or after substantial time has passed, even after the software has been extensively validated for other queries.”


# Warnings
CAreful about urban-rural vs urban_rural in main.py
Doing urban_rural throws NO ERROR - it just randomly samples from the whole distribution. OK for testing but not for the experiments!


# To Run all Analyses
set link_types = ['results']
python analysis.py --db dbs\population_weighted_40_all.db "dbs\2018-01-18_population_weighted_40_extra.db" --plot

# What's up with importance_df.csv? Why so large?
`importance_df.csv` has all the "general results" (i.e. not geographical comparisons) in pure long-form
This means there is there is one row for every SERP we looked at, for every domain we observed, for every metric we looked at
If you check at `analysis.py` under the `# ANCHOR: MELT` comment, you can see how it's created.
There four metrics of interest, "domain_appears" (true or false), "domain_frac" (a fraction), "domain_rank" (int), and "domain_count" (int)
Each of these metrics is repeated for the FULL_PAGE and the TOP_THREE subset.
Furthermore, each row must be duplicated for each query category.

There are 5353921 rows.

# How can I see the geographic comparisons?


# Warnings relate to running comparisons
Don't run ALL comparisons for ALL databses - for instance running a high-income vs. low-income test on the urban-rural database.
Use collect_comparisons.py to get a manageable summary of all the tests

# Sanity Checking Analysis
There's a lot of data manipulating going in analysis.py and then plotters.py
One easy to sanity check some results is to manually inspect the serp_df.csv file that's written in the subdirectories of outputs/

For instance, if we want to double check the fact that our "importance_plot" suggets that medical queries appear in about 45% of all first-page results, we can open serp_df.describe().csv in `2018-01-18_population_weighted_40_extra.db__med_sample_first_20/` and verify that indeed, Wikipedia appeared in 360/800 = 0.45 of all pages.
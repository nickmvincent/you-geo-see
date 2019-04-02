# You-Geo-See, or the analysis code for "Measuring the Importance of User-Generated Content to Search Engines"

In addition to providing automated processing of key components of the SERP (e.g. the links, their position, and their destination) our software is also designed to save the raw HTML of the SERPs. This allowed for more detailed human verficiation, i.e. we make sure that the representation of each SERP in the database matches how a human interprets the SERP. This is an important feature, as there is no guarantee that SERPs will keep the same content format or styles, which the software uses to distinguish different types of results. Therefore, it is important that human validation is performed when exploring new queries or after substantial time has passed, even after the software has been extensively validated for other queries.”

# Notes for potential readers
Before doing any scraping, please consider reading through the following resources relating to algorthmic auditing:

["Auditing Algorithms:
Research Methods for Detecting Discrimination
on Internet Platforms" - Christian Sandvig, Kevin Hamilton, Karrie Karahalios. & Cedric Langbort](http://www-personal.umich.edu/~csandvig/research/Auditing%20Algorithms%20--%20Sandvig%20--%20ICA%202014%20Data%20and%20Discrimination%20Preconference.pdf)

https://en.wikipedia.org/wiki/Web_scraping

As mentioned in the paper, you may also want to consider using a headless browser-based scraping software.

# To Run all Analyses
in analysis.py, ensure that `link_types = ['results']` (note to self: why isn't this a command line option?)

`python analysis.py --db dbs/population_weighted_40_all.db "dbs/2018-01-18_population_weighted_40_extra.db" --write_long`
This will produce the longform importance_df from which Figure 1 is produced. You can also load it up to do other analyses.

To see counts of our qualitative codes (ugc, corporate, political, journalistic):
`python qual_code.py --db dbs/population_weighted_40_all.db "dbs/2018-01-18_population_weighted_40_extra.db" --count`

# What's up with importance_df.csv? Why so large?
`importance_df.csv` has all the "general results" (i.e. not geographical comparisons) in pure long-form
This means there is there is one row for every SERP we looked at, for every domain we observed, for every metric we looked at
If you check at `analysis.py` under the `# ANCHOR: MELT` comment, you can see how it's created.
The main metrics of interest are "domain_appears" (true or false), (a fraction), "domain_rank" (int), and "domain_count" (int).
You can also compute "domain_frac" or "domain_maps" (mean average precisions) by editing analysis.py.

Each of these metrics is repeated for the FULL_PAGE and the TOP_THREE subset.
Furthermore, each row must be duplicated for each query category. We can verify the long-form df contains each link 6240 times, e.g. `df.domain.value_counts()`.

# How can I see the geographic comparisons?
The statistically significant comparisons are written to fisher_comparisons.csv or ttest_comparisons.csv. All UGC comparisons are in comparisons_df.csv. You can also go through the output/ folders to find a particular comparison, or a comparison for a specific query.

# Warnings relate to running comparisons
Don't run ALL comparisons for ALL databses - for instance running a high-income vs. low-income test on the urban-rural database.
Use collect_comparisons.py to get a manageable summary of all the tests

# Sanity Checking Analysis
There's a lot of data manipulating going in analysis.py and then plotters.py
One easy to sanity check some results is to manually inspect the serp_df.csv file that's written in the subdirectories of outputs/

For instance, if we want to double check the fact that our "importance_plot" suggets that medical queries appear in about 45% of all first-page results, we can open serp_df.describe().csv in `2018-01-18_population_weighted_40_extra.db__med_sample_first_20/` and verify that indeed, Wikipedia appeared in 360/800 = 0.45 of all pages.
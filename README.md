# MTER

A parallel implementation of MTER with derived gradients based on the idea of parameter server. Directly run MTER_tripletensor_tucker.py in folder 'parallel_implementation' to test our MTER model by tunning hyper parameters. The learned models are stored in 'Results'. The processed Yelp dataset is included in yelp_restaurant_recursive_entry_sigir.

The provided training and testing set are split from yelp_recursive.entry and are for testing the algorithm. You can split it to train, test and validation sets with cross validation if you want to conduct serious experiments. 

Please refer to our paper ['Explainable Recommendation via Multi-Task Learning in Opinionated Text Data'](https://dl.acm.org/citation.cfm?id=3210010) for more details.

Feel free to contact me if any questions. Thank you!

Please consider to cite:
```
@inproceedings{Wang:2018:ERV:3209978.3210010,
 author = {Wang, Nan and Wang, Hongning and Jia, Yiling and Yin, Yue},
 title = {Explainable Recommendation via Multi-Task Learning in Opinionated Text Data},
 booktitle = {The 41st International ACM SIGIR Conference on Research \&\#38; Development in Information Retrieval},
 series = {SIGIR '18},
 year = {2018},
 isbn = {978-1-4503-5657-2},
 location = {Ann Arbor, MI, USA},
 pages = {165--174},
 numpages = {10},
 url = {http://doi.acm.org/10.1145/3209978.3210010},
 doi = {10.1145/3209978.3210010},
 acmid = {3210010},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {explainable recommendation, multi-task learning, sentiment analysis, tensor decomposition},
} 
```

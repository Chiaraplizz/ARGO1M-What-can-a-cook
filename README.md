
## What can a cook in Italy teach a mechanic in India? Action Recognition Generalisation Over Scenarios and Locations

This is the official implementation for the paper "What can a cook in Italy teach a mechanic in India? Action Recognition Generalisation Over Scenarios and Locations" 

## BibTeX

If you use this repository, please cite: 

<code>@inproceedings{jermsurawong2015predicting,
  title={Predicting the structure of cooking recipes},
  author={Jermsurawong, Jermsak and Habash, Nizar},
  booktitle={Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing},
  pages={781--786},
  year={2015}}</code>

## Dataset: ARGO1M

We provide both the <a href="http://en.wikipedia.org/wiki/Main_Page">CSV</a> files with the corresponding splits, which are <a href="http://en.wikipedia.org/wiki/Main_Page">training.csv</a>, <a href="http://en.wikipedia.org/wiki/Main_Page">seen.csv</a>,  <a href="http://en.wikipedia.org/wiki/Main_Page">japanese_cooking.csv</a>, <a href="http://en.wikipedia.org/wiki/Main_Page">rwanda.csv</a>, <a href="http://en.wikipedia.org/wiki/Main_Page">mechanic.csv</a>, <a href="http://en.wikipedia.org/wiki/Main_Page">sport.csv</a>. <a href="http://en.wikipedia.org/wiki/Main_Page">knitting.csv</a>, <a href="http://en.wikipedia.org/wiki/Main_Page">mechanic_colombia.csv</a>, <a href="http://en.wikipedia.org/wiki/Main_Page">sport_colombia.csv</a>. 
'video_source', 'device', 'scenario', 'clean_scenario', 'label',
       'narration', 'action_start_feature_idx', 'action_end_feature_idx',
       'source_idx', 'scenario_idx', 'scenario_idx_multi', 'verb', 'noun',
       'timestamp', 'timeframe', 'label_idx', 'device_idx']

Those contain the following entries: 

- <code>uid</code>: uid of the video clip; 
- <code>scenario_idx</code>: scenario label;
- <code>location_idx</code>: location label; 
- <code>label</code>: action label;
- <code>timestamp</code>: starting timestamp;
- <code>timeframe</code>: starting timeframe;
- <code>narration</code>: narration; 
- <code>start_feature_idx</code>: starting feature index;
- <code>end_feature_idx</code>: endinf feature index.

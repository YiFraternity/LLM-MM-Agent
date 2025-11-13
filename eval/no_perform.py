import os

# unfinished BestPaper inverted index
"""
bestpaper_path = 'eval/output/bestpaper/2_inverted_index'
bestpaper_finished = os.listdir(bestpaper_path)
bestpaper_finished = [_.split('.')[0] for _ in bestpaper_finished]

raw_path = 'MMBench/CPMCM/BestPaper'
raw_finished = os.listdir(raw_path)
unfinish = set(raw_finished) - set(bestpaper_finished)
unfinish = sorted(list(unfinish))
unfinish = [f'{raw_path}/{_}' for _ in unfinish]
for _ in unfinish:
    print(f'{_} \\')
"""

# unfinished BestPaper
problem_path = 'MMBench/CPMCM/problem'
problem_finished = os.listdir(problem_path)
problem_finished = [_.split('.')[0] for _ in problem_finished]

raw_path = 'MMBench/CPMCM/BestPaper'
raw_finished = os.listdir(raw_path)
unfinish = set(problem_finished) - set(raw_finished)
unfinish = sorted(list(unfinish))
for _ in unfinish:
    print(f'best_paper_tex_clean_format/{_}/1_cleaned.tex \\')
    print(f'best_paper_tex_clean_format/{_}/2_cleaned.tex \\')
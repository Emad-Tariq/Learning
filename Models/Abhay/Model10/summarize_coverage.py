"""Render the separate coverage-input follow-up without overwriting pure results."""
from pathlib import Path
p=Path(__file__).resolve().parent
source=(p/'summarize.py').read_text(encoding='utf-8')
old="HERE=Path(__file__).resolve().parent; OUT=HERE/'local_runs/difficult_2052'"
new="BASE=Path(__file__).resolve().parent; HERE=BASE/'coverage_followup'; HERE.mkdir(exist_ok=True); OUT=BASE/'local_runs/coverage_difficult_2052'"
assert source.count(old)==1
source=source.replace(old,new)
source=source.replace("'difficult25':'25% difficult','difficult50':'50% difficult'", "'difficult25':'25% + detector coverage','difficult50':'50% + detector coverage'")
source=source.replace("'# Model 10: difficult-episode full-pipeline retraining'", "'# Model 10 follow-up: detector coverage inputs and difficult training'")
source=source.replace('Full-pipeline retraining with difficult episodes</h1>', 'Follow-up: detector coverage inputs and difficult training</h1>')
source=source.replace('`local_runs/difficult_2052`','`../local_runs/coverage_difficult_2052`')
anchor="plt.rcParams.update("
assert source.count(anchor)==1
source=source.replace(anchor,"limits+=' This follow-up adds 23 observable candidate-coverage inputs to the trained tree unknown detector in the experimental variants only. It was designed after the episode-only outcome; it is exploratory and separate from the requested pure-mixture comparison.'\n"+anchor)
exec(compile(source,str(p/'summarize_coverage.py'),'exec'))

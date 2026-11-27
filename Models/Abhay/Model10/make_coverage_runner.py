"""Generate the follow-up runner with explicit, reviewable differences."""
from pathlib import Path
p=Path(__file__).resolve().parent
source=(p/'difficult_eval.py').read_text()
changes=[
 ('from difficult_protocol import hard_positions,difficult_episode,slices',
  'from difficult_protocol import hard_positions,difficult_episode,slices\nimport coverage_features as coverage'),
 ("default=HERE/'local_runs/difficult_2052'", "default=HERE/'local_runs/coverage_difficult_2052'"),
 ("'hard_schedules':", "'followup':'23 candidate-coverage features supplied to the tree unknown detector only',\n         'hard_schedules':"),
 ("paths=[HERE/'difficult_protocol.py',HERE/'difficult_eval.py',", "paths=[HERE/'difficult_protocol.py',HERE/'coverage_train.py',HERE/'coverage_features.py',"),
 ('frames[kind]=protocol.frame(gg,qq,tt)', 'frames[kind]=coverage.frame(gg,qq,tt)'),
 ('b.augment(frame,b.predict(model,frame))', 'b.augment(frame,coverage.predict(model,frame))'),
 ('models[method].append(b.fit(training[method]))', 'models[method].append(coverage.fit(training[method],method))'),
 ('raw=protocol.frame(g,q,t)', 'raw=coverage.frame(g,q,t)'),
 ('s,r=protocol.through_stack(raw,models[method])', 's,r=coverage.through_stack(raw,models[method])'),
]
for old,new in changes:
    assert source.count(old)==1,(old,source.count(old))
    source=source.replace(old,new)
source=source.replace('"""Full three-stage retraining: ordinary, 25%, and 50% difficult episodes."""',
                      '"""Separate follow-up: difficult training plus detector coverage inputs."""')
(p/'coverage_train.py').write_text(source,encoding='utf-8')
print('Generated coverage_train.py; original difficult_eval.py is unchanged.')

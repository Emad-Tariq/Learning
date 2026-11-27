"""Standalone experiment report and static comparison figure."""
from pathlib import Path
import json,html,base64
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
RUN=HERE/'local_runs/large_2052'

def main():
    report=json.loads((RUN/'metrics.json').read_text())
    checks=json.loads((RUN/'independent_checks.json').read_text())
    cfg=json.loads((RUN/'config.json').read_text())
    summary=pd.DataFrame(report['slices'])
    diag=pd.read_csv(RUN/'training_episode_diagnostics.csv')
    ranges=diag.groupby(['stage','species']).agg(original_min=('original_gallery','min'),
        original_max=('original_gallery','max'),large_min=('large_gallery','min'),
        large_max=('large_gallery','max'),gallery_identities_min=('gallery_identities','min'),
        gallery_identities_max=('gallery_identities','max'),query_min=('queries','min'),query_max=('queries','max')).reset_index()
    core=summary[summary['slice'].isin(['overall','known','unknown','hyena','leopard'])]
    predictions=pd.read_csv(RUN/'validation_predictions.csv')
    joint=predictions.groupby(['species','unknown'])[['baseline_rr5','large_rr5']].mean().reset_index()
    joint['delta']=joint.large_rr5-joint.baseline_rr5
    joint.to_csv(RUN/'species_status_diagnostics.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(12,4.7))
    xx=np.arange(len(core));axes[0].bar(xx-.18,core.baseline_map5,.36,color='#2d6175',label='Unchanged baseline')
    axes[0].bar(xx+.18,core.large_map5,.36,color='#d59039',label='Larger-gallery training')
    axes[0].set(xticks=xx,xticklabels=core['slice'],ylabel='MAP@5',ylim=(0,1.04),title='Same larger-gallery validation')
    axes[0].legend(fontsize=8,loc='lower center',bbox_to_anchor=(.5,1.0),ncol=2,frameon=False)
    axes[0].set_title('Same larger-gallery validation',pad=28)
    colors={'hyena':'#2d6175','leopard':'#d59039'}
    for sp,z in ranges.groupby('species'):
        center=(z.large_min+z.large_max)/2
        axes[1].errorbar(z.stage,center,yerr=[center-z.large_min,z.large_max-center],fmt='o-',capsize=5,color=colors[sp],label=sp+' enlarged')
        axes[1].plot(z.stage,(z.original_min+z.original_max)/2,'o--',alpha=.55,color=colors[sp],label=sp+' original')
    axes[1].set(xticks=[1,2,3],xlabel='Training stage',ylabel='Gallery reference photos',title='Progressive gallery growth')
    axes[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(HERE/'comparison.png',dpi=170);plt.close(fig)
    image=base64.b64encode((HERE/'comparison.png').read_bytes()).decode()
    table=lambda d:d.to_html(index=False,border=0,na_rep='—',float_format=lambda x:f'{x:.6f}')
    bullets=lambda items:'<ul>'+''.join('<li>'+html.escape(str(x))+'</li>' for x in items)+'</ul>'
    verdict='Passed the predeclared conservative local promotion gate.' if report['promotion_gate_passed'] else 'Did not pass the predeclared conservative local promotion gate; review the measured tradeoffs below.'
    overall=summary[summary['slice']=='overall'].iloc[0]
    interpretation=json.loads((HERE/'interpretation.json').read_text())
    body=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Dalvin larger-gallery retraining</title>
    <style>body{{font:16px/1.55 system-ui,sans-serif;color:#172936;max-width:1100px;margin:40px auto;padding:0 22px}}h1,h2{{line-height:1.2;color:#123e52}}h2{{margin-top:34px}}table{{border-collapse:collapse;width:100%;font-size:13px;margin:18px 0}}td,th{{padding:8px 10px;text-align:left;border-bottom:1px solid #dce5ea}}th{{background:#eaf2f5}}img{{width:100%}}.note{{background:#eef5f6;border-left:4px solid #568d9e;padding:14px 18px}}li{{margin-bottom:8px}}.scroll{{overflow-x:auto}}</style>
    <h1>Dalvin: retraining with larger galleries</h1><p>26 November 2026 · Model 12 · Fixed experiment at seed 2052</p>
    <div class="note"><strong>{html.escape(interpretation['headline'])}</strong><p>{html.escape(interpretation['summary'])}</p><p>{verdict}</p></div>
    <h2>Result</h2><p>On identical larger-gallery validation, unchanged baseline MAP@5 is {overall.baseline_map5:.6f}; larger-gallery retraining scores {overall.large_map5:.6f}. Paired difference: {overall.delta:+.6f}. The 95% animal-cluster bootstrap interval is {report['overall_delta_cluster_interval']}. Intervals condition on the fitted models and this previously inspected identity allocation; they do not estimate retraining variance or leaderboard uncertainty.</p>
    <img src="data:image/png;base64,{image}" alt="Validation scores and training gallery sizes">
    <div class="scroll">{table(summary)}</div>
    <h2>Known and unknown results within each species</h2>{table(joint)}
    <h2>Interpretation and next step</h2>{bullets(interpretation['findings'])}{bullets(interpretation['next_steps'])}
    <h2>Exactly what changed</h2><p>The retrained arm uses the same original 36 query episodes (12 per stage), query photos, known/unknown truths, true-identity reference photos, near-duplicate regimes, seed, 15 candidates, 150 SIFT neighbours, PCA/LDA, SIFT matcher, features, 400-tree candidate scorers, 300-tree unknown detectors, stacking and final ranking rule. Only the training galleries are enlarged with eligible background identities. The unknown detector and both candidate scorers are retrained at every stage. No hyperparameters or mixture strengths were selected on validation.</p>
    <p>Stage 1 adds the dedicated background role (the old calibration/metric-fit role), which is never used for query targets. Stage 2 also adds all stage-1 animals; stage 3 also adds all stage-2 animals. Earlier models never see later-stage query animals, even as gallery distractors. All 110 validation animals remain excluded throughout supervised tree fitting. Episode-specific PCA/LDA still uses allowed gallery labels and unlabelled queries at evaluation, exactly as the baseline does.</p>
    <p>The larger galleries add labelled background exposure that the frozen baseline did not receive during fitting. This is a practical larger-gallery retraining comparison, not an equal-gallery-data comparison of architectures. Stage gallery contexts deliberately overlap on background and earlier animals; query identity roles remain disjoint and future-stage identities remain excluded.</p>
    <h2>Training distributions</h2><p>Original and enlarged ranges below use the same query episodes. Query pools remain unchanged.</p>{table(ranges)}
    <h2>Paired validation protocol</h2><p>The exact three enlarged-gallery manifests from the domain-shift analysis are reused. Both models receive identical query pools, gallery images, candidate lists, PCA/LDA and SIFT evidence. Validation unknown animals are absent from the gallery; known query animals have only their allowed reference photos. The frozen baseline reproduces all 1,829 earlier larger-gallery predictions exactly. These are 1,147 distinct photos from 105 queried identities, not 1,829 independent animals.</p>
    <p>Predeclared promotion gate: positive overall and every-episode MAP difference, positive lower animal-cluster interval bound, no decrease in known or unknown MAP, and no decrease in unknown top-one detection. Slice outcomes are reported rather than used to retune the arm.</p>
    <h2>Independent checks</h2>{bullets([f'{k}: {v}' for k,v in checks.items()])}
    <h2>Limits</h2>{bullets(report['limitations'])}<p>No competition test labels, submission generation or leaderboard tuning were used. Model outputs and original checkpoints are preserved separately.</p>
    <h2>Artifacts and reproduction</h2><pre>python Models/Abhay/Model12/check_plan.py
python -u Models/Abhay/Model12/run.py
python Models/Abhay/Model12/verify_matching.py
python Models/Abhay/Model12/verify.py
python Models/Abhay/Model12/make_report.py</pre><p>Saved model: <code>local_runs/large_2052/large_models.joblib</code>. Exact configuration, source hashes, training contexts, episode manifests, predictions, candidate scores, metrics and independent checks are retained beside it. The runner resumes completed stages only when the configuration and source hashes match.</p></html>'''
    (HERE/'report.html').write_text(body,encoding='utf-8')
    print(HERE/'report.html')

if __name__=='__main__':main()

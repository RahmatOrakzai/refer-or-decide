import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
r=pd.read_csv('paper/results/referral_resplits_T-adj.csv')
s=r[r.method=='LTT-bonferroni']
for a in [0.15,0.2]:
    x=s[np.isclose(s.alpha,a)]; c=x[x.auto_rate>0]
    print(a,'certified share',round(len(c)/len(x),3),'auto|cert mean',round(c.auto_rate.mean(),3),'median',round(c.auto_rate.median(),3),'err|cert mean',round(c.err_auto.mean(),3), 'n_cal',x.n_cal.mean().round(1))
o=pd.read_csv('paper/results/baseline_oof_predictions.csv',dtype={'media_id':str})
m=pd.read_csv('data/derived/pitt_master_recordings.csv',dtype={'media_id':str,'participant_id':str})
d=o.merge(m[['media_id','participant_id','visit','label','mmse']],on='media_id')
T="T-adj: roberta-base, pause tokens, adjusted"; F="F-adj: features, demographics+interviewer removed"
w=1/d.groupby('participant_id').media_id.transform('count')
first=d.loc[d.groupby('participant_id').visit.idxmin()]
print('n rec',len(d),'n part',d.participant_id.nunique(),'dementia part share',round(first.label.mean(),3))
for col,nm in [(T,'T-adj'),(F,'F-adj')]:
    print(nm,'rec AUC',round(roc_auc_score(d.label,d[col]),3),'part-weighted',round(roc_auc_score(d.label,d[col],sample_weight=w),3),'first visit',round(roc_auc_score(first.label,first[col]),3))
for nm,dd,ww in [('recording',d,np.ones(len(d))),('weighted',d,w.values),('first visit',first,np.ones(len(first)))]:
    p=(dd[T]>=0.5).values; y=(dd.label==1).values
    se=np.sum(ww*(p&y))/np.sum(ww*y); sp=np.sum(ww*(~p&~y))/np.sum(ww*~y)
    ppv=lambda pr: se*pr/(se*pr+(1-sp)*(1-pr))
    print(nm,'sens',round(se,3),'spec',round(sp,3),'PPV10',round(ppv(.1),3),'PPV5',round(ppv(.05),3))
rng=np.random.default_rng(0); v=[]
for _ in range(200):
    one=d.groupby('participant_id').sample(1,random_state=int(rng.integers(1e9)))
    v.append(roc_auc_score(one.label,one[T]))
print('random one-visit T-adj AUC mean',round(np.mean(v),3),'2.5-97.5',np.percentile(v,[2.5,97.5]).round(3))
c=pd.read_csv('paper/results/classwise/classwise_resplits_T-adj.csv')
for (afn,afp),g in c.groupby(['alpha_fn','alpha_fp']):
    if afn!=0.1: continue
    print('alpha_fp',afp,{k:round(g[k].mean(),3) for k in g.columns if k.startswith(('missed','caught'))}, 'auto_neg',round(g.auto_neg_rate.mean(),4))

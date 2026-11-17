"""Causal snapshot sequences, train-only preprocessing, validation BCE stopping."""
import time
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from run_temporal_models import LSTMClassifier, GRUClassifier, TemporalConvNet
from research_experiment import KEY, ROUND_KEY, LABEL_COL, threshold, score, write_json

def sequences(d,t,cols):
    steps=sorted(d.loc[d.prediction_time_s<=t,'prediction_time_s'].unique())
    last=d[d.prediction_time_s==t].sort_values(ROUND_KEY).copy()
    # Explicit key reindex prevents accidental row-order alignment across cutoffs.
    idx=pd.MultiIndex.from_frame(last[ROUND_KEY])
    frames=[]
    for s in steps:
        frame=d[d.prediction_time_s==s].set_index(ROUND_KEY).reindex(idx)
        assert frame[LABEL_COL].notna().all(), 'Missing earlier metadata snapshot'
        assert np.array_equal(frame[LABEL_COL],last[LABEL_COL])
        frames.append(frame[cols].to_numpy(float))
    return np.stack(frames,axis=1),last,steps

def run(out,rounds,d,families,times):
    torch.set_num_threads(4)
    # CPU deterministic execution avoids incompatible local CUDA driver dependencies.
    for t in times:
        cols=families['baseline']; raw,last,steps=sequences(d,t,cols)
        masks={s:(last.split==s).to_numpy() for s in ['train','val','test']}
        imputer=SimpleImputer(keep_empty_features=True,add_indicator=True)
        imputer.fit(raw[masks['train']].reshape(-1,len(cols)))
        flat=imputer.transform(raw.reshape(-1,len(cols)))
        x=flat.reshape(len(raw),len(steps),-1)
        scaler=StandardScaler().fit(x[masks['train']].reshape(-1,x.shape[-1]))
        x=scaler.transform(x.reshape(-1,x.shape[-1])).reshape(x.shape).astype('float32')
        y=last[LABEL_COL].to_numpy(dtype='float32')
        for name,ctor in [('lstm',LSTMClassifier),('gru',GRUClassifier),('tcn',TemporalConvNet)]:
            tag=f'{name}__baseline__{t}__42'; done=out/'checkpoints'/f'{tag}.json'
            if done.exists(): continue
            print('TEMPORAL',tag,flush=True); torch.manual_seed(42); np.random.seed(42)
            net=ctor(x.shape[-1]); opt=torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=1e-4)
            lossfn=torch.nn.BCEWithLogitsLoss()
            loader=DataLoader(TensorDataset(torch.from_numpy(x[masks['train']]),torch.from_numpy(y[masks['train']])),batch_size=256,shuffle=True)
            vx=torch.from_numpy(x[masks['val']]); vy=torch.from_numpy(y[masks['val']])
            best=float('inf'); wait=0; history=[]; begin=time.perf_counter()
            for epoch in range(20):
                net.train()
                for bx,by in loader:
                    opt.zero_grad(); loss=lossfn(net(bx),by); loss.backward(); opt.step()
                net.eval()
                with torch.no_grad(): val=float(lossfn(net(vx),vy))
                history.append({'epoch':epoch+1,'validation_bce':val})
                if val<best:
                    best=val; weights={k:v.clone() for k,v in net.state_dict().items()}; best_epoch=epoch+1; wait=0
                else: wait+=1
                if wait>=4: break
            cost=time.perf_counter()-begin; net.load_state_dict(weights); net.eval()
            with torch.no_grad(): pv=torch.sigmoid(net(vx)).numpy()
            thr,vf=threshold(y[masks['val']],pv)
            write_json(out/'selection'/f'{tag}.json',{'seed':42,'steps':steps,'features':cols,'epochs_cap':20,'best_epoch':best_epoch,
                'learning_rate':.002,'batch_size':256,'history':history,'threshold':thr,'validation_macro_f1':vf,
                'architecture':str(net),'parameter_count':sum(p.numel() for p in net.parameters())})
            metrics=[]; preds=[]
            for label in masks:
                mask=masks[label]; begin=time.perf_counter()
                with torch.no_grad():
                    p=np.concatenate([torch.sigmoid(net(torch.from_numpy(b))).numpy() for b in np.array_split(x[mask],max(1,int(mask.sum()/512)))])
                latency=time.perf_counter()-begin
                saved=last.loc[mask,KEY+[LABEL_COL,'split']].copy(); saved['p_ct']=p; saved['threshold']=thr; saved['prediction']=(p>=thr).astype(int); preds.append(saved)
                for mode,th in [('default',.5),('validation_selected',thr)]:
                    metrics.append(dict(model=name,feature_set='baseline',t_s=t,seed=42,split=label,threshold_mode=mode,threshold=th,
                        coverage=int(mask.sum())/int((rounds.split==label).sum()),search_seconds=cost,predict_seconds=latency,**score(y[mask],p,th)))
            pd.concat(preds).to_csv(out/'predictions'/f'{tag}.csv.gz',index=False,compression={'method':'gzip','mtime':0})
            write_json(done,{'metrics':metrics})

import argparse, csv, random, re
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data" / "metadata"
PAIR_FILE = META / "verified_matching_pairs.csv"
MODEL_NAME = "facebook/dinov2-small"
SEED = 42

def norm(value):
    value=(value or "").replace("\\","/").strip().split("/")[-1]
    return re.sub(r"[^a-z0-9]","",value.casefold())

def read_csv(path):
    if not path.exists(): raise FileNotFoundError(f"Missing file: {path}")
    with path.open(newline="",encoding="utf-8-sig") as f: return list(csv.DictReader(f))

def resolve(name, rows, kind):
    key=norm(name)
    found=[r for r in rows if key and (key==norm(r.get("filename","")) or key==norm(r.get("local_path","")))]
    unique={}
    for r in found:
        unique[(r.get("sha256",""),r.get("local_path",""),r.get("gridfs_file_id",""),r.get("filename",""))]=r
    found=list(unique.values())
    if len(found)!=1:
        raise ValueError(f"{kind} {name!r} matched {len(found)} manifest rows; use an exact unique filename.")
    path=(found[0].get("local_path") or "").strip()
    if not path: raise ValueError(f"{kind} {name!r} has no local_path. This script does not connect to MongoDB/GridFS.")
    p=Path(path)
    if not p.is_absolute(): p=ROOT/p
    if not p.is_file(): raise FileNotFoundError(f"Image missing: {p}")
    return p.resolve()

def open_rgb(path):
    with Image.open(path) as im: return im.convert("RGB")

def augment(im):
    if random.random()<0.7:
        im=im.rotate(random.uniform(-12,12),resample=Image.Resampling.BICUBIC,fillcolor=(255,255,255))
    if random.random()<0.7:
        im=ImageEnhance.Brightness(im).enhance(random.uniform(0.88,1.12))
        im=ImageEnhance.Contrast(im).enhance(random.uniform(0.90,1.10))
    return im

class Head(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(n,n),nn.GELU(),nn.Linear(n,256))
    def forward(self,x): return F.normalize(self.net(x),p=2,dim=1)

def features(images, processor, backbone, device):
    batch=processor(images=images,return_tensors="pt")
    with torch.no_grad():
        out=backbone(pixel_values=batch["pixel_values"].to(device))
        return out.last_hidden_state[:,0,:]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--epochs",type=int,default=8)
    ap.add_argument("--batch-size",type=int,default=4)
    ap.add_argument("--val-fraction",type=float,default=0.2)
    args=ap.parse_args()
    if args.epochs<1 or args.batch_size<2: raise ValueError("Use epochs >=1 and batch-size >=2.")
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.set_num_threads(max(1,min(4,torch.get_num_threads())))
    gold_manifest=read_csv(META/"gold_query_manifest.csv")
    proto_manifest=read_csv(META/"prototype_unique_manifest.csv")
    raw=read_csv(PAIR_FILE)
    if len(raw)<5: raise ValueError("Need at least 5 resolved positive pairs; 20+ is preferable.")
    pairs=[]; seen=set()
    for r in raw:
        if r.get("label","").strip()!="1": raise ValueError("Only verified positive label=1 rows are supported.")
        gn=(r.get("gold_filename") or "").strip(); pn=(r.get("prototype_filename") or "").strip()
        key=(norm(gn),norm(pn))
        if key in seen: raise ValueError(f"Duplicate pair: {gn} / {pn}")
        seen.add(key)
        pairs.append({"g":resolve(gn,gold_manifest,"Gold"),"p":resolve(pn,proto_manifest,"Prototype")})
    indices=list(range(len(pairs))); random.Random(SEED).shuffle(indices)
    nval=max(1,round(len(pairs)*args.val_fraction))
    val_set=set(indices[:nval])
    train_pairs=[p for i,p in enumerate(pairs) if i not in val_set]
    val_pairs=[p for i,p in enumerate(pairs) if i in val_set]
    if len(train_pairs)<2: raise ValueError("Not enough training pairs.")
    device=torch.device("cpu")
    print(f"Device: {device}; pairs: {len(pairs)}; train: {len(train_pairs)}; held-out: {len(val_pairs)}")
    print(f"Loading pretrained {MODEL_NAME}; first run downloads weights.")
    processor=AutoImageProcessor.from_pretrained(MODEL_NAME)
    backbone=AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
    for param in backbone.parameters(): param.requires_grad=False
    head=Head(int(backbone.config.hidden_size)).to(device)
    opt=torch.optim.AdamW(head.parameters(),lr=1e-4,weight_decay=1e-3)
    for epoch in range(1,args.epochs+1):
        head.train(); shuffled=train_pairs.copy(); random.shuffle(shuffled); losses=[]
        for start in range(0,len(shuffled),args.batch_size):
            batch=shuffled[start:start+args.batch_size]
            if len(batch)<2: continue
            gi=[augment(open_rgb(x["g"])) for x in batch]
            pi=[augment(open_rgb(x["p"])) for x in batch]
            gf=head(features(gi,processor,backbone,device))
            pf=head(features(pi,processor,backbone,device))
            logits=(gf @ pf.T)/0.07
            target=torch.arange(len(batch),device=device)
            loss=0.5*(F.cross_entropy(logits,target)+F.cross_entropy(logits.T,target))
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss.detach()))
        print(f"Epoch {epoch}/{args.epochs} contrastive loss: {np.mean(losses) if losses else float('nan'):.4f}")
    head.eval()
    sims=[]
    with torch.inference_mode():
        for pair in val_pairs:
            gf=head(features([open_rgb(pair["g"])],processor,backbone,device))
            pf=head(features([open_rgb(pair["p"])],processor,backbone,device))
            sims.append(float((gf*pf).sum().item()))
    print(f"Held-out positive-pair cosine similarity mean: {np.mean(sims):.4f}")
    print("This is a diagnostic, not top-5 accuracy. Evaluate against the full catalogue before deployment.")
    outdir=ROOT/"models"/"checkpoints"; outdir.mkdir(parents=True,exist_ok=True)
    out=outdir/"model3_dinov2_projection.pt"
    torch.save({"model_name":MODEL_NAME,"hidden_size":int(backbone.config.hidden_size),
                "projection_state_dict":head.state_dict(),"projection_output_dim":256,
                "seed":SEED,"epochs":args.epochs,"train_pair_count":len(train_pairs),
                "validation_pair_count":len(val_pairs),
                "note":"Experimental projection head; not production validated."},out)
    print(f"Saved experimental checkpoint: {out}")
    print("No original images or MongoDB records were modified.")

if __name__=="__main__": main()

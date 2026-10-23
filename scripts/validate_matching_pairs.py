import csv, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data" / "metadata"
PAIR_FILE = META / "verified_matching_pairs.csv"
GOLD_FILE = META / "gold_query_manifest.csv"
PROTO_FILE = META / "prototype_unique_manifest.csv"
REPORT = META / "matching_pair_validation_report.csv"

def norm(value):
    value = (value or "").replace("\\", "/").strip().split("/")[-1]
    return re.sub(r"[^a-z0-9]", "", value.casefold())

def read(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file missing: {path}")
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def candidates(target, rows):
    key = norm(target)
    found = [r for r in rows if key and (
        key == norm(r.get("filename", "")) or key == norm(r.get("local_path", "")))]
    unique = {}
    for r in found:
        unique[(r.get("sha256",""), r.get("local_path",""),
                 r.get("gridfs_file_id",""), r.get("filename",""))] = r
    return list(unique.values())

def main():
    pairs = read(PAIR_FILE)
    gold, proto = read(GOLD_FILE), read(PROTO_FILE)
    if pairs and not {"gold_filename","prototype_filename","label"}.issubset(pairs[0]):
        raise ValueError("Pair CSV needs gold_filename, prototype_filename, and label columns.")
    report = []
    seen = set()
    for n, p in enumerate(pairs, start=2):
        gn, pn = p.get("gold_filename","").strip(), p.get("prototype_filename","").strip()
        problems = []
        if p.get("label","").strip() != "1":
            problems.append("label_must_be_verified_positive_1")
        key = (norm(gn), norm(pn))
        if key in seen: problems.append("duplicate_pair")
        seen.add(key)
        gc, pc = candidates(gn, gold), candidates(pn, proto)
        if len(gc) == 0: problems.append("gold_not_found")
        elif len(gc) > 1: problems.append("gold_ambiguous")
        if len(pc) == 0: problems.append("prototype_not_found")
        elif len(pc) > 1: problems.append("prototype_ambiguous")
        gp = gc[0].get("local_path","") if len(gc)==1 else ""
        pp = pc[0].get("local_path","") if len(pc)==1 else ""
        if len(gc)==1 and not gp: problems.append("gold_not_available_locally")
        if len(pc)==1 and not pp: problems.append("prototype_not_available_locally")
        report.append({"csv_row":n,"gold_filename":gn,"prototype_filename":pn,
                       "label":p.get("label",""),"gold_matches":len(gc),
                       "prototype_matches":len(pc),"gold_local_path":gp,
                       "prototype_local_path":pp,"status":"OK" if not problems else ";".join(problems)})
    fields = ["csv_row","gold_filename","prototype_filename","label","gold_matches",
              "prototype_matches","gold_local_path","prototype_local_path","status"]
    with REPORT.open("w", newline="", encoding="utf-8-sig") as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(report)
    print(f"Pairs checked: {len(report)}")
    print(f"Resolved and locally available: {sum(r['status']=='OK' for r in report)}")
    print(f"Needs attention: {sum(r['status']!='OK' for r in report)}")
    print(f"Report: {REPORT}")
    print("Read-only: original images and MongoDB were not changed.")

if __name__ == "__main__": main()

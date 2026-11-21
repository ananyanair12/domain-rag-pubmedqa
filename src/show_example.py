import json
from src.vector_store import query_collection

rows = [json.loads(l) for l in open("data/eval_questions.jsonl", encoding="utf-8")][:60]

def show(tag, r, res):
    print(f"\n=== {tag} ===")
    print("QUESTION  :", r["question"])
    print("GOLD PMID :", r["pmid"])
    for i, x in enumerate(res, 1):
        pm = x["metadata"]["pmid"]
        mark = "   <-- CORRECT" if pm == r["pmid"] else ""
        print(f"{i}. PMID {pm}  distance={x['distance']:.3f}{mark}")
        print("   ", x["text"][:130].replace("\n", " ") + "...")

hit = miss = None
for r in rows:
    res = query_collection(r["question"], "baseline", "whole", 5)
    top1 = res[0]["metadata"]["pmid"] == r["pmid"]
    if top1 and hit is None:
        hit = (r, res)
    if not top1 and miss is None:
        miss = (r, res)
    if hit and miss:
        break
if hit:
    show("SUCCESS: correct abstract ranked first", *hit)
if miss:
    show("FAILURE CASE: correct abstract not first", *miss)
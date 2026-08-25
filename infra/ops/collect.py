#!/usr/bin/env python3
"""Pull on-pod results every 2 min; print '<name> FINISHED ...' once per pod and ALL_FINISHED at the end."""
import subprocess, time, os, sys
A='/Users/tk/Documents/Personal/ais/projects26/leakage-probing/artifacts'
PODS=[('ra','103.207.149.71','10086','step4d_results_ra.jsonl'),('rd2','103.207.149.106','10628','step4d_results_rd2.jsonl'),
('rd3','103.207.149.173','15538','step4d_results_rd3.jsonl'),('rd4','103.207.149.172','15383','step4d_results_rd4.jsonl'),
('rd5','103.207.149.173','10843','step4d_results_rd5.jsonl'),('g1','64.247.201.33','17745','step4g_results_g1.jsonl'),
('g2','103.207.149.126','10611','step4g_results_g2.jsonl'),('g3','103.207.149.80','11519','step4g_results_g3.jsonl'),
('g4','103.207.149.115','14955','step4g_results_g4.jsonl')]
O=['-o','StrictHostKeyChecking=no','-o','UserKnownHostsFile=/dev/null','-o','ConnectTimeout=15','-o','BatchMode=yes']
fin=set(a for a in sys.argv[1:])  # pods already known finished
while True:
    for name,host,port,dest in PODS:
        if name in fin: continue
        subprocess.run(['scp','-q',*O,'-P',port,f'root@{host}:/workspace/run/results.jsonl',f'{A}/{dest}.tmp'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        if os.path.exists(f'{A}/{dest}.tmp'): os.replace(f'{A}/{dest}.tmp',f'{A}/{dest}')
        r=subprocess.run(['ssh',*O,'-p',port,f'root@{host}','cat /workspace/run/DONE 2>/dev/null; tail -1 /workspace/run/run.log 2>/dev/null | cut -c1-70'],capture_output=True,text=True)
        st=' '.join(r.stdout.split())
        if 'EXIT=' in st or 'TIMEOUT' in st:
            fin.add(name); print(f'{name} FINISHED {st}', flush=True)
    if all(n in fin for n,*_ in PODS): print('ALL_FINISHED', flush=True); break
    time.sleep(120)

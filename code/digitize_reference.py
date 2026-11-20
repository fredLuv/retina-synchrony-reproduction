"""Extract plotted central values from the authors' original Figure 5B vector artwork."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile
from urllib.request import Request, urlopen

import numpy as np
from pypdf import PdfReader
from pypdf.generic import ContentStream

ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'https://arxiv.org/src/1306.3061v1'
FIGURE = 'https://doi.org/10.1371/journal.pcbi.1003408.g005'
FIGURE_SHA = '1864f626da3856cbd6e05686e31c4fb4a2903de4efe81822e9bcaab23d8189e8'


def extract_strokes(art):
    pdf = PdfReader(io.BytesIO(art))
    matrix = np.eye(3)
    stack, path, strokes = [], [], []
    def point(x,y):
        a,b,_ = matrix @ np.array([float(x),float(y),1.0]); return (round(a,5),round(b,5))
    for vals,op in ContentStream(pdf.pages[0]['/Contents'],pdf).operations:
        if op==b'q': stack.append(matrix.copy())
        elif op==b'Q': matrix=stack.pop()
        elif op==b'cm':
            a,b,c,d,e,f=map(float,vals); matrix=matrix@np.array([[a,c,e],[b,d,f],[0,0,1.]])
        elif op in (b'm',b'l'): path.append(point(*vals))
        elif op==b'c': path.append(point(*vals[-2:]))
        elif op==b're':
            x,y,w,h=map(float,vals); path.extend((point(x,y),point(x+w,y),point(x+w,y+h),point(x,y+h),point(x,y)))
        elif op in (b'S',b's',b'f',b'f*',b'b',b'B',b'n'):
            if path: strokes.append(np.array(path))
            path=[]
    return strokes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pdf',type=Path,help='F5.pdf extracted from the authors’ arXiv source. Otherwise the source downloads automatically.')
    opts=parser.parse_args()
    if opts.pdf:
        art=opts.pdf.read_bytes()
    else:
        with urlopen(Request(SOURCE,headers={'User-Agent':'retina-reproduction/1.0'}),timeout=60) as res:
            with tarfile.open(fileobj=io.BytesIO(res.read()),mode='r:*') as archive:
                art=archive.extractfile('F5.pdf').read()
    if hashlib.sha256(art).hexdigest()!=FIGURE_SHA: raise ValueError('The source artwork differs from the verified figure.')
    paths=extract_strokes(art)
    # These stroke indices are specific to the verified vector file. The two axis strokes
    # span a log probability from 1 at the top to 1e-6 at the bottom of panel B.
    top,bottom=paths[182][0,1],paths[181][0,1]
    independent,pair,retina=(paths[i] for i in (281,282,388))
    probability=lambda y:10**(6*(y-top)/(top-bottom))
    gray=1000*probability(independent[:,1]); model=1000*probability(pair[:,1]); observed=1000*probability(retina[:,1])
    final_x=9+(independent[-1,0]-pair[9,0])/(pair[10,0]-pair[9,0])
    frac=(np.log(.001)-np.log(gray[-2]))/(np.log(gray[-1])-np.log(gray[-2])); edge_x=9+frac*(final_x-9)
    assert 9<edge_x<10 and len(observed)==15 and round(observed[10],1)==1.3 and round(model[10],1)==2.2
    folder=ROOT/'results';folder.mkdir(exist_ok=True)
    with (folder/'figure5b-digitized.csv').open('w',newline='') as out:
        writer=csv.writer(out);writer.writerow(['cells_active','independent_per_1000','pairwise_per_1000','observed_per_1000'])
        for k in range(15):writer.writerow([k,format(gray[k],'.9g') if k<10 else '',format(model[k],'.9g'),format(observed[k],'.9g')])
    meta={'units':'windows per 1,000','independent_curve_floor_per_1000':.001,'independent_floor_crossing_x':edge_x,
          'k10_independent_upper_bound_per_1000':.001,'k10_pairwise_per_1000':model[10],'k10_observed_per_1000':observed[10],
          'paper_text_observed_over_independent_approx':10000,'figure':FIGURE,'vector_source':'https://arxiv.org/abs/1306.3061',
          'figure_pdf_sha256':FIGURE_SHA,'note':'Central vector curves only. The independent value at 10 is below the plot. Blank cells are unmeasured, not zero. Original error bars are not digitized.'}
    (folder/'figure5b-readout.json').write_text(json.dumps(meta,indent=2)+'\n'); print('Figure 5B:',model[10],observed[10], 'per 1,000 at K=10.')

if __name__=='__main__':main()

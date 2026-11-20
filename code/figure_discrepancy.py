"""Check the clipped ten-cell independence curve in both published vector renderings."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
from urllib.request import Request, urlopen

import numpy as np
from pypdf import PdfReader

from digitize_reference import FIGURE_SHA, SOURCE, extract_strokes


ROOT = Path(__file__).resolve().parents[1]
JOURNAL = 'https://journals.plos.org/ploscompbiol/article/file?id=10.1371/journal.pcbi.1003408&type=printable'
JOURNAL_SHA = 'db87c4d2f66b74db8981ca43bb22ff8290dfb3e30c22bde2e8ea6a1259d4df86'


def fetch(url):
    with urlopen(Request(url, headers={'User-Agent': 'retina-figure-audit/1.0'}), timeout=60) as source:
        return source.read()


def checked(raw, sha):
    if hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError('Figure input differs from the identified publication. Check the new file before extracting.')
    return raw


def measure(top, bottom, gray_nine, gray_end, x_ten, red_ten, black_ten):
    # All probabilities use windows per 1,000. The panel itself spans six decades.
    def rate(y):
        return float(1_000 * 10**(6 * (y-top) / (top-bottom)))

    x9, y9 = gray_nine
    x_end, y_end = gray_end
    assert x9 < x_end < x_ten and y9 > bottom >= y_end
    slope = (y_end-y9) / (x_end-x9)
    x_floor = x9 + (bottom-y9) / slope
    fraction = (x_floor-x9) / (x_ten-x9)
    continuation = rate(y9 + slope * (x_ten-x9))
    observed = rate(red_ten)
    needed_for_10000 = observed / 10000
    fraction_if_10000 = np.log(.001 / rate(y9)) / np.log(needed_for_10000 / rate(y9))
    return {
        'observed_k10_per_1000': observed,
        'pairwise_k10_per_1000': rate(black_ten),
        'visible_gray_k9_per_1000': rate(y9),
        'visible_gray_crosses_floor_at_k': float(9 + fraction),
        'gray_k10_per_1000_if_segment_continues': continuation,
        'fold_if_segment_continues': observed / continuation,
        'gray_k10_per_1000_needed_for_10000': needed_for_10000,
        'crossing_k_if_10000_and_straight': float(9 + fraction_if_10000),
    }


def author_figure(raw):
    paths = extract_strokes(checked(raw, FIGURE_SHA))
    gray, black, red = (paths[i] for i in (281, 282, 388))
    assert len(gray) == 11 and len(red) == 15
    np.testing.assert_allclose(gray[:10, 0], black[:10, 0], atol=.002)
    return measure(paths[182][0, 1], paths[181][0, 1], gray[-2], gray[-1],
                   black[10, 0], red[10, 1], black[10, 1])


def journal_figure(raw):
    # PDF page 7. The printer's file consolidates several line segments, while keeping
    # the individual data markers. Read coordinates in the shared local chart frame.
    page = PdfReader(io.BytesIO(checked(raw, JOURNAL_SHA))).pages[6]
    ops = page.get_contents().operations

    def operands(i, expected):
        values, op = ops[i]
        assert op == expected
        return np.array(values, dtype=float)

    def origin(i):
        values = operands(i, b'cm')
        np.testing.assert_array_equal(values[:4], [1, 0, 0, 1])
        return values[-2:]

    bottom = origin(7043)[1]
    top = bottom + origin(7065)[1]
    gray_origin = origin(7081)
    np.testing.assert_allclose(operands(7078, b'RG'), [.5, .5, .5])
    assert ops[7093][1] == b'S'   # The gray path ends before the integer ten.
    gray_nine, gray_end = [gray_origin + operands(i, b'l') for i in (7091, 7092)]
    red_marker, black_marker = origin(7888), origin(7201)
    # The marker paths begin at their rightmost edge. The second cubic segment
    # ends at the leftmost edge. Half the signed diameter locates the center.
    diameter = operands(7891, b'c')[-2]
    assert diameter < 0
    x_ten = red_marker[0] + diameter / 2
    np.testing.assert_allclose(x_ten, black_marker[0] + operands(7204, b'c')[-2]/2, atol=.002)
    return measure(top, bottom, gray_nine, gray_end, x_ten, red_marker[1], black_marker[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--art-pdf', type=Path, help='Local authors’ F5.pdf, otherwise downloaded from arXiv')
    parser.add_argument('--journal-pdf', type=Path, help='Local published PLOS paper, otherwise downloaded from PLOS')
    args = parser.parse_args()
    if args.art_pdf:
        art = args.art_pdf.read_bytes()
    else:
        with tarfile.open(fileobj=io.BytesIO(fetch(SOURCE)), mode='r:*') as bundle:
            art = bundle.extractfile('F5.pdf').read()
    journal = args.journal_pdf.read_bytes() if args.journal_pdf else fetch(JOURNAL)
    author = author_figure(art)
    published = journal_figure(journal)
    np.testing.assert_allclose(published['fold_if_segment_continues'], author['fold_if_segment_continues'], rtol=.01)
    result = {
        'quantity': 'Exactly 10 active cells among 40 in one 20-ms window.',
        'published_direct_gray_value_at_10': None,
        'did_authors_measure_10000_for_the_plotted_group': 'Unknown.',
        'interpretation': 'The continuation assumes a straight connection on the log-probability axis from K=9 to K=10. It is a graphical estimate from the clipped path, not an author-reported underlying number.',
        'panel_floor_per_1000': .001,
        'author_vector': dict(url=SOURCE, member='F5.pdf', sha256=FIGURE_SHA, **author),
        'journal_vector': dict(url=JOURNAL, page=7, sha256=JOURNAL_SHA, **published),
    }
    (ROOT/'results'/'figure-discrepancy.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Implied fold under log-linear continuation:', author['fold_if_segment_continues'], published['fold_if_segment_continues'])


if __name__ == '__main__':
    main()

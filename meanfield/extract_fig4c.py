import sys, numpy as np, pymupdf
d = pymupdf.open(sys.argv[1]); p = d[3]
clip = pymupdf.Rect(300, 205, 435, 292)
panels = {'on': dict(col=(0.893, 0.435, 0.191), x0=318.8, x_per_ms=(361.78 - 318.8) / 600, ytop=219.84, yval=8e-3),
          'off': dict(col=(0.217, 0.357, 0.642), x0=380.12, x_per_ms=(421.6 - 380.12) / 600, ytop=219.84, yval=1.2e-2)}
y0 = 274.32
out = {}
for name, pp in panels.items():
    bars = []
    for dr in p.get_drawings():
        f = dr.get('fill')
        if f is None or not dr['rect'].intersects(clip) or np.abs(np.array(f) - pp['col']).max() > 0.01:
            continue
        if len(dr['items']) < 20:
            continue                      # stray single rectangles (not part of the 28-bar histogram)
        for it in dr['items']:
            if it[0] == 're':
                r = it[1]
                bars.append(((r.x0 - pp['x0']) / pp['x_per_ms'], (r.x1 - pp['x0']) / pp['x_per_ms'],
                             (y0 - r.y0) / (y0 - pp['ytop']) * pp['yval']))
    bars = sorted(bars)
    assert len(bars) == 28, len(bars)
    bars = [(40 + 20 * i, 60 + 20 * i, h) for i, (_, _, h) in enumerate(bars)]   # bins 40-600 ms
    out[name] = bars
    print(name, len(bars), 'sum p =', round(sum(h for *_, h in bars), 4))
    for a, b, h in bars: print(f'  {a:6.1f}-{b:6.1f}  {h:.5f}')
edges = np.arange(0, 620, 20)
with open(sys.argv[2], 'w') as fh:
    fh.write('# Panichello et al. 2024 Nature 636:422, Fig. 4c, extracted from the vector bars of the PDF.\n')
    fh.write('# Columns: bin_lo_ms, bin_hi_ms, p_on (orange, n=3890 On states), p_off (blue, n=6174 Off states); '
             'probability per 20 ms bin as plotted (bins with no bar = 0).\n')
    fh.write('bin_lo,bin_hi,p_on,p_off\n')
    for lo in edges[:-1]:
        g = lambda bars: next((h for a, b, h in bars if abs(a - lo) < 3), 0.0)
        fh.write(f'{lo},{lo + 20},{g(out["on"]):.6f},{g(out["off"]):.6f}\n')

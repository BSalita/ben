"""Build the board-by-board page from the table of brill_board_table.py.

    elo/.venv/Scripts/python.exe ben/scripts/match/brill/brill_board_page.py OUT_DIR MODEL_NAME [--dd DD.json]
        [--views VIEWS.json --view-id ID]

Reads OUT_DIR/boards.parquet, events.csv, standings.csv, lifetime.csv, meta.json and dd.json; writes OUT_DIR/brill-board-table.html
(from brill_board_page.template.html), OUT_DIR/leader-board.html (the same page as a
whole document, for a web server) and OUT_DIR/groups.csv, and prints the group table.
VIEWS.json (brill_views.py) lists the pages of the other views, linked at the top
of the page; ID is this page's ('' for the main one).
"""
import argparse
import json
import pathlib
import re
import sys
from datetime import datetime

import polars as pl

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from brill_board_table import LIFETIME_MIN_EVENTS  # noqa: E402
from score_brill_replay import imps, score_ns  # noqa: E402

TOP = 10
MORTY_GIF = HERE.parents[3] / 'assets' / 'logo_assistant.gif'
STANDALONE_HEAD = ('<!doctype html>\n<html lang="en">\n'
                   '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
                   '<style>body { margin: 0 } img { max-width: 100% }</style>\n')

GROUPS = [
    ('Same contract as most tables', 'No bidding difference: what is left is the card play and the lead.'),
    ('We bid game, most tables a partscore', 'Pushing to game.'),
    ('We stopped in a partscore, most tables bid game', 'Missed games.'),
    ('We bid slam, most tables stopped lower', 'Slam bidding.'),
    ('We stopped below slam, most tables bid it', 'Missed slams.'),
    ('Different strain, same range', 'Choice of strain: notrump or a suit, major or minor.'),
    ('Same strain, different level', 'One level higher or lower in the same partscore, game or slam range.'),
    ('We declared, most tables defended', 'Competing: we bought the contract.'),
    ('We defended, most tables declared', 'Competing: we sold out, or pushed them up.'),
    ('Opponents in a different contract', 'They declare at both; our calls changed where they play.'),
    ('We were doubled', 'Our side plays doubled where most tables were not.'),
    ('We doubled them', 'The opponents play doubled where most tables did not double.'),
    ('Passed out or most tables passed out', 'One of the two is a pass-out.'),
]


def icon_link(size=64):
    """<link rel="icon"> holding the Morty GIF (src/assets/logo_assistant.gif, 512 px, 1.6 MB)
    scaled to `size` pixels with its animation kept, as a data URI. '' when the GIF is missing."""
    import base64
    import io
    from PIL import Image, ImageSequence
    if not MORTY_GIF.exists():
        print(f'warning: {MORTY_GIF} not found; the page gets no icon')
        return ''
    frames, durations = [], []
    for frame in ImageSequence.Iterator(Image.open(MORTY_GIF)):
        durations.append(frame.info.get('duration', 100))
        frames.append(frame.convert('RGBA').resize((size, size), Image.LANCZOS))
    data = io.BytesIO()
    frames[0].save(data, format='GIF', save_all=True, append_images=frames[1:], duration=durations, loop=0,
                   disposal=2, optimize=True)
    return f'<link rel="icon" type="image/gif" href="data:image/gif;base64,{base64.b64encode(data.getvalue()).decode()}">\n'


def parse(text):
    """(level, strain, doubling, declarer) of '4S N' / '4SX N' / '3NT S', or None for Pass."""
    m = re.match(r'^(\d)(NT|[SHDCN])(X{0,2})\s+([NESW])$', (text or '').strip())
    return (int(m[1]), m[2][0], m[3], m[4]) if m else None


def level_range(contract):
    level, strain = contract[0], contract[1]
    if level >= 6:
        return 2
    return 1 if (strain == 'N' and level >= 3) or (strain in 'HS' and level >= 4) or (strain in 'CD' and level >= 5) else 0


def group_of(ours, common):
    if ours is None or common is None:
        return 'Same contract as most tables' if ours == common else 'Passed out or most tables passed out'
    our_side, their_side = ours[3] in 'NS', common[3] in 'NS'
    if ours[:3] == common[:3] and our_side == their_side:
        return 'Same contract as most tables'
    if ours[2] and not common[2]:
        return 'We were doubled' if our_side else 'We doubled them'
    if our_side != their_side:
        return 'We declared, most tables defended' if our_side else 'We defended, most tables declared'
    if not our_side:
        return 'Opponents in a different contract'
    a, b = level_range(ours), level_range(common)
    if a != b:
        if 2 in (a, b):
            return 'We bid slam, most tables stopped lower' if a == 2 else 'We stopped below slam, most tables bid it'
        return 'We bid game, most tables a partscore' if a == 1 else 'We stopped in a partscore, most tables bid game'
    return 'Different strain, same range' if ours[1] != common[1] else 'Same strain, different level'


def par_text(text):
    """'2NTE= / 2NTW=' -> '2NT EW'."""
    by_contract = {}
    for part in text.split(' / '):
        m = re.match(r'^(\d)(NT|[♠♥♦♣SHDC])(X{0,2})([NESW])', part)
        if not m:
            return text
        strain = {'♠': 'S', '♥': 'H', '♦': 'D', '♣': 'C'}.get(m[2], m[2])
        by_contract.setdefault(f'{m[1]}{strain}{m[3]}', []).append(m[4])
    return ', '.join(f'{c} {"".join(dict.fromkeys(d))}' for c, d in by_contract.items())


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('out_dir', type=pathlib.Path)
    parser.add_argument('model', help='name of the replayed model')
    parser.add_argument('--dd', type=pathlib.Path, default=None, help='double-dummy cache (default OUT_DIR/dd.json)')
    parser.add_argument('--views', type=pathlib.Path, default=None)
    parser.add_argument('--view-id', default='')
    a = parser.parse_args()
    views = json.loads(a.views.read_text(encoding='utf-8')) if a.views else []
    build_page(a.out_dir, a.model, a.dd, views, a.view_id)


def build_page(out_dir, model, dd_path=None, views=(), view_id=''):
    boards = pl.read_parquet(out_dir / 'boards.parquet')
    events = pl.read_csv(out_dir / 'events.csv')
    dd = json.loads((dd_path or out_dir / 'dd.json').read_text(encoding='utf-8'))
    meta_path = out_dir / 'meta.json'
    meta = json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.exists() else {
        'view': 'as-played', 'lifetime_min': LIFETIME_MIN_EVENTS, 'events_left_out': 0, 'models': []}
    if 'model_key' not in boards.columns:
        boards = boards.with_columns(pl.lit('').alias('model_key'), pl.lit(model).alias('model'), pl.lit(None, pl.String).alias('played_at'))

    extra = []
    for b in boards.iter_rows(named=True):
        ours = parse(f"{b['contract']} {b['declarer']}") if b['declarer'] else None
        common = parse(b['field_common'])
        tricks = dd[f"{b['deal']}|{b['dealer']}|{b['vul']}"]['tricks']
        common_dd = 0 if common is None else score_ns(*common, tricks[common[1] + common[3]], b['vul'])
        extra.append({'group': group_of(ours, common), 'imp_dd_vs_field': int(imps(b['dd_score'] - common_dd))})
    boards = pl.concat([boards, pl.DataFrame(extra)], how='horizontal').with_row_index('i')
    boards.drop('deal').write_csv(out_dir / 'boards.csv')

    groups = []
    for name, text in GROUPS:
        g = boards.filter(pl.col('group') == name)
        if not g.height:
            continue
        mp, im = g.filter(pl.col('scoring') == 'MP'), g.filter(pl.col('scoring') == 'IMP')
        groups.append({'name': name, 'boards': g.height,
                       'pct': round(mp['pct'].mean(), 1) if mp.height else None,
                       'imp': round(im['imp'].mean(), 2) if im.height else None,
                       'vsfield': round(g['imp_dd_vs_field'].mean(), 2), 'vsfield_total': int(g['imp_dd_vs_field'].sum()), 'bid': round(g['imp_dd_vs_par'].mean(), 2),
                       'ev': round((g['dd_ev'] - g['par_ev']).mean()), 'text': text})
    pl.DataFrame(groups).write_csv(out_dir / 'groups.csv')
    with pl.Config(tbl_rows=30, tbl_width_chars=200, fmt_str_lengths=50):
        print(pl.DataFrame(groups).drop('text'))

    differs = boards.filter(pl.col('group') != 'Same contract as most tables')
    best = differs.sort('imp', descending=True).head(15)['i'].to_list()
    worst = differs.sort('imp').head(15)['i'].to_list()
    mp, im = boards.filter(pl.col('scoring') == 'MP'), boards.filter(pl.col('scoring') == 'IMP')
    mp_events, imp_events = events.filter(pl.col('scoring') == 'MP'), events.filter(pl.col('scoring') == 'IMP')
    same = (boards['group'] == 'Same contract as most tables').mean() * 100
    figures = [
        {'value': f"{mp['pct'].mean():.1f}%", 'label': f'average board in {mp_events.height} matchpoint events'},
        {'value': f"{im['imp'].mean():+.2f} IMP", 'label': f'per board in {imp_events.height} IMP events'},
        {'value': f"{events['rank'].median():.0f} of {events['entrants'].median():.0f}", 'label': 'median rank in an event, of the median field'},
        {'value': f"{(events['rank'] == 1).sum()} win{'' if (events['rank'] == 1).sum() == 1 else 's'}", 'label': f"and {(events['rank'] <= 3).sum()} top-three finishes in {events.height} events"},
        {'value': f'{same:.0f}%', 'label': 'of boards in the contract most tables reached'},
        {'value': f"{boards['imp_dd_vs_field'].mean():+.2f} IMP", 'label': 'per board from the bidding alone: double-dummy result of our contract against that of most tables'},
    ]
    # the models behind the rows: an index into `models` on every board and event
    models = [{'key': m['model_key'], 'name': m['model'], 'events': m['events'], 'boards': m['boards'],
               'first': m['first'], 'last': m['last']} for m in meta['models']] or [
        {'key': k, 'name': n, 'events': 0, 'boards': 0, 'first': '', 'last': ''}
        for k, n in boards.select('model_key', 'model').unique(maintain_order=True).iter_rows()]
    model_index = {m['key']: k for k, m in enumerate(models)}
    event_model = {t: model_index[g['model_key'].mode()[0]] for (t,), g in boards.group_by('tournament_id')}
    page_events = [{'id': e['tournament_id'], 'date': e['date'], 'event': e['event'], 'short': e['event'].split(' — ')[0],
                    'scoring': e['scoring'], 'measure': e['measure'], 'rank': e['rank'], 'entrants': e['entrants'],
                    'winner': e['winner'], 'winner_measure': e['winner_measure'], 'pending': e['pending'],
                    'bid': e['imp_dd_vs_par'], 'm': event_model[e['tournament_id']]}
                   for e in events.sort('date', 'event', descending=True).iter_rows(named=True)]
    page_boards = [{'i': b['i'], 't': b['tournament_id'], 'sc': b['scoring'], 'b': b['board'], 'dl': b['dealer'], 'v': b['vul'],
                    'deal': b['deal'], 'a': b['auction'], 'c': b['contract'], 'by': b['declarer'], 'r': b['result'],
                    's': b['score'], 'pct': b['pct'], 'imp': b['imp'], 'ddr': b['dd_result'], 'dds': b['dd_score'],
                    'ddev': b['dd_ev'], 'par': b['par_score'], 'parc': par_text(b['par']), 'parev': b['par_ev'],
                    'parevc': b['par_ev_contract'], 'src': b['result_source'], 'rob': b['robots'], 'fc': b['field_common'],
                    'fcs': b['field_common_share'], 'fm': b['field_median'], 'n': b['tables'], 'bc': b['brill_contract'],
                    'bs': b['brill_score'], 'bid': b['imp_dd_vs_par'], 'vf': b['imp_dd_vs_field'], 'g': b['group'],
                    'm': model_index[b['model_key']], 'at': (b['played_at'] or '')[:16].replace('T', ' ')}
                   for b in boards.iter_rows(named=True)]
    # rankings: the top ten of every event and of each scoring's lifetime table, and our row when it is lower
    def top(rows, extra=()):
        return [[r['rank'], r['name'], r['measure'], int(r['player_id'] == 'OURS')] + [r[k] for k in extra]
                for r in rows if r['rank'] <= TOP or r['player_id'] == 'OURS']
    standings = pl.read_csv(out_dir / 'standings.csv').sort('rank', 'name')
    ranks = {t[0]: top(g.iter_rows(named=True)) for t, g in standings.group_by('tournament_id')}
    life = pl.read_csv(out_dir / 'lifetime.csv').filter(pl.col('rank').is_not_null()).sort('rank', 'name')
    lifetime = {t[0]: {'ranked': g['ranked'][0], 'rows': top(g.iter_rows(named=True), ('events', 'wins', 'top3'))}
                for t, g in life.group_by('scoring')}
    # the events our model played itself as an entrant: their rows are its real tables, not the replay
    live = boards.filter(pl.col('robots') == 'live')
    entered = {'events': live['tournament_id'].n_unique(), 'boards': live.height, 'first': live['date'].min()} if live.height else None
    nav = [{'id': v['id'], 'title': v['title'], 'text': v.get('text', ''), 'current': v['id'] == view_id} for v in views]
    data = {'made': f'{datetime.now():%Y-%m-%d %H:%M}', 'model': model, 'first': boards['date'].min(), 'last': boards['date'].max(),
            'entered': entered, 'view': meta['view'], 'left_out': meta.get('events_left_out', 0), 'models': models, 'views': nav,
            'ranks': ranks, 'lifetime': lifetime, 'lifetime_min': meta['lifetime_min'],
            'pending': int((boards['result_source'] == 'dd-pending').sum()), 'figures': figures, 'groups': groups,
            'best': best, 'worst': worst, 'events': page_events, 'boards': page_boards}
    template = (HERE / 'brill_board_page.template.html').read_text(encoding='utf-8')
    payload = json.dumps(data, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')
    out = out_dir / 'brill-board-table.html'
    page = template.replace('/*DATA*/', payload)
    out.write_text(page, encoding='utf-8')
    # The same page as a whole document, for a web server (prod serves it at brill.7nt.info).
    # brill-board-table.html has no doctype or viewport: the Artifact host wraps it in its own.
    # Its tab icon is the Morty GIF, inside the file so the page stays one file on any path.
    standalone = STANDALONE_HEAD + page.replace('</title>\n', '</title>\n' + icon_link(), 1)
    (out_dir / 'leader-board.html').write_text(standalone, encoding='utf-8')
    print(f'{out}  {out.stat().st_size / 1e6:.2f} MB, {len(page_boards)} boards, {len(page_events)} events, pending {data["pending"]}')


if __name__ == '__main__':
    main()

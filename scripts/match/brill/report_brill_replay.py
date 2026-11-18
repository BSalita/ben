"""Tables from scored Brill replays (score_brill_replay.py), in the Brill Elo app's terms.

    elo/.venv/Scripts/python.exe ben/scripts/match/brill/report_brill_replay.py FIELD.jsonl OUT_DIR [baseline-config]

Reads OUT_DIR/boards.parquet, prints the tables and writes OUT_DIR/report.txt.
"""
import io
import json
import pathlib
import sys
from datetime import datetime

import numpy as np
import polars as pl

SRC = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(SRC / 'elo'))
import elo_brill  # noqa: E402
from elo_common import zscore_to_chess  # noqa: E402

REFERENCE = ('BBOBen', 'BEN', 'Brill')
MIN_GAMES = 10
ELO_WINDOW = 100  # Elo_Low / Elo_High: the range of the rating over the entrant's last this many events


def real_boards(field):
    rows = []
    boards_in_event = {}
    for b in field:
        boards_in_event[b['tournament_id']] = max(boards_in_event.get(b['tournament_id'], 0), b['board'])
    for b in field:
        for t in b['tables']:
            rows.append({
                'tournament_id': b['tournament_id'], 'Event': b['event'],
                'Event_Type': elo_brill.event_type(b['event']), 'Scoring': b['scoring'],
                'Event_Time': b['date'], 'Date': b['date'],
                'Boards_In_Event': boards_in_event[b['tournament_id']], 'Board': b['board'],
                'Player_ID': t['player_id'], 'Player_Name': t['name'], 'Player_Type': t['type'],
                'Site_Rank': None, 'Score_NS': t['ns_score'],
            })
    return pl.DataFrame(rows, schema_overrides={'Site_Rank': pl.Int32, 'Boards_In_Event': pl.Int32,
                                                'Board': pl.Int32, 'Score_NS': pl.Int32})


def virtual_rows(real, scored, name):
    """The config as one more entrant of every event it replayed."""
    facts = real.unique(subset=['tournament_id', 'Board'], keep='first').drop(
        'Player_ID', 'Player_Name', 'Player_Type', 'Score_NS')
    rows = scored.filter(pl.col('config') == name).select(
        'tournament_id', pl.col('board').cast(pl.Int32).alias('Board'), pl.col('score').cast(pl.Int32).alias('Score_NS'))
    return rows.join(facts, on=['tournament_id', 'Board'], how='left', validate='1:1').with_columns(
        pl.lit(f'V:{name}').alias('Player_ID'), pl.lit(f'[{name}]').alias('Player_Name'),
        pl.lit('Replay').alias('Player_Type'),
    ).select(real.columns)


def standings(boards, scale=None):
    """(table, scale): one row per entrant, Elo replayed over all events as the app does."""
    boards = elo_brill.add_field_scores(boards)
    sessions = elo_brill.rate_sessions(elo_brill.session_table(boards))
    if scale is None:
        scale = elo_brill.elo_scale(sessions)
    won = ((pl.col('Scoring') == 'MP') & (pl.col('Pct_Rank') == 1)) | ((pl.col('Scoring') == 'IMP') & (pl.col('IMP_Rank') == 1))
    rank = pl.when(pl.col('Scoring') == 'MP').then(pl.col('Pct_Rank')).otherwise(pl.col('IMP_Rank'))
    table = sessions.sort('Event_Time', 'tournament_id').group_by('Player_ID').agg(
        pl.col('Player_Name').last().alias('Entrant'),
        pl.col('Player_Type').last().alias('Type'),
        pl.len().alias('Events'),
        pl.col('Boards').sum(),
        pl.col('Elo_After').last().alias('Elo_Raw'),
        pl.col('Elo_After').tail(ELO_WINDOW).min().alias('Elo_Raw_Low'),
        pl.col('Elo_After').tail(ELO_WINDOW).max().alias('Elo_Raw_High'),
        won.sum().alias('Wins'),
        rank.mean().round(2).alias('Avg_Rank'),
        pl.col('Field_Size').mean().round(1).alias('Avg_Field'),
        pl.col('Pct').filter(pl.col('Scoring') == 'MP').mean().round(2).alias('MP_Events_Pct'),
        pl.col('IMP').filter(pl.col('Scoring') == 'IMP').mean().round(3).alias('IMP_Events_IMP'),
    )
    quality = boards.group_by('Player_ID').agg(
        pl.col('Pct_NS').mean().round(2).alias('Avg_Pct'), pl.col('IMP_NS').mean().round(3).alias('Avg_IMP'))
    table = table.join(quality, on='Player_ID').filter(pl.col('Events') >= MIN_GAMES).with_columns(
        *[pl.col(raw).map_elements(lambda r: int(round(zscore_to_chess(r, *scale))), return_dtype=pl.Int32).alias(shown)
          for raw, shown in (('Elo_Raw', 'Elo'), ('Elo_Raw_Low', 'Elo_Low'), ('Elo_Raw_High', 'Elo_High'))],
    ).with_columns(pl.col('Elo_Raw').rank(method='min', descending=True).cast(pl.Int32).alias('Elo_Rank'))
    return table.sort('Elo_Rank').select(
        'Elo_Rank', 'Entrant', 'Type', 'Elo', 'Elo_Low', 'Elo_High', 'Events', 'Boards', 'Avg_Pct', 'Avg_IMP',
        'MP_Events_Pct', 'IMP_Events_IMP', 'Wins', 'Avg_Rank', 'Player_ID'), scale


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 2:
        return float('nan'), float('nan'), n
    return float(values.mean()), float(1.96 * values.std(ddof=1) / np.sqrt(n)), n


def main():
    field_path, out_dir = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    baseline = sys.argv[3] if len(sys.argv) > 3 else None
    out = io.StringIO()

    def emit(*parts):
        text = ' '.join(str(p) for p in parts)
        print(text)
        out.write(text + '\n')

    pl.Config.set_tbl_cols(40)
    pl.Config.set_tbl_width_chars(250)
    pl.Config.set_tbl_rows(80)
    pl.Config.set_fmt_str_lengths(40)
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_hide_column_data_types(True)

    emit(f'Brill replay report {datetime.now():%Y-%m-%d %H:%M}')
    field = [json.loads(line) for line in field_path.open(encoding='utf-8')]
    scored = pl.read_parquet(out_dir / 'boards.parquet')
    configs = scored['config'].unique(maintain_order=True).to_list()
    real = real_boards(field)
    emit(f'{len(field)} boards, {real["tournament_id"].n_unique()} events, {real["Date"].min()} to {real["Date"].max()}, '
         f'{real.height} real tables; configs: {", ".join(configs)}')

    emit('\n1. How the replayed auctions were completed and scored (share of boards, %)')
    emit(scored.group_by('config', maintain_order=True).agg(
        pl.len().alias('boards'),
        ((pl.col('status') == 'record').mean() * 100).round(1).alias('all robot calls real'),
        ((pl.col('source') == 'auction').mean() * 100).round(1).alias('result: same auction'),
        ((pl.col('source') == 'contract').mean() * 100).round(1).alias('result: same contract'),
        ((pl.col('source') == 'dd').mean() * 100).round(1).alias('result: double dummy'),
    ))

    emit('\n1b. Result against all real tables, by how the board was scored')
    emit(scored.group_by('config', 'source').agg(
        pl.len().alias('boards'), pl.col('pct_all').mean().round(2), pl.col('imp_all').mean().round(3),
    ).sort('config', 'source'))

    emit('\n2. Agreement with what the real entrants did, by period (South calls on their own auctions %, whole auction %)')
    period = (
        pl.when(pl.col('date') < '2026-10-01').then(pl.lit('1 Aug-Sep'))
        .when(pl.col('date') < '2026-11-01').then(pl.lit('2 Oct'))
        .when(pl.col('date') < '2026-11-12').then(pl.lit('3 Nov 1-11'))
        .otherwise(pl.lit('4 Nov 12-16')).alias('period')
    )
    aggs = []
    for tag in REFERENCE:
        aggs += [
            (pl.col(f'{tag}_same').sum() / pl.col(f'{tag}_turns').sum() * 100).round(1).alias(f'{tag} calls'),
            (pl.col(f'{tag}_auction_same').mean() * 100).round(1).alias(f'{tag} auction'),
        ]
    emit(scored.with_columns(period).group_by('config', 'period').agg(pl.len().alias('boards'), *aggs).sort('config', 'period'))
    emit(scored.group_by('config', maintain_order=True).agg(pl.len().alias('boards'), *aggs))

    emit(f'\n3. The real field, Elo replayed over all events (entrants with {MIN_GAMES}+ events)')
    table, scale = standings(real)
    emit(table.drop('Player_ID'))

    emit('\n4. Each config added to the real field as one more entrant (one at a time)')
    lines = []
    for name in configs:
        with_virtual, _ = standings(pl.concat([real, virtual_rows(real, scored, name)]), scale)
        lines.append(with_virtual.filter(pl.col('Player_ID') == f'V:{name}').with_columns(
            pl.lit(with_virtual.height).alias('Of')))
    emit(pl.concat(lines).drop('Player_ID', 'Type'))

    emit('\n5. On the boards a real entrant played: the config in its place, against the same other tables')
    real_scored = elo_brill.add_field_scores(real)
    for tag in REFERENCE:
        theirs = real_scored.filter(pl.col('Player_ID') == f'R:{tag}').select(
            'tournament_id', pl.col('Board').alias('board'), pl.col('Pct_NS').alias('their_pct'),
            pl.col('IMP_NS').alias('their_imp'), pl.col('Score_NS').alias('their_score'))
        rows = []
        for name in configs:
            j = scored.filter(pl.col('config') == name).join(theirs, on=['tournament_id', 'board'])
            if j.height < 2:
                continue
            d_pct, ci_pct, n = mean_ci(j[f'pct_R:{tag}'] - j['their_pct'])
            d_imp, ci_imp, _ = mean_ci(j[f'imp_R:{tag}'] - j['their_imp'])
            rows.append({
                'config': name, 'boards': n,
                f'{tag} pct': round(j['their_pct'].mean(), 2), 'config pct': round(j[f'pct_R:{tag}'].mean(), 2),
                'diff pct': round(d_pct, 2), '+-': round(ci_pct, 2),
                f'{tag} imp': round(j['their_imp'].mean(), 3), 'config imp': round(j[f'imp_R:{tag}'].mean(), 3),
                'diff imp': round(d_imp, 3), '+- ': round(ci_imp, 3),
                'same score %': round((j['score'] == j['their_score']).mean() * 100, 1),
            })
        if rows:
            emit(pl.DataFrame(rows))

    if baseline:
        emit(f'\n6. Each config against {baseline}, board by board (matchpoint % and IMPs against all real tables)')
        base = scored.filter(pl.col('config') == baseline).select(
            'tournament_id', 'board', pl.col('pct_all').alias('b_pct'), pl.col('imp_all').alias('b_imp'),
            pl.col('auction').alias('b_auction'), pl.col('contract').alias('b_contract'), pl.col('score').alias('b_score'))
        rows = []
        for name in configs:
            if name == baseline:
                continue
            j = scored.filter(pl.col('config') == name).join(base, on=['tournament_id', 'board'])
            d_pct, ci_pct, n = mean_ci(j['pct_all'] - j['b_pct'])
            d_imp, ci_imp, _ = mean_ci(j['imp_all'] - j['b_imp'])
            mp = j.filter(pl.col('scoring') == 'MP')
            im = j.filter(pl.col('scoring') == 'IMP')
            d_mp, ci_mp, n_mp = mean_ci(mp['pct_all'] - mp['b_pct'])
            d_im, ci_im, n_im = mean_ci(im['imp_all'] - im['b_imp'])
            rows.append({
                'config': name, 'boards': n,
                'auction differs %': round((j['auction'] != j['b_auction']).mean() * 100, 1),
                'contract differs %': round((j['contract'] != j['b_contract']).mean() * 100, 1),
                'score differs %': round((j['score'] != j['b_score']).mean() * 100, 1),
                'diff pct': round(d_pct, 2), '+-': round(ci_pct, 2),
                'diff imp': round(d_imp, 3), '+- ': round(ci_imp, 3),
                'MP events: diff pct': round(d_mp, 2), '+-  ': round(ci_mp, 2),
                'IMP events: diff imp': round(d_im, 3), '+-   ': round(ci_im, 3),
            })
        emit(pl.DataFrame(rows))

    emit(f'\n7. Each config in the place of each real entrant ({MIN_GAMES}+ events), on the boards that entrant played, '
         'against the same other tables: config minus entrant, with the 95% interval. '
         'Matchpoint % over all its boards; IMPs likewise.')
    by_key = {(b['tournament_id'], b['board']): b for b in field}
    events_of = real.group_by('Player_ID').agg(pl.col('tournament_id').n_unique().alias('n'))
    entrants = events_of.filter(pl.col('n') >= MIN_GAMES)['Player_ID'].to_list()
    names = dict(real.select('Player_ID', 'Player_Name').unique(subset='Player_ID', keep='last').iter_rows())
    for name in configs:
        diffs = {e: ([], [], [], []) for e in entrants}
        for tid, board, mine in scored.filter(pl.col('config') == name).select('tournament_id', 'board', 'score').iter_rows():
            tables = by_key[(tid, board)]['tables']
            scores = np.array([t['ns_score'] for t in tables])
            for i, t in enumerate(tables):
                if t['player_id'] not in diffs:
                    continue
                others = np.delete(scores, i)
                theirs = scores[i]
                pct = lambda s: (np.sum(others < s) + 0.5 * np.sum(others == s)) / len(others) * 100  # noqa: E731
                imp = lambda s: (np.sign(s - others) * np.searchsorted(elo_brill._IMP_BOUNDS, np.abs(s - others), side='right')).mean()  # noqa: E731
                d = diffs[t['player_id']]
                d[0].append(pct(mine) - pct(theirs))
                d[1].append(imp(mine) - imp(theirs))
                d[2].append(pct(theirs))
                d[3].append(pct(mine))
        rows = []
        for e, (d_pct, d_imp, theirs, mine) in diffs.items():
            m_pct, ci_pct, n = mean_ci(d_pct)
            m_imp, ci_imp, _ = mean_ci(d_imp)
            verdict = 'ahead' if m_pct - ci_pct > 0 else 'behind' if m_pct + ci_pct < 0 else 'level'
            rows.append({'entrant': names[e], 'type': e[0], 'boards': n, 'entrant pct': round(float(np.mean(theirs)), 2),
                         f'[{name}] pct': round(float(np.mean(mine)), 2), 'diff pct': round(m_pct, 2), '+-': round(ci_pct, 2),
                         'diff imp': round(m_imp, 3), '+- ': round(ci_imp, 3), f'[{name}] is': verdict})
        emit(pl.DataFrame(rows).sort('diff pct'))

    (out_dir / 'report.txt').write_text(out.getvalue(), encoding='utf-8')


if __name__ == '__main__':
    main()

"""Fetch official submission scores and rank teams by their best valid score."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    snapshot = ROOT / 'data/external/leaderboard_autoresearch_latest.json'
    subprocess.run([sys.executable, str(ROOT/'scripts/fetch_leaderboard.py'),
                    '--output', str(snapshot)], check=True)
    data = json.loads(snapshot.read_text())
    valid = [r for r in data['items'] if isinstance(r.get('score'), (int, float))
             and r.get('usedPairs') == 344841]
    best = {}
    for row in valid:
        key = row['teamId']
        if key not in best or row['score'] < best[key]['score']:
            best[key] = row
    ranked = sorted(best.values(), key=lambda r: r['score'])
    ours = [r for r in valid if r['teamName'] == 'elegant-alligator']
    mine = next((r for r in ranked if r['teamName'] == 'elegant-alligator'), None)
    report = {'retrieved_at': data['retrievedAt'], 'source': data['source'],
              'rank': None if mine is None else 1+sum(r['score'] < mine['score'] for r in ranked),
              'ranked_teams': len(ranked), 'best': mine, 'leaders': ranked[:3],
              'our_submissions': sorted(ours, key=lambda r: r['processedAt'])}
    (ROOT/'docs/autoresearch_official_status.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

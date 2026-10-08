import csv

fields = [
    'id','title','authors','year','venue','type','task','datasets','models','shift',
    'calibration','uncertainty','selective_prediction','grounding_image_reliance',
    'hallucination','external_validation','key_finding','limitation','relevance',
    'verification_status','priority','url','notes'
]

rows = []
with open('references/_matrix_rows.txt', encoding='utf-8') as f:
    for lineno, line in enumerate(f, 1):
        line = line.rstrip('\n')
        if not line.strip():
            continue
        parts = line.split('|~|')
        if len(parts) != len(fields):
            raise SystemExit(f'line {lineno}: expected {len(fields)} fields, got {len(parts)}')
        rows.append(dict(zip(fields, parts)))

with open('references/literature_matrix_v1_1.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

print('wrote', len(rows), 'rows')

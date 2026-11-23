"""用法：在專案根目錄執行 python3 tests/run_acceptance.py。"""
from pathlib import Path
import json
import sys
import math

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from parsing.extract_rules import extract_features, extract_awarded_consolation


def main():
    base = Path(__file__).resolve().parent / 'acceptance'
    manifest = json.loads((base / 'expected.json').read_text(encoding='utf-8'))
    results = []
    for case in manifest['cases']:
        text = (base / case['file']).read_text(encoding='utf-8')
        expected = dict(case['expected'])
        try:
            actual = extract_features(text)
            if 'expected_awarded_consolation' in case:
                expected['awarded_consolation'] = case['expected_awarded_consolation']
                actual['awarded_consolation'] = extract_awarded_consolation(text)
            error = None
        except Exception as exc:
            actual, error = {}, f'{type(exc).__name__}: {exc}'
        fields = []
        for key, wanted in expected.items():
            got = actual.get(key)
            same = key in actual and got == wanted
            if isinstance(wanted, float) and isinstance(got, (int, float)):
                same = math.isclose(got, wanted, rel_tol=1e-9, abs_tol=1e-9)
            fields.append({'field':key,'expected':wanted,'actual':got,'passed':same and error is None})
        passed = all(f['passed'] for f in fields)
        results.append({'id':case['id'],'title':case['title'],'passed':passed,'error':error,'fields':fields})
        print(f"{'通過' if passed else '待修正'} {case['id']} {case['title']}")
        for f in fields:
            if not f['passed']:
                print(f"  {f['field']}: 預期 {f['expected']!r}，實際 {f['actual']!r}")
        if error:
            print('  '+error)
    all_fields = [f for r in results for f in r['fields']]
    summary = {'case_passed':sum(r['passed'] for r in results),'case_total':len(results),'field_passed':sum(f['passed'] for f in all_fields),'field_total':len(all_fields)}
    output = base / 'results'
    output.mkdir(exist_ok=True)
    (output/'report.json').write_text(json.dumps({'synthetic':True,'summary':summary,'results':results},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['# 模擬驗收結果','',f"案例通過：{summary['case_passed']}/{summary['case_total']}",f"已標註欄位通過：{summary['field_passed']}/{summary['field_total']}",'','此為模擬案例、部分欄位的精確比對，不代表真實判決準確率。','','| 案例 | 欄位 | 預期 | 實際 |','|---|---|---|---|']
    for r in results:
        for f in r['fields']:
            if not f['passed']:
                lines.append(f"| {r['id']} | {f['field']} | {f['expected']} | {f['actual']} |")
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f"\n案例通過 {summary['case_passed']}/{summary['case_total']}；欄位通過 {summary['field_passed']}/{summary['field_total']}")
    print('報告：',output/'report.md')
    return 0 if summary['case_passed']==summary['case_total'] else 1


if __name__=='__main__':
    sys.exit(main())

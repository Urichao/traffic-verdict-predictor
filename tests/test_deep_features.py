"""在專案根目錄執行：python3 tests/test_deep_features.py"""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parsing.extract_rules import extract_deep_features, extract_features, extract_case, extract_awarded_consolation


class DeepFeaturesTests(unittest.TestCase):
    def test_two_parties(self):
        f = extract_features('原告大學畢業，從事工程師工作，每月收入五萬元；被告高中畢業，從事司機工作，每月收入四萬元。原告經醫囑休養兩個月，勞動能力減損百分之十。')
        expected = dict(plaintiff_education='大學', defendant_education='高中', plaintiff_occupation='工程師', defendant_occupation='司機', plaintiff_monthly_income=50000, defendant_monthly_income=40000, plaintiff_rest_months=2, plaintiff_labor_loss_ratio=0.1)
        for key, value in expected.items():
            self.assertEqual(f[key], value, key)

    def test_missing(self):
        self.assertTrue(all(v is None for v in extract_deep_features('發生車禍。').values()))

    def test_defendant_medical(self):
        f = extract_deep_features('被告醫囑休養兩個月，勞動能力減損10%。')
        self.assertIsNone(f['plaintiff_rest_months'])
        self.assertIsNone(f['plaintiff_labor_loss_ratio'])

    def test_negation_and_family(self):
        for text in ['原告父親月收入五萬元。', '原告月收入不詳。', '原告否認月收入五萬元。', '原告沒有勞動能力減損10%。']:
            self.assertTrue(all(v is None for v in extract_deep_features(text).values()), text)

    def test_conflict_and_multiple(self):
        f = extract_deep_features('原告月收入五萬元。原告月收入六萬元。')
        self.assertIsNone(f['plaintiff_monthly_income'])
        self.assertTrue(all(v is None for v in extract_deep_features('原告二人月收入五萬元。').values()))

    def test_units(self):
        f = extract_deep_features('原告月收入50,000元。原告醫囑休養一個月半，勞動能力減損１２．５％。')
        self.assertEqual(f['plaintiff_monthly_income'], 50000)
        self.assertEqual(f['plaintiff_rest_months'], 1.5)
        self.assertEqual(f['plaintiff_labor_loss_ratio'], .125)
        self.assertIsNone(extract_deep_features('原告年收入六十萬元。')['plaintiff_monthly_income'])
        self.assertIsNone(extract_deep_features('原告醫囑休養30天。')['plaintiff_rest_months'])

    def test_no_inferred_subject_or_doctor(self):
        # 新規格：緊接單一原告背景句的收入可延續歸屬。
        self.assertEqual(extract_deep_features('原告大學畢業。月收入五萬元。')['plaintiff_monthly_income'], 50000)
        self.assertIsNone(extract_deep_features('原告自行休養兩個月。')['plaintiff_rest_months'])
        self.assertEqual(extract_deep_features('我月收入五萬元。')['plaintiff_monthly_income'],50000)

    def test_court_separation(self):
        doc = {'JID':'TPDV,115,訴,1,20260801,1', 'JFULL':'主文\n被告應給付原告100,000元。\n事實及理由\n原告主張車禍受傷。被告抗辯無過失。本院之判斷：原告大學畢業，月收入五萬元。精神慰撫金以100,000元為適當。'}
        f = extract_case(doc)
        self.assertIsNone(f['plaintiff_monthly_income'])
        self.assertEqual(f['court_plaintiff_monthly_income'],50000)
        self.assertEqual(f['awarded_consolation'],100000)

    def test_previous_rules(self):
        self.assertEqual(extract_features('請假扣薪35,000元。')['claim_work_loss'],35000)
        self.assertEqual(extract_features('計程車資兩千元。')['claim_transport'],2000)
        self.assertEqual(extract_features('修車零件八千元、工資兩千元。')['claim_vehicle'],10000)
        self.assertEqual(extract_awarded_consolation('精神慰撫金請求10萬元，應酌減為6萬元。'),60000)
        self.assertIsNone(extract_awarded_consolation('原告請求精神慰撫金10萬元。'))

if __name__ == '__main__':
    unittest.main(verbosity=2)

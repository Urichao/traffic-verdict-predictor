import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parsing.extract_rules import extract_features, extract_deep_features


class BoundaryTests(unittest.TestCase):
    def test_negative_expense(self):
        for text in ['我沒有請假扣薪，醫療費五千元。', '我未請求工作損失5000元。']:
            self.assertEqual(extract_features(text)['claim_work_loss'], 0)

    def test_unknown_transport(self):
        for text in ['交通費金額不清楚，醫療費五千元。', '交通費不詳醫療費五千元。']:
            self.assertEqual(extract_features(text)['claim_transport'], 0)

    def test_thousands_and_other_expenses(self):
        f = extract_features('醫療費35,000元，交通費2,000元。')
        self.assertEqual(f['claim_medical'], 35000)
        self.assertEqual(f['claim_transport'], 2000)

    def test_later_positive(self):
        self.assertEqual(extract_features('先前沒有扣薪。這次扣薪5000元。')['claim_work_loss'], 5000)

    def test_designer(self):
        f = extract_deep_features('被告從事設計師工作。')
        self.assertEqual(f['defendant_occupation'], '設計師')
        self.assertIsNone(f['plaintiff_occupation'])

    def test_subject_switch(self):
        f = extract_deep_features('原告大學畢業。被告高中畢業。月收入四萬元。')
        self.assertIsNone(f['plaintiff_monthly_income'])
        self.assertEqual(f['defendant_monthly_income'], 40000)

    def test_family_and_two_subjects(self):
        for text in ['原告父親大學畢業。月收入五萬元。', '原告大學畢業，被告高中畢業。月收入五萬元。']:
            f = extract_deep_features(text)
            self.assertIsNone(f['plaintiff_monthly_income'])
            self.assertIsNone(f['defendant_monthly_income'])

    def test_context_reset(self):
        for text in ['原告大學畢業。\n月收入五萬元。', '原告大學畢業；月收入五萬元。', '原告大學畢業。發生車禍。月收入五萬元。', '原告主張自己大學畢業。月收入五萬元。']:
            self.assertIsNone(extract_deep_features(text)['plaintiff_monthly_income'], text)

if __name__ == '__main__':
    unittest.main(verbosity=2)
